from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from ..parsing.buckets import c_to_f, f_to_c, parse_temperature_bucket, round_half_up
from ..sources.wunderground import WundergroundCollector
from .continuity import continuity_at, detect_continuity_gaps
from .io import (
    dedupe_weather_observations,
    iter_events,
    latest_before,
    load_events,
    load_events_across_runs,
    resolve_run_dir,
    weather_event_sort_key,
    weather_observation_key,
)
from .meteorology import (
    enrich_surface_payload,
    station_time_features,
    surface_observation_features,
)


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def build_predictive_dataset(cfg: AppConfig, run_id: str | None = None) -> dict[str, Path]:

    run_dir = resolve_run_dir(cfg.data_dir, run_id)
    small_sources = {
        "awc_metar",
        "nws_tgftp_metar",
        "knmi_10m",
        "wis2_metar",
        "wunderground_browser",
        "polymarket_ws",
        "awc_peer_metar",
        "wis2_peer_metar",
        "system",
        *{
            f"forecast_{provider}"
            for station in cfg.stations.values()
            for provider in station.forecast_providers
        },
    }


    events = load_events(run_dir, small_sources)
    cutoff_ns = max((event.get("received_wall_ns", 0) for event in events), default=0)
    weather_context = load_events_across_runs(
        cfg.data_dir,
        {"awc_metar", "nws_tgftp_metar", "knmi_10m", "wis2_metar"},
        received_at_or_before_ns=cutoff_ns,
    )
    peer_weather_context = load_events_across_runs(
        cfg.data_dir,
        {"awc_peer_metar", "wis2_peer_metar"},
        received_at_or_before_ns=cutoff_ns,
    )
    forecast_context = load_events_across_runs(
        cfg.data_dir,
        {
            f"forecast_{provider}"
            for station in cfg.stations.values()
            for provider in station.forecast_providers
        },
        received_at_or_before_ns=cutoff_ns,
    )
    wu_label_context = load_events_across_runs(cfg.data_dir, {"wunderground_browser"})
    feature_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    revision_reaction_rows: list[dict[str, Any]] = []

    gaps = detect_continuity_gaps(
        events,
        max_gap_seconds=cfg.analysis.continuity_gap_seconds,
        resume_warmup_seconds=cfg.analysis.resume_warmup_seconds,
    )
    ws_health = sorted(
        (
            event
            for event in events
            if event.get("source") == "polymarket_ws"
            and event.get("event_type")
            in {"subscribed", "heartbeat_rtt", "connection_error", "collector_error"}
        ),
        key=lambda event: event["received_wall_ns"],
    )

    station_weather: dict[str, list[dict[str, Any]]] = {}
    current_keys_by_station: dict[str, set[tuple[Any, ...]]] = {}
    timed_decisions_by_station: dict[
        str, list[tuple[tuple[Any, ...], str, int, str, str]]
    ] = {}
    revision_decisions_by_station: dict[
        str, list[tuple[dict[str, Any], int, tuple[Any, ...], str, int]]
    ] = {}
    decisions: list[tuple[tuple[Any, ...], str, str, int]] = []
    for station in cfg.stations.values():
        weather = dedupe_weather_observations(
            [
                event
                for event in weather_context
                if event.get("station_id") == station.station_id
                and event.get("event_type") == "weather_observation"
            ]
        )
        weather.sort(key=weather_event_sort_key)
        station_weather[station.station_id] = weather
        current_weather = [
            event
            for event in events
            if event.get("station_id") == station.station_id
            and event.get("event_type") == "weather_observation"
            and event.get("source")
            in {"awc_metar", "nws_tgftp_metar", "knmi_10m", "wis2_metar"}
            and event.get("payload", {}).get("backfill") is not True
        ]
        current_keys = {_event_identity(event) for event in current_weather}
        current_keys_by_station[station.station_id] = current_keys
        for observation in weather:
            identity = _event_identity(observation)
            if identity not in current_keys:
                continue
            decisions.append(
                (
                    identity,
                    station.station_id,
                    _observation_local_datetime(observation, station).date().isoformat(),
                    int(observation["received_wall_ns"]),
                )
            )
        timed_decisions: list[tuple[tuple[Any, ...], str, int, str, str]] = []
        if current_weather and cutoff_ns:
            interval_ns = cfg.predictive.scheduled_interval_minutes * 60 * 1_000_000_000
            first_ns = min(int(event["received_wall_ns"]) for event in current_weather)
            schedule_ns = ((first_ns + interval_ns - 1) // interval_ns) * interval_ns
            timezone = ZoneInfo(station.timezone)
            while schedule_ns <= cutoff_ns:
                local_date = datetime.fromtimestamp(schedule_ns / 1e9, tz=UTC).astimezone(
                    timezone
                ).date().isoformat()
                identity = _scheduled_identity(run_dir.name, station.station_id, schedule_ns)
                timed_decisions.append(
                    (identity, local_date, schedule_ns, "scheduled", "timer")
                )
                decisions.append((identity, station.station_id, local_date, schedule_ns))
                schedule_ns += interval_ns
        timezone = ZoneInfo(station.timezone)
        for forecast_event in events:
            source = str(forecast_event.get("source") or "")
            if (
                forecast_event.get("station_id") != station.station_id
                or forecast_event.get("event_type") != "forecast_snapshot"
                or not source.startswith("forecast_")
                or forecast_event.get("payload", {}).get("backfill") is True
            ):
                continue
            forecast_ns = int(forecast_event["received_wall_ns"])
            local_date = datetime.fromtimestamp(forecast_ns / 1e9, tz=UTC).astimezone(
                timezone
            ).date().isoformat()
            identity = _forecast_identity(
                run_dir.name, station.station_id, source, forecast_ns
            )
            timed_decisions.append(
                (identity, local_date, forecast_ns, "forecast_update", source)
            )
            decisions.append((identity, station.station_id, local_date, forecast_ns))
        timed_decisions_by_station[station.station_id] = timed_decisions
        revision_decisions: list[
            tuple[dict[str, Any], int, tuple[Any, ...], str, int]
        ] = []
        for revision_event in events:
            source = str(revision_event.get("source") or "")
            if (
                revision_event.get("station_id") != station.station_id
                or revision_event.get("event_type") != "forecast_revision"
                or not source.startswith("forecast_")
            ):
                continue
            revision_ns = int(revision_event["received_wall_ns"])
            local_date = str(revision_event.get("payload", {}).get("local_date") or "")
            if not local_date:
                local_date = datetime.fromtimestamp(revision_ns / 1e9, tz=UTC).astimezone(
                    timezone
                ).date().isoformat()
            for horizon_seconds in (0, *cfg.execution.markout_seconds):
                decision_ns = revision_ns + horizon_seconds * 1_000_000_000
                if decision_ns > cutoff_ns:
                    continue
                identity = _forecast_reaction_identity(
                    run_dir.name,
                    station.station_id,
                    source,
                    revision_ns,
                    horizon_seconds,
                )
                revision_decisions.append(
                    (revision_event, horizon_seconds, identity, local_date, decision_ns)
                )
                decisions.append((identity, station.station_id, local_date, decision_ns))
        revision_decisions_by_station[station.station_id] = revision_decisions

    market_snapshots, current_labels = _stream_market_quote_snapshots(run_dir, decisions)


    all_labels = current_labels

    continuity_excluded_rows = 0
    run_start_ns = min(
        (int(event.get("received_wall_ns") or 0) for event in events),
        default=0,
    )
    startup_warmup_until_ns = run_start_ns + int(
        cfg.analysis.resume_warmup_seconds * 1e9
    )

    for station in cfg.stations.values():
        weather = station_weather[station.station_id]
        forecasts = [
            e
            for e in forecast_context
            if e.get("station_id") == station.station_id
            and str(e.get("source") or "").startswith("forecast_")
            and e.get("event_type") == "forecast_snapshot"
        ]
        peer_weather = [
            e
            for e in peer_weather_context
            if e.get("station_id") == station.station_id
            and e.get("source") in {"awc_peer_metar", "wis2_peer_metar"}
            and e.get("event_type") == "peer_weather_observation"
            and e.get("payload", {}).get("backfill") is not True
        ]
        peers_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in peer_weather:
            peer_id = str(event.get("payload", {}).get("peer_station_id") or "")
            if peer_id:
                peers_by_id[peer_id].append(event)
        for peer_events in peers_by_id.values():
            peer_events.sort(key=lambda e: e["received_wall_ns"])
        forecasts_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in forecasts:
            forecasts_by_source[str(event["source"])].append(event)
        for source_events in forecasts_by_source.values():
            source_events.sort(key=lambda e: e["received_wall_ns"])

        wu = [
            e
            for e in wu_label_context
            if e.get("station_id") == station.station_id
            and e.get("source") == "wunderground_browser"
            and e.get("event_type") == "resolution_snapshot"
        ]
        wu_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for snapshot in wu:
            snapshot_date = str(snapshot.get("payload", {}).get("local_date") or "")
            if snapshot_date:
                wu_by_date[snapshot_date].append(snapshot)
        final_max_by_date = {
            local_date: _final_wu_max(snapshots) for local_date, snapshots in wu_by_date.items()
        }
        final_label_by_date = {
            local_date: _winning_label_from_labels(
                final_max,
                station,
                all_labels.get((station.station_id, local_date), set()),
            )
            if final_max is not None
            else None
            for local_date, final_max in final_max_by_date.items()
        }

        revision_groups: defaultdict[tuple[str, int], list[tuple[Any, ...]]] = defaultdict(
            list
        )
        for revision_event, horizon_seconds, identity, local_date, decision_ns in (
            revision_decisions_by_station[station.station_id]
        ):
            revision_groups[
                (
                    str(revision_event.get("source") or ""),
                    int(revision_event["received_wall_ns"]),
                )
            ].append(
                (revision_event, horizon_seconds, identity, local_date, decision_ns)
            )
        for (_, revision_ns), revision_points in revision_groups.items():
            revision_event = revision_points[0][0]
            payload = revision_event.get("payload", {})
            local_date = revision_points[0][3]
            labels = all_labels.get((station.station_id, local_date), set())
            previous_market_max = _num(payload.get("previous_market_max"))
            current_market_max = _num(payload.get("current_market_max"))
            previous_label = (
                _winning_label_from_labels(previous_market_max, station, labels)
                if previous_market_max is not None
                else None
            )
            current_label = (
                _winning_label_from_labels(current_market_max, station, labels)
                if current_market_max is not None
                else None
            )
            economic_bucket_changed = bool(
                previous_label and current_label and previous_label != current_label
            )
            t0_identity = next(
                identity
                for _, horizon, identity, _, _ in revision_points
                if horizon == 0
            )
            t0_quotes = market_snapshots.get(t0_identity, _empty_market_quotes())
            t0_ask = t0_quotes["ask"].get(current_label) if current_label else None
            t0_mid = t0_quotes["mid"].get(current_label) if current_label else None
            t0_market_ws_healthy = _ws_healthy_at(ws_health, revision_ns, cfg)
            for _, horizon_seconds, identity, _, decision_ns in sorted(
                revision_points, key=lambda item: item[1]
            ):
                quotes = market_snapshots.get(identity, _empty_market_quotes())
                mid = quotes["mid"].get(current_label) if current_label else None
                bid = quotes["bid"].get(current_label) if current_label else None
                ask = quotes["ask"].get(current_label) if current_label else None
                probabilities = _normalize(quotes["mid"])
                final_label = final_label_by_date.get(local_date)
                revision_reaction_rows.append(
                    {
                        "run_id": run_dir.name,
                        "station_id": station.station_id,
                        "city": station.city,
                        "local_date": local_date,
                        "provider": payload.get("provider")
                        or str(revision_event.get("source") or "").removeprefix("forecast_"),
                        "source": revision_event.get("source"),
                        "revision_received_wall_ns": revision_ns,
                        "revision_received_utc": datetime.fromtimestamp(
                            revision_ns / 1e9, tz=UTC
                        ).isoformat(),
                        "horizon_seconds": horizon_seconds,
                        "quote_timestamp_ns": decision_ns,
                        "previous_content_sha256": payload.get("previous_content_sha256"),
                        "content_sha256": payload.get("content_sha256"),
                        "previous_today_max_c": payload.get("previous_today_max_c"),
                        "current_today_max_c": payload.get("current_today_max_c"),
                        "today_max_delta_c": payload.get("today_max_delta_c"),
                        "previous_remaining_max_c": payload.get("previous_remaining_max_c"),
                        "current_remaining_max_c": payload.get("current_remaining_max_c"),
                        "previous_bucket_value": payload.get("previous_bucket_value"),
                        "current_bucket_value": payload.get("current_bucket_value"),
                        "previous_bucket_label": previous_label,
                        "current_bucket_label": current_label,
                        "rounded_degree_changed": payload.get("bucket_changed") is True,
                        "economic_bucket_changed": economic_bucket_changed,


                        "bucket_changed": economic_bucket_changed,
                        "market_mid": mid,
                        "market_bid": bid,
                        "market_ask": ask,
                        "market_bid_size": (
                            quotes["bid_size"].get(current_label) if current_label else None
                        ),
                        "market_ask_size": (
                            quotes["ask_size"].get(current_label) if current_label else None
                        ),
                        "market_probability": probabilities.get(current_label, 0.0)
                        if current_label
                        else None,
                        "market_mid_change_from_t0": (
                            mid - t0_mid if mid is not None and t0_mid is not None else None
                        ),
                        "market_ask_change_from_t0": (
                            ask - t0_ask if ask is not None and t0_ask is not None else None
                        ),
                        "mid_markout_from_t0_ask": (
                            mid - t0_ask if mid is not None and t0_ask is not None else None
                        ),
                        "executable_bid_markout_from_t0_ask": (
                            bid - t0_ask if bid is not None and t0_ask is not None else None
                        ),
                        "fee_rate": (
                            t0_quotes["fee_rate"].get(current_label)
                            if current_label
                            else None
                        ),
                        "fee_exponent": (
                            t0_quotes["fee_exponent"].get(current_label)
                            if current_label
                            else None
                        ),
                        "fees_enabled": (
                            t0_quotes["fees_enabled"].get(current_label)
                            if current_label
                            else None
                        ),
                        "market_quote_coverage": quotes["coverage"].get("ratio", 0.0),
                        "t0_market_ws_healthy": t0_market_ws_healthy,
                        "market_ws_healthy": _ws_healthy_at(ws_health, decision_ns, cfg),
                        "final_wu_max": final_max_by_date.get(local_date),
                        "winning_bucket": final_label,
                        "revision_correct": bool(
                            current_label and final_label and current_label == final_label
                        )
                        if final_label
                        else None,
                    }
                )

        histories: dict[str, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=500))
        running_max_by_date: dict[str, float] = {}
        earliest_minute_by_date: dict[str, int] = {}
        current_event_keys = current_keys_by_station[station.station_id]
        for obs in weather:
            observation_payload = enrich_surface_payload(obs["payload"])
            temp_c = _num(observation_payload.get("temperature_c"))
            if temp_c is None:
                continue
            temp_market = c_to_f(temp_c) if station.market_unit == "F" else temp_c
            ts = obs["received_wall_ns"]
            local_dt = _observation_local_datetime(obs, station)
            local_date = local_dt.date().isoformat()
            local_minute = local_dt.hour * 60 + local_dt.minute
            earliest_minute_by_date[local_date] = min(
                local_minute, earliest_minute_by_date.get(local_date, local_minute)
            )
            running_max = max(running_max_by_date.get(local_date, -math.inf), temp_market)
            running_max_by_date[local_date] = running_max
            running_max_c = f_to_c(running_max) if station.market_unit == "F" else running_max
            history = histories[local_date]
            history.append((ts, temp_market))
            event_key = _event_identity(obs)
            if event_key not in current_event_keys:
                continue
            ws_healthy = _ws_healthy_at(ws_health, ts, cfg)
            quotes = market_snapshots.get(event_key, _empty_market_quotes())
            if not ws_healthy:
                quotes = _empty_market_quotes()
            continuity_segment, continuity_eligible = continuity_at(ts, gaps)
            continuity_eligible = bool(
                continuity_eligible and ts >= startup_warmup_until_ns
            )
            if not continuity_eligible:
                continuity_excluded_rows += 1
            model_features = _forecast_features_at(
                forecasts_by_source,
                ts,
                station,
                current_temperature_c=temp_c,
                running_max_c=running_max_c,
            )
            peer_features = _peer_features_at(
                peers_by_id,
                ts,
                station,
                target_temperature_c=temp_c,
                target_wind_direction_deg=_num(
                    observation_payload.get("wind_direction_deg")
                ),
                target_wind_speed_kt=_num(
                    observation_payload.get("wind_speed_kt")
                    or observation_payload.get("wind_speed")
                ),
                max_age_minutes=cfg.predictive.peer_max_age_minutes,
            )
            observation_time = str(observation_payload.get("observation_time") or "")
            physical_features = {
                **station_time_features(local_dt, station),
                **surface_observation_features(observation_payload),
            }
            final_max = final_max_by_date.get(local_date)
            final_label = final_label_by_date.get(local_date)
            resolution_running_max = _running_wu_max_at(
                wu_by_date.get(local_date, []), ts
            )
            row = {
                "run_id": obs["run_id"],
                "station_id": station.station_id,
                "city": station.city,
                "source": obs["source"],
                "decision_mode": "event_driven",
                "trigger_source": obs["source"],
                "received_wall_ns": ts,
                "observation_time": observation_time,
                "observation_key": weather_observation_key(obs),
                "local_date": local_date,
                "day_history_complete": (
                    earliest_minute_by_date.get(local_date, 24 * 60)
                    <= cfg.analysis.max_day_start_lag_minutes
                ),
                "continuity_segment": continuity_segment,
                "continuity_eligible": continuity_eligible,
                "local_hour": local_dt.hour + local_dt.minute / 60.0,
                "minutes_since_local_midnight": local_dt.hour * 60 + local_dt.minute,
                "minutes_since_last_observation": 0.0,
                "temperature_c": temp_c,
                "temperature_market_unit": temp_market,
                "running_max_market_unit": running_max,
                "resolution_running_max_market_unit": resolution_running_max,
                "running_max_source": "wunderground_resolution_page",
                "running_max_c": running_max_c,
                "dewpoint_c": observation_payload.get("dewpoint_c"),
                "wind_direction_deg": observation_payload.get("wind_direction_deg"),
                "wind_speed_kt": observation_payload.get("wind_speed_kt")
                or observation_payload.get("wind_speed"),
                "wind_gust_kt": observation_payload.get("wind_gust_kt"),
                "pressure": observation_payload.get("altimeter_hpa")
                or observation_payload.get("pressure_hpa")
                or observation_payload.get("pressure"),
                "global_radiation": observation_payload.get("global_radiation"),
                "slope_10m": _slope(history, ts, 10 * 60),
                "slope_30m": _slope(history, ts, 30 * 60),
                "slope_60m": _slope(history, ts, 60 * 60),
                "market_distribution_json": json.dumps(quotes["mid"], sort_keys=True),
                "market_labels_json": json.dumps(
                    sorted(
                        all_labels.get(
                            (station.station_id, local_date), set()
                        )
                    )
                ),
                "market_bid_json": json.dumps(quotes["bid"], sort_keys=True),
                "market_ask_json": json.dumps(quotes["ask"], sort_keys=True),
                "market_bid_size_json": json.dumps(quotes["bid_size"], sort_keys=True),
                "market_ask_size_json": json.dumps(quotes["ask_size"], sort_keys=True),
                "market_fee_rate_json": json.dumps(quotes["fee_rate"], sort_keys=True),
                "market_fee_exponent_json": json.dumps(quotes["fee_exponent"], sort_keys=True),
                "market_fees_enabled_json": json.dumps(quotes["fees_enabled"], sort_keys=True),
                "market_ws_healthy": ws_healthy,
                "market_quote_coverage": quotes["coverage"].get("ratio", 0.0),
                "final_wu_max": final_max,
                "winning_bucket": final_label,
                **physical_features,
                **model_features,
                **peer_features,
            }
            feature_rows.append(row)
            if continuity_eligible and final_label and quotes["mid"]:
                normalized = _normalize(quotes["mid"])
                score_rows.append(
                    {
                        "station_id": station.station_id,
                        "received_wall_ns": ts,
                        "local_date": local_date,
                        "decision_mode": "event_driven",
                        "winning_bucket": final_label,
                        "market_brier": _multiclass_brier(normalized, final_label),
                        "market_log_loss": -math.log(max(normalized.get(final_label, 0.0), 1e-12)),
                        "winning_bucket_probability": normalized.get(final_label, 0.0),
                        "distribution_json": json.dumps(normalized, sort_keys=True),
                    }
                )


        received_order = sorted(weather, key=lambda event: int(event["received_wall_ns"]))
        for event_key, local_date, ts, decision_mode, trigger_source in sorted(
            timed_decisions_by_station[station.station_id], key=lambda item: item[2]
        ):
            known = [
                event
                for event in received_order
                if int(event["received_wall_ns"]) <= ts
                and _observation_local_datetime(event, station).date().isoformat() == local_date
                and _num(event.get("payload", {}).get("temperature_c")) is not None
            ]
            if not known:
                continue
            latest_observation = known[-1]
            latest_payload = enrich_surface_payload(latest_observation.get("payload", {}))
            temp_c = _num(latest_payload.get("temperature_c"))
            if temp_c is None:
                continue
            temp_market = c_to_f(temp_c) if station.market_unit == "F" else temp_c
            known_temperatures = [
                (
                    int(event["received_wall_ns"]),
                    c_to_f(float(event["payload"]["temperature_c"]))
                    if station.market_unit == "F"
                    else float(event["payload"]["temperature_c"]),
                )
                for event in known[-500:]
            ]
            history = deque(known_temperatures, maxlen=500)
            running_max = max(value for _, value in known_temperatures)
            running_max_c = f_to_c(running_max) if station.market_unit == "F" else running_max
            earliest_local = min(_observation_local_datetime(event, station) for event in known)
            local_dt = datetime.fromtimestamp(ts / 1e9, tz=UTC).astimezone(
                ZoneInfo(station.timezone)
            )
            ws_healthy = _ws_healthy_at(ws_health, ts, cfg)
            quotes = market_snapshots.get(event_key, _empty_market_quotes())
            if not ws_healthy:
                quotes = _empty_market_quotes()
            continuity_segment, continuity_eligible = continuity_at(ts, gaps)
            continuity_eligible = bool(
                continuity_eligible and ts >= startup_warmup_until_ns
            )
            if not continuity_eligible:
                continuity_excluded_rows += 1
            model_features = _forecast_features_at(
                forecasts_by_source,
                ts,
                station,
                current_temperature_c=temp_c,
                running_max_c=running_max_c,
            )
            peer_features = _peer_features_at(
                peers_by_id,
                ts,
                station,
                target_temperature_c=temp_c,
                target_wind_direction_deg=_num(latest_payload.get("wind_direction_deg")),
                target_wind_speed_kt=_num(
                    latest_payload.get("wind_speed_kt") or latest_payload.get("wind_speed")
                ),
                max_age_minutes=cfg.predictive.peer_max_age_minutes,
            )
            final_max = final_max_by_date.get(local_date)
            final_label = final_label_by_date.get(local_date)
            resolution_running_max = _running_wu_max_at(
                wu_by_date.get(local_date, []), ts
            )
            physical_features = {
                **station_time_features(local_dt, station),
                **surface_observation_features(latest_payload),
            }
            row = {
                "run_id": run_dir.name,
                "station_id": station.station_id,
                "city": station.city,
                "source": trigger_source,
                "decision_mode": decision_mode,
                "trigger_source": trigger_source,
                "received_wall_ns": ts,
                "observation_time": str(latest_payload.get("observation_time") or ""),
                "observation_key": event_key[2],
                "local_date": local_date,
                "day_history_complete": (
                    earliest_local.hour * 60 + earliest_local.minute
                    <= cfg.analysis.max_day_start_lag_minutes
                ),
                "continuity_segment": continuity_segment,
                "continuity_eligible": continuity_eligible,
                "local_hour": local_dt.hour + local_dt.minute / 60.0,
                "minutes_since_local_midnight": local_dt.hour * 60 + local_dt.minute,
                "minutes_since_last_observation": (
                    ts - int(latest_observation["received_wall_ns"])
                )
                / 60e9,
                "temperature_c": temp_c,
                "temperature_market_unit": temp_market,
                "running_max_market_unit": running_max,
                "resolution_running_max_market_unit": resolution_running_max,
                "running_max_source": "wunderground_resolution_page",
                "running_max_c": running_max_c,
                "dewpoint_c": latest_payload.get("dewpoint_c"),
                "wind_direction_deg": latest_payload.get("wind_direction_deg"),
                "wind_speed_kt": latest_payload.get("wind_speed_kt")
                or latest_payload.get("wind_speed"),
                "wind_gust_kt": latest_payload.get("wind_gust_kt"),
                "pressure": latest_payload.get("altimeter_hpa")
                or latest_payload.get("pressure_hpa")
                or latest_payload.get("pressure"),
                "global_radiation": latest_payload.get("global_radiation"),
                "slope_10m": _slope(history, ts, 10 * 60),
                "slope_30m": _slope(history, ts, 30 * 60),
                "slope_60m": _slope(history, ts, 60 * 60),
                "market_distribution_json": json.dumps(quotes["mid"], sort_keys=True),
                "market_labels_json": json.dumps(
                    sorted(
                        all_labels.get(
                            (station.station_id, local_date), set()
                        )
                    )
                ),
                "market_bid_json": json.dumps(quotes["bid"], sort_keys=True),
                "market_ask_json": json.dumps(quotes["ask"], sort_keys=True),
                "market_bid_size_json": json.dumps(quotes["bid_size"], sort_keys=True),
                "market_ask_size_json": json.dumps(quotes["ask_size"], sort_keys=True),
                "market_fee_rate_json": json.dumps(quotes["fee_rate"], sort_keys=True),
                "market_fee_exponent_json": json.dumps(quotes["fee_exponent"], sort_keys=True),
                "market_fees_enabled_json": json.dumps(
                    quotes["fees_enabled"], sort_keys=True
                ),
                "market_ws_healthy": ws_healthy,
                "market_quote_coverage": quotes["coverage"].get("ratio", 0.0),
                "final_wu_max": final_max,
                "winning_bucket": final_label,
                **physical_features,
                **model_features,
                **peer_features,
            }
            feature_rows.append(row)
            if continuity_eligible and final_label and quotes["mid"]:
                normalized = _normalize(quotes["mid"])
                score_rows.append(
                    {
                        "station_id": station.station_id,
                        "received_wall_ns": ts,
                        "local_date": local_date,
                        "decision_mode": decision_mode,
                        "winning_bucket": final_label,
                        "market_brier": _multiclass_brier(normalized, final_label),
                        "market_log_loss": -math.log(
                            max(normalized.get(final_label, 0.0), 1e-12)
                        ),
                        "winning_bucket_probability": normalized.get(final_label, 0.0),
                        "distribution_json": json.dumps(normalized, sort_keys=True),
                    }
                )

    report_dir = cfg.reports_dir / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_path = report_dir / "predictive_features.csv"
    score_path = report_dir / "market_baseline_scores.csv"
    revision_reaction_path = report_dir / "forecast_revision_reactions.csv"
    _write_csv(feature_path, feature_rows)
    _write_csv(score_path, score_rows)
    _write_csv(revision_reaction_path, revision_reaction_rows)
    station_feature_coverage = _station_feature_coverage(feature_rows, cfg)
    summary_path = report_dir / "predictive_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "feature_rows": len(feature_rows),
                "event_driven_rows": sum(
                    row.get("decision_mode") == "event_driven" for row in feature_rows
                ),
                "scheduled_rows": sum(
                    row.get("decision_mode") == "scheduled" for row in feature_rows
                ),
                "forecast_update_rows": sum(
                    row.get("decision_mode") == "forecast_update" for row in feature_rows
                ),
                "scored_rows": len(score_rows),
                "continuity_gaps": len(gaps),
                "continuity_excluded_rows": continuity_excluded_rows,
                "station_feature_coverage": station_feature_coverage,
                "streaming_order_books": True,
                "note": (
                    "Leakage-safe live feature dataset. Forecast snapshots are joined only when "
                    "their local first-seen time precedes the decision timestamp."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    continuity_path = report_dir / "continuity_report.json"
    continuity_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "gap_threshold_seconds": cfg.analysis.continuity_gap_seconds,
                "resume_warmup_seconds": cfg.analysis.resume_warmup_seconds,
                "gaps": [gap.as_dict() for gap in gaps],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    revision_t0 = [
        row for row in revision_reaction_rows if row.get("horizon_seconds") == 0
    ]
    revision_summary_path = report_dir / "forecast_revision_summary.json"
    revision_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "revision_events": len(revision_t0),
                "rounded_degree_change_events": sum(
                    row.get("rounded_degree_changed") is True for row in revision_t0
                ),
                "economic_bucket_change_events": sum(
                    row.get("bucket_changed") is True for row in revision_t0
                ),
                "t0_executable_quotes": sum(
                    _num(row.get("market_ask")) is not None
                    and _num(row.get("market_ask_size")) is not None
                    for row in revision_t0
                ),
                "finalized_revisions": sum(
                    row.get("revision_correct") is not None for row in revision_t0
                ),
                "correct_revisions": sum(
                    row.get("revision_correct") is True for row in revision_t0
                ),
                "reaction_rows": len(revision_reaction_rows),
                "horizons_seconds": [0, *cfg.execution.markout_seconds],
                "note": (
                    "Forecast revisions are content-deduplicated. Economic bucket changes use "
                    "the actual Polymarket label, so adjacent Fahrenheit degrees inside one "
                    "two-degree contract are not signals. Reaction rows remain descriptive "
                    "until the station-day has a final WU label."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "features": feature_path,
        "scores": score_path,
        "forecast_reactions": revision_reaction_path,
        "forecast_revision_summary": revision_summary_path,
        "summary": summary_path,
        "continuity": continuity_path,
    }


def _event_identity(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("run_id"),
        event.get("received_wall_ns"),
        weather_observation_key(event),
    )


def _forecast_identity(
    run_id: str, station_id: str, source: str, received_wall_ns: int
) -> tuple[Any, ...]:
    return (run_id, received_wall_ns, f"forecast:{station_id}:{source}:{received_wall_ns}")


def _forecast_reaction_identity(
    run_id: str,
    station_id: str,
    source: str,
    revision_ns: int,
    horizon_seconds: int,
) -> tuple[Any, ...]:
    timestamp_ns = revision_ns + horizon_seconds * 1_000_000_000
    return (
        run_id,
        timestamp_ns,
        f"forecast-reaction:{station_id}:{source}:{revision_ns}:{horizon_seconds}",
    )


def _scheduled_identity(run_id: str, station_id: str, timestamp_ns: int) -> tuple[Any, ...]:
    return (run_id, timestamp_ns, f"scheduled:{station_id}:{timestamp_ns}")


def _stream_market_quote_snapshots(
    run_dir: Path,
    decisions: list[tuple[tuple[Any, ...], str, str, int]],
) -> tuple[
    dict[tuple[Any, ...], dict[str, Any]],
    defaultdict[tuple[str, str], set[str]],
]:

    ordered = sorted(decisions, key=lambda item: item[3])
    latest: defaultdict[tuple[str, str], dict[str, tuple[Any, ...]]] = defaultdict(dict)
    labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    raw_snapshots: dict[tuple[Any, ...], dict[str, tuple[Any, ...]]] = {}
    decision_context: dict[tuple[Any, ...], tuple[str, str]] = {}
    decision_index = 0

    def capture_before(timestamp_ns: int | None) -> None:
        nonlocal decision_index
        while decision_index < len(ordered):
            identity, station_id, local_date, decision_ns = ordered[decision_index]
            if timestamp_ns is not None and decision_ns >= timestamp_ns:
                break
            raw_snapshots[identity] = dict(latest.get((station_id, local_date), {}))
            decision_context[identity] = (station_id, local_date)
            decision_index += 1

    for event in iter_events(run_dir, {"polymarket_book"}):
        if event.get("event_type") != "book_state":
            continue
        event_ns = int(event.get("received_wall_ns") or 0)
        capture_before(event_ns)
        station_id = str(event.get("station_id") or "")
        payload = event.get("payload", {})
        event_date = str(payload.get("event_date") or "")
        label = str(payload.get("bucket_label") or "")
        if not station_id or not event_date or not label:
            continue
        key = (station_id, event_date)
        labels[key].add(label)
        latest[key][label] = (
            _num(payload.get("mid")),
            _num(payload.get("best_bid")),
            _num(payload.get("best_ask")),
            _best_level_size(payload, "bids", _num(payload.get("best_bid"))),
            _best_level_size(payload, "asks", _num(payload.get("best_ask"))),
            _num(payload.get("fee_rate")),
            _num(payload.get("fee_exponent")),
            1.0 if payload.get("fees_enabled") else 0.0,
        )
    capture_before(None)

    snapshots = {
        identity: _quotes_from_latest(
            states,
            labels.get(decision_context[identity], set()),
        )
        for identity, states in raw_snapshots.items()
    }
    return snapshots, labels


def _quotes_from_latest(
    latest: dict[str, tuple[Any, ...]], expected_labels: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mid": {label: values[0] for label, values in latest.items() if values[0] is not None},
        "bid": {label: values[1] for label, values in latest.items() if values[1] is not None},
        "ask": {label: values[2] for label, values in latest.items() if values[2] is not None},
        "bid_size": {
            label: values[3] for label, values in latest.items() if values[3] is not None
        },
        "ask_size": {
            label: values[4] for label, values in latest.items() if values[4] is not None
        },
        "fee_rate": {
            label: values[5] for label, values in latest.items() if values[5] is not None
        },
        "fee_exponent": {
            label: values[6] for label, values in latest.items() if values[6] is not None
        },
        "fees_enabled": {label: values[7] for label, values in latest.items()},
    }
    quoted_labels = set(result["mid"]) & set(result["ask"])
    result["coverage"] = {
        "ratio": len(quoted_labels) / len(expected_labels) if expected_labels else 0.0
    }
    return result


def _best_level_size(
    payload: dict[str, Any], side: str, best_price: float | None
) -> float | None:
    if best_price is None:
        return None
    levels = payload.get(side)
    if not isinstance(levels, list):
        return None
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = _num(level.get("price"))
        size = _num(level.get("size"))
        if price is not None and size is not None and abs(price - best_price) <= 1e-12:
            return size
    return None


def _observation_local_datetime(event: dict[str, Any], station: StationConfig) -> datetime:
    raw = event.get("payload", {}).get("observation_time")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(ZoneInfo(station.timezone))
        except ValueError:
            pass
    return datetime.fromtimestamp(event["received_wall_ns"] / 1e9, tz=UTC).astimezone(
        ZoneInfo(station.timezone)
    )


def _market_quotes_at(
    books: list[dict[str, Any]], timestamp_ns: int
) -> dict[str, dict[str, float]]:
    latest: dict[
        str,
        tuple[
            int,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float,
        ],
    ] = {}
    for event in books:
        if event["received_wall_ns"] > timestamp_ns:
            break
        label = str(event["payload"].get("bucket_label") or "")
        if not label:
            continue
        latest[label] = (
            event["received_wall_ns"],
            _num(event["payload"].get("mid")),
            _num(event["payload"].get("best_bid")),
            _num(event["payload"].get("best_ask")),
            _best_level_size(
                event["payload"], "bids", _num(event["payload"].get("best_bid"))
            ),
            _best_level_size(
                event["payload"], "asks", _num(event["payload"].get("best_ask"))
            ),
            _num(event["payload"].get("fee_rate")),
            _num(event["payload"].get("fee_exponent")),
            1.0 if event["payload"].get("fees_enabled") else 0.0,
        )
    expected_labels = {
        str(event.get("payload", {}).get("bucket_label") or "")
        for event in books
        if event.get("payload", {}).get("bucket_label")
    }
    result = {
        "mid": {label: values[1] for label, values in latest.items() if values[1] is not None},
        "bid": {label: values[2] for label, values in latest.items() if values[2] is not None},
        "ask": {label: values[3] for label, values in latest.items() if values[3] is not None},
        "bid_size": {
            label: values[4] for label, values in latest.items() if values[4] is not None
        },
        "ask_size": {
            label: values[5] for label, values in latest.items() if values[5] is not None
        },
        "fee_rate": {label: values[6] for label, values in latest.items() if values[6] is not None},
        "fee_exponent": {
            label: values[7] for label, values in latest.items() if values[7] is not None
        },
        "fees_enabled": {label: values[8] for label, values in latest.items()},
    }
    quoted_labels = set(result["mid"]) & set(result["ask"])
    result["coverage"] = {
        "ratio": len(quoted_labels) / len(expected_labels) if expected_labels else 0.0
    }
    return result


def _empty_market_quotes() -> dict[str, dict[str, float]]:
    return {
        "mid": {},
        "bid": {},
        "ask": {},
        "bid_size": {},
        "ask_size": {},
        "fee_rate": {},
        "fee_exponent": {},
        "fees_enabled": {},
        "coverage": {"ratio": 0.0},
    }


def _ws_healthy_at(events: list[dict[str, Any]], timestamp_ns: int, cfg: AppConfig) -> bool:
    event = latest_before(events, timestamp_ns)
    if event is None or event.get("event_type") not in {"subscribed", "heartbeat_rtt"}:
        return False
    return (
        timestamp_ns - event["received_wall_ns"]
        <= cfg.execution.max_ws_health_age_seconds * 1_000_000_000
    )


def _forecast_features_at(
    forecasts_by_source: dict[str, list[dict[str, Any]]],
    timestamp_ns: int,
    station: StationConfig,
    *,
    current_temperature_c: float | None = None,
    running_max_c: float | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    provider_maxima: list[float] = []
    provider_remaining_maxima: list[float] = []
    decision_local = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC).astimezone(
        ZoneInfo(station.timezone)
    )
    for source, events in forecasts_by_source.items():
        snapshot = latest_before(events, timestamp_ns)
        if snapshot is None:
            continue
        provider = source.removeprefix("forecast_")
        prefix = f"model_{provider}"
        payload = snapshot.get("payload") or {}
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        forecast = payload.get("forecast") if isinstance(payload, dict) else {}
        output[f"{prefix}_age_minutes"] = (timestamp_ns - snapshot["received_wall_ns"]) / 60e9
        hourly = forecast.get("hourly") if isinstance(forecast, dict) else None
        today_max, remaining_max = (
            _forecast_maxima_at(hourly, decision_local, station)
            if isinstance(hourly, dict)
            else (None, None)
        )
        if today_max is None:
            today_max = _num((summary or {}).get("today_hourly_max_c"))
        if remaining_max is None:
            remaining_max = _num((summary or {}).get("remaining_hourly_max_c"))
        output[f"{prefix}_today_max_c"] = today_max
        output[f"{prefix}_remaining_max_c"] = remaining_max
        if today_max is not None:
            provider_maxima.append(today_max)
            if current_temperature_c is not None:
                output[f"{prefix}_today_max_minus_current_c"] = (
                    today_max - current_temperature_c
                )
            if running_max_c is not None:
                output[f"{prefix}_today_max_minus_running_max_c"] = today_max - running_max_c
        if remaining_max is not None:
            provider_remaining_maxima.append(remaining_max)
            if current_temperature_c is not None:
                output[f"{prefix}_remaining_max_minus_current_c"] = (
                    remaining_max - current_temperature_c
                )
            if running_max_c is not None:
                output[f"{prefix}_remaining_max_minus_running_max_c"] = (
                    remaining_max - running_max_c
                )
        if isinstance(hourly, dict):
            for hours in (1, 2, 3, 6):
                values = _nearest_hourly_values(
                    hourly, decision_local + timedelta(hours=hours), station
                )
                for name in (
                    "temperature_2m",
                    "dew_point_2m",
                    "cloud_cover",
                    "shortwave_radiation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "precipitation",
                    "temperature_925hPa",
                    "temperature_850hPa",
                ):
                    if name in values:
                        output[f"{prefix}_{name}_plus_{hours}h"] = values[name]
    if provider_maxima:
        output["model_consensus_today_max_c"] = sum(provider_maxima) / len(provider_maxima)
        output["model_spread_today_max_c"] = max(provider_maxima) - min(provider_maxima)
        output["model_provider_count"] = len(provider_maxima)
    if provider_remaining_maxima:
        output["model_consensus_remaining_max_c"] = sum(provider_remaining_maxima) / len(
            provider_remaining_maxima
        )
        output["model_spread_remaining_max_c"] = max(provider_remaining_maxima) - min(
            provider_remaining_maxima
        )
    return output


def _forecast_maxima_at(
    hourly: dict[str, Any], decision_local: datetime, station: StationConfig
) -> tuple[float | None, float | None]:
    times = hourly.get("time")
    temperatures = hourly.get("temperature_2m")
    if not isinstance(times, list) or not isinstance(temperatures, list):
        return None, None
    today: list[tuple[datetime, float]] = []
    for raw_time, raw_temperature in zip(times, temperatures, strict=False):
        try:
            timestamp = datetime.fromisoformat(str(raw_time))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ZoneInfo(station.timezone))
            else:
                timestamp = timestamp.astimezone(ZoneInfo(station.timezone))
            temperature = float(raw_temperature)
        except (TypeError, ValueError):
            continue
        if timestamp.date() == decision_local.date() and math.isfinite(temperature):
            today.append((timestamp, temperature))
    if not today:
        return None, None
    remaining = [temperature for timestamp, temperature in today if timestamp >= decision_local]
    return max(temperature for _, temperature in today), max(remaining) if remaining else None


def _peer_features_at(
    peers_by_id: dict[str, list[dict[str, Any]]],
    timestamp_ns: int,
    station: StationConfig,
    *,
    target_temperature_c: float,
    target_wind_direction_deg: float | None,
    target_wind_speed_kt: float | None,
    max_age_minutes: int,
) -> dict[str, Any]:

    output: dict[str, Any] = {}
    peer_temps: list[float] = []
    weighted_temp_sum = 0.0
    weight_sum = 0.0
    distance_weighted_temp_sum = 0.0
    distance_weight_sum = 0.0
    upstream_etas: list[float] = []
    peer_ages: list[float] = []
    peer_slopes_30m: list[float] = []
    peer_slopes_60m: list[float] = []

    for configured_peer in station.peer_station_ids:
        peer_id = configured_peer.upper()
        event = latest_before(peers_by_id.get(peer_id, []), timestamp_ns)
        if event is None:
            continue
        age_minutes = (timestamp_ns - event["received_wall_ns"]) / 60e9
        if age_minutes < 0 or age_minutes > max_age_minutes:
            continue
        payload = enrich_surface_payload(event.get("payload", {}))
        temp_c = _num(payload.get("temperature_c"))
        if temp_c is None:
            continue

        prefix = f"peer_{peer_id}"
        delta_c = temp_c - target_temperature_c
        peer_ages.append(age_minutes)
        output[f"{prefix}_age_minutes"] = age_minutes
        output[f"{prefix}_temperature_c"] = temp_c
        output[f"{prefix}_delta_c"] = delta_c
        output[f"{prefix}_dewpoint_c"] = _num(payload.get("dewpoint_c"))
        output[f"{prefix}_wind_direction_deg"] = _num(payload.get("wind_direction_deg"))
        output[f"{prefix}_wind_speed_kt"] = _num(payload.get("wind_speed_kt"))
        for name, value in surface_observation_features(payload).items():
            output[f"{prefix}_{name}"] = value
        slope_30m = _peer_temperature_slope(
            peers_by_id.get(peer_id, []), timestamp_ns, window_seconds=30 * 60
        )
        slope_60m = _peer_temperature_slope(
            peers_by_id.get(peer_id, []), timestamp_ns, window_seconds=60 * 60
        )
        if slope_30m is not None:
            output[f"{prefix}_slope_30m_c_per_hour"] = slope_30m
            peer_slopes_30m.append(slope_30m)
        if slope_60m is not None:
            output[f"{prefix}_slope_60m_c_per_hour"] = slope_60m
            peer_slopes_60m.append(slope_60m)
        peer_temps.append(temp_c)

        peer_lat = _num(payload.get("latitude"))
        peer_lon = _num(payload.get("longitude"))
        if peer_lat is None or peer_lon is None:
            peer_lat, peer_lon = _latest_peer_coordinates(
                peers_by_id.get(peer_id, []), timestamp_ns
            )
        if peer_lat is None or peer_lon is None:
            continue
        distance_km, bearing_deg = _distance_and_bearing(
            station.latitude, station.longitude, peer_lat, peer_lon
        )
        output[f"{prefix}_distance_km"] = distance_km
        output[f"{prefix}_bearing_deg"] = bearing_deg

        distance_weight = math.exp(-age_minutes / 60.0) / max(distance_km, 5.0)
        distance_weighted_temp_sum += distance_weight * temp_c
        distance_weight_sum += distance_weight

        if target_wind_direction_deg is None:
            continue
        angle = _angular_difference_deg(bearing_deg, target_wind_direction_deg)
        alignment = max(0.0, math.cos(math.radians(angle)))
        output[f"{prefix}_upwind_alignment"] = alignment

        weight = alignment * math.exp(-age_minutes / 60.0) / max(distance_km, 5.0)
        if weight > 0:
            weighted_temp_sum += weight * temp_c
            weight_sum += weight
        if alignment >= 0.5 and target_wind_speed_kt and target_wind_speed_kt > 1:
            eta_hours = (distance_km / 1.852) / target_wind_speed_kt
            output[f"{prefix}_advection_eta_hours"] = eta_hours
            upstream_etas.append(eta_hours)

    if peer_temps:
        output["peer_station_count"] = len(peer_temps)
        output["peer_expected_station_count"] = len(station.peer_station_ids)
        output["peer_station_coverage"] = len(peer_temps) / len(station.peer_station_ids)
        output["peer_temperature_mean_c"] = sum(peer_temps) / len(peer_temps)
        output["peer_temperature_min_c"] = min(peer_temps)
        output["peer_temperature_max_c"] = max(peer_temps)
        output["peer_temperature_spread_c"] = max(peer_temps) - min(peer_temps)
        output["peer_mean_delta_c"] = output["peer_temperature_mean_c"] - target_temperature_c
        output["peer_warmest_delta_c"] = max(peer_temps) - target_temperature_c
        output["peer_coolest_delta_c"] = min(peer_temps) - target_temperature_c
        output["peer_newest_age_minutes"] = min(peer_ages)
        output["peer_oldest_age_minutes"] = max(peer_ages)
    if distance_weight_sum > 0:
        distance_weighted_temp = distance_weighted_temp_sum / distance_weight_sum
        output["peer_distance_weighted_temperature_c"] = distance_weighted_temp
        output["peer_distance_weighted_delta_c"] = distance_weighted_temp - target_temperature_c
    if weight_sum > 0:
        upwind_temp = weighted_temp_sum / weight_sum
        output["peer_upwind_weighted_temperature_c"] = upwind_temp
        output["peer_upwind_weighted_delta_c"] = upwind_temp - target_temperature_c
    if upstream_etas:
        output["peer_min_advection_eta_hours"] = min(upstream_etas)
    if peer_slopes_30m:
        output["peer_mean_slope_30m_c_per_hour"] = sum(peer_slopes_30m) / len(peer_slopes_30m)
    if peer_slopes_60m:
        output["peer_mean_slope_60m_c_per_hour"] = sum(peer_slopes_60m) / len(peer_slopes_60m)
    return output


def _latest_peer_coordinates(
    events: list[dict[str, Any]], timestamp_ns: int
) -> tuple[float | None, float | None]:
    for event in reversed(events):
        if int(event.get("received_wall_ns") or 0) > timestamp_ns:
            continue
        payload = event.get("payload", {})
        latitude = _num(payload.get("latitude"))
        longitude = _num(payload.get("longitude"))
        if latitude is not None and longitude is not None:
            return latitude, longitude
    return None, None


def _peer_temperature_slope(
    events: list[dict[str, Any]], timestamp_ns: int, *, window_seconds: int
) -> float | None:
    cutoff_ns = timestamp_ns - window_seconds * 1_000_000_000
    points: list[tuple[int, float]] = []
    seen_observations: set[str] = set()
    for event in events:
        received_ns = int(event.get("received_wall_ns") or 0)
        if received_ns > timestamp_ns:
            break
        if received_ns < cutoff_ns:
            continue
        identity = weather_observation_key(event)
        if identity in seen_observations:
            continue
        temperature = _num(event.get("payload", {}).get("temperature_c"))
        if temperature is None:
            continue
        seen_observations.add(identity)
        points.append((received_ns, temperature))
    if len(points) < 2:
        return None
    first_ns, first_temperature = points[0]
    last_ns, last_temperature = points[-1]
    elapsed_hours = (last_ns - first_ns) / 3_600_000_000_000
    if elapsed_hours <= 0:
        return None
    return (last_temperature - first_temperature) / elapsed_hours


def _distance_and_bearing(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> tuple[float, float]:
    radius_km = 6371.0088
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon2_deg - lon1_deg)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    distance = 2 * radius_km * math.asin(min(1.0, math.sqrt(a)))
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _angular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _nearest_hourly_values(
    hourly: dict[str, Any], target: datetime, station: StationConfig
) -> dict[str, float]:
    raw_times = hourly.get("time") or []
    if not isinstance(raw_times, list) or not raw_times:
        return {}
    best_index: int | None = None
    best_delta = math.inf
    for index, raw_time in enumerate(raw_times):
        try:
            dt = datetime.fromisoformat(str(raw_time))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(station.timezone))
            delta = abs((dt - target).total_seconds())
        except ValueError:
            continue
        if delta < best_delta:
            best_delta = delta
            best_index = index
    if best_index is None or best_delta > 45 * 60:
        return {}
    output: dict[str, float] = {}
    for name, series in hourly.items():
        if name == "time" or not isinstance(series, list) or best_index >= len(series):
            continue
        value = _num(series[best_index])
        if value is not None:
            output[name] = value
    return output


def _final_wu_max(wu: list[dict[str, Any]]) -> float | None:
    wu.sort(key=lambda event: event["received_wall_ns"])
    values = [
        _num(e["payload"].get("daily_max_from_network"))
        for e in wu
        if e.get("payload", {}).get("resolution_final") is True
    ]
    valid = [value for value in values if value is not None]
    return valid[-1] if valid else None


def _running_wu_max_at(wu: list[dict[str, Any]], decision_ns: int) -> float | None:

    values = [
        value
        for event in wu
        if int(event.get("received_wall_ns") or 0) <= decision_ns
        and (value := _wu_snapshot_max(event)) is not None
    ]
    return max(values) if values else None


def _wu_snapshot_max(event: dict[str, Any]) -> float | None:
    payload = event.get("payload", {})
    value = _num(payload.get("daily_max_from_network"))
    if value is not None:
        return value
    return WundergroundCollector._visible_day_high(
        str(payload.get("body_text_sample") or ""),
        str(payload.get("market_unit") or ""),
    )


def _winning_label(
    final_max: float, station: StationConfig, books: list[dict[str, Any]]
) -> str | None:
    labels = {str(e["payload"].get("bucket_label") or "") for e in books}
    return _winning_label_from_labels(final_max, station, labels)


def _winning_label_from_labels(
    final_max: float, station: StationConfig, labels: set[str]
) -> str | None:
    represented = round_half_up(final_max)
    for label in labels:
        if not label:
            continue
        try:
            if parse_temperature_bucket(label, station.market_unit).contains(represented):
                return label
        except ValueError:
            continue
    return None


def _slope(history: deque[tuple[int, float]], now_ns: int, window_seconds: int) -> float | None:
    cutoff = now_ns - window_seconds * 1_000_000_000
    candidates = [(ts, value) for ts, value in history if ts >= cutoff]
    if len(candidates) < 2:
        return None
    first_ts, first_value = candidates[0]
    last_ts, last_value = candidates[-1]
    minutes = (last_ts - first_ts) / 60_000_000_000
    return (last_value - first_value) / minutes if minutes > 0 else None


def _normalize(distribution: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, value) for key, value in distribution.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()} if total else {}


def _multiclass_brier(probabilities: dict[str, float], winner: str) -> float:
    labels = set(probabilities) | {winner}
    return sum(
        (probabilities.get(label, 0.0) - (1.0 if label == winner else 0.0)) ** 2 for label in labels
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _station_feature_coverage(
    rows: list[dict[str, Any]], cfg: AppConfig
) -> list[dict[str, Any]]:
    core_features = (
        "relative_humidity_pct",
        "wind_u_kt",
        "cloud_fraction",
        "visibility_sm",
        "solar_elevation_deg",
        "model_consensus_today_max_c",
        "model_consensus_remaining_max_c",
        "peer_station_coverage",
        "peer_upwind_weighted_temperature_c",
        "peer_mean_slope_60m_c_per_hour",
    )
    output: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        station_rows = [row for row in rows if row.get("station_id") == station.station_id]
        coverage = {
            feature: (
                sum(_num(row.get(feature)) is not None for row in station_rows) / len(station_rows)
                if station_rows
                else 0.0
            )
            for feature in core_features
        }
        peer_coverage_values = [
            value
            for row in station_rows
            if (value := _num(row.get("peer_station_coverage"))) is not None
        ]
        output.append(
            {
                "station_id": station.station_id,
                "feature_rows": len(station_rows),
                "configured_peer_stations": len(station.peer_station_ids),
                "mean_peer_station_coverage": (
                    sum(peer_coverage_values) / len(peer_coverage_values)
                    if peer_coverage_values
                    else 0.0
                ),
                "coverage": coverage,
            }
        )
    return output
