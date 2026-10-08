from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .io import iter_events
from .model import _bootstrap_mean_ci, _normalize
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .station_specific_alpha import _load_csv, _truth


POINT_MATERIALITY_C = 0.02
PROBABILITY_TOTAL_VARIATION_MATERIALITY = 0.005
MINIMUM_MARKET_MID_MASS = 0.90
MAXIMUM_MARKET_MID_MASS = 1.20
MINIMUM_INDEPENDENT_STATION_DAYS = 20
MINIMUM_CANDIDATE_MODEL_PROBABILITY = 0.20


def refresh_station_maximum_v6_update_alpha(
    cfg: AppConfig,
    *,
    deployment_dir: Path,
    predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:

    predictions_path = deployment_dir / "shadow_predictions.csv"
    if predictions is None:
        predictions = _load_csv(predictions_path)
    forecast_cache_path = (
        deployment_dir / "update_alpha_forecast_triggers.csv"
    )
    forecast_metadata = _refresh_forecast_trigger_cache(
        cfg,
        predictions,
        _load_csv(forecast_cache_path),
    )
    _write_csv_atomic(forecast_cache_path, forecast_metadata)
    annotated_predictions = _annotate_forecast_novelty(
        predictions, forecast_metadata
    )
    events = _information_delta_rows(annotated_predictions)
    cache_path = deployment_dir / "update_alpha_market_snapshots.csv"
    cached_snapshots = _load_csv(cache_path)
    cached_snapshots = _refresh_market_snapshot_cache(
        cfg,
        events,
        cached_snapshots,
    )
    _write_csv_atomic(cache_path, cached_snapshots)
    reactions = _join_market_reactions(events, cached_snapshots)
    groups = _group_summaries(reactions)
    episodes, episode_reactions = _candidate_episodes(
        events, reactions
    )
    episode_groups = _episode_group_summaries(episode_reactions)
    live_candidates = [
        row
        for row in episodes
        if str(row.get("episode_end_reason") or "")
        == "still_open_at_last_observation"
        and not _truth(row.get("settled"))
    ]
    summary = _summary(
        events,
        reactions,
        groups,
        episodes,
        episode_reactions,
        episode_groups,
    )

    events_path = deployment_dir / "update_alpha_events.csv"
    reactions_path = deployment_dir / "update_alpha_reactions.csv"
    groups_path = deployment_dir / "update_alpha_groups.csv"
    episodes_path = deployment_dir / "update_alpha_episodes.csv"
    episode_reactions_path = (
        deployment_dir / "update_alpha_episode_reactions.csv"
    )
    episode_groups_path = (
        deployment_dir / "update_alpha_episode_groups.csv"
    )
    live_candidates_path = (
        deployment_dir / "update_alpha_live_candidates.csv"
    )
    summary_path = deployment_dir / "update_alpha_summary.json"
    report_path = deployment_dir / "UPDATE_ALPHA_REPORT.md"
    _write_csv_atomic(events_path, events)
    _write_csv_atomic(reactions_path, reactions)
    _write_csv_atomic(groups_path, groups)
    _write_csv_atomic(episodes_path, episodes)
    _write_csv_atomic(episode_reactions_path, episode_reactions)
    _write_csv_atomic(episode_groups_path, episode_groups)
    _write_csv_atomic(live_candidates_path, live_candidates)
    _write_json_atomic(summary_path, summary)
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {
        "update_alpha_events": events_path,
        "update_alpha_forecast_triggers": forecast_cache_path,
        "update_alpha_reactions": reactions_path,
        "update_alpha_groups": groups_path,
        "update_alpha_episodes": episodes_path,
        "update_alpha_episode_reactions": episode_reactions_path,
        "update_alpha_episode_groups": episode_groups_path,
        "update_alpha_live_candidates": live_candidates_path,
        "update_alpha_summary": summary_path,
        "update_alpha_report": report_path,
    }


def _information_delta_rows(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted = [
        row
        for row in predictions
        if not str(row.get("rejection_reason") or "").strip()
        and _finite(row.get("v6_point_c")) is not None
        and _json_probabilities(row.get("v6_distribution_json"))
    ]
    accepted.sort(
        key=lambda row: (
            str(row.get("station_id") or ""),
            str(row.get("local_date") or ""),
            _integer(row.get("received_wall_ns")) or 0,
            str(row.get("decision_id") or ""),
        )
    )
    previous_by_station_day: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    output: list[dict[str, Any]] = []
    for current in accepted:
        station_id = str(current.get("station_id") or "")
        local_date = str(current.get("local_date") or "")
        key = (station_id, local_date)
        previous = previous_by_station_day.get(key)
        previous_by_station_day[key] = current
        if previous is None:
            continue
        current_ns = _integer(current.get("received_wall_ns"))
        previous_ns = _integer(previous.get("received_wall_ns"))
        current_distribution = _normalize(
            _json_probabilities(current.get("v6_distribution_json"))
        )
        previous_distribution = _normalize(
            _json_probabilities(previous.get("v6_distribution_json"))
        )
        labels = sorted(
            set(current_distribution) | set(previous_distribution)
        )
        probability_delta = {
            label: current_distribution.get(label, 0.0)
            - previous_distribution.get(label, 0.0)
            for label in labels
        }
        total_variation = 0.5 * sum(
            abs(value) for value in probability_delta.values()
        )
        point_delta = float(current["v6_point_c"]) - float(
            previous["v6_point_c"]
        )
        increased_bucket = max(
            labels, key=lambda label: probability_delta[label]
        )
        decreased_bucket = min(
            labels, key=lambda label: probability_delta[label]
        )
        winner = str(current.get("winning_bucket") or "")
        previous_brier = _bucket_brier(previous_distribution, winner)
        current_brier = _bucket_brier(current_distribution, winner)
        mode = str(current.get("decision_mode") or "")
        trigger = str(current.get("trigger_source") or "")
        material = bool(
            abs(point_delta) >= POINT_MATERIALITY_C
            or total_variation
            >= PROBABILITY_TOTAL_VARIATION_MATERIALITY
        )
        market_t0 = current.get("market_distribution_json")
        market_t0_raw = _json_probabilities(market_t0)
        market_t0_valid = _valid_market_mass(
            sum(market_t0_raw.values())
        )
        market_t0_normalized = (
            _normalize(market_t0_raw) if market_t0_valid else {}
        )
        increased_labels = [
            label
            for label in labels
            if probability_delta[label] > 0.0
        ]
        candidate_bucket = max(
            increased_labels,
            key=lambda label: (
                current_distribution.get(label, 0.0)
                - market_t0_normalized.get(label, 0.0)
            ),
            default=increased_bucket,
        )
        candidate_model_probability = current_distribution.get(
            candidate_bucket, 0.0
        )
        ranked_buckets = sorted(
            labels,
            key=lambda label: (
                -current_distribution.get(label, 0.0),
                label,
            ),
        )
        candidate_bucket_rank = ranked_buckets.index(candidate_bucket) + 1
        model_top_bucket = ranked_buckets[0]
        model_top_probability = current_distribution.get(
            model_top_bucket, 0.0
        )
        model_top_probability_increase = probability_delta.get(
            model_top_bucket, 0.0
        )
        model_top_weather_market_gap = (
            model_top_probability
            - market_t0_normalized.get(model_top_bucket, 0.0)
        )
        candidate_probability_increase = probability_delta.get(
            candidate_bucket, 0.0
        )
        weather_market_gap = (
            candidate_model_probability
            - market_t0_normalized.get(candidate_bucket, 0.0)
        )
        active_information = _active_information_update(
            mode, current.get("forecast_content_changed")
        )
        candidate_lead = bool(
            material
            and active_information
            and market_t0_valid
            and candidate_probability_increase > 0.0
            and weather_market_gap > 0.0
        )
        qualified_candidate = bool(
            candidate_lead
            and candidate_model_probability
            >= MINIMUM_CANDIDATE_MODEL_PROBABILITY
        )
        qualified_positive_gap_state = bool(
            material
            and market_t0_valid
            and candidate_probability_increase > 0.0
            and weather_market_gap > 0.0
            and candidate_model_probability
            >= MINIMUM_CANDIDATE_MODEL_PROBABILITY
        )
        modal_candidate = bool(
            material
            and active_information
            and market_t0_valid
            and model_top_probability_increase > 0.0
            and model_top_weather_market_gap > 0.0
            and model_top_probability
            >= MINIMUM_CANDIDATE_MODEL_PROBABILITY
        )
        modal_positive_gap_state = bool(
            material
            and market_t0_valid
            and model_top_probability_increase > 0.0
            and model_top_weather_market_gap > 0.0
            and model_top_probability
            >= MINIMUM_CANDIDATE_MODEL_PROBABILITY
        )
        output.append(
            {
                "decision_id": current.get("decision_id"),
                "run_id": current.get("run_id"),
                "station_id": station_id,
                "local_date": local_date,
                "local_hour": current.get("local_hour"),
                "local_hour_bin": _local_hour_bin(
                    _finite(current.get("local_hour"))
                ),
                "received_utc": current.get("received_utc"),
                "received_wall_ns": current_ns,
                "evaluation_phase": current.get("evaluation_phase"),
                "processed_at_ns": current.get("processed_at_ns"),
                "processing_lag_seconds": current.get(
                    "processing_lag_seconds"
                ),
                "decision_mode": mode,
                "trigger_source": trigger,
                "information_class": _information_class(
                    mode,
                    trigger,
                    current.get("forecast_content_changed"),
                ),
                "forecast_provider": (
                    trigger.removeprefix("forecast_")
                    if trigger.startswith("forecast_")
                    else ""
                ),
                "forecast_snapshot_kind": current.get(
                    "forecast_snapshot_kind"
                ),
                "forecast_content_sha256": current.get(
                    "forecast_content_sha256"
                ),
                "forecast_previous_content_sha256": current.get(
                    "forecast_previous_content_sha256"
                ),
                "forecast_content_changed": current.get(
                    "forecast_content_changed"
                ),
                "previous_decision_id": previous.get("decision_id"),
                "previous_received_wall_ns": previous_ns,
                "seconds_since_previous_decision": (
                    (current_ns - previous_ns) / 1e9
                    if current_ns is not None
                    and previous_ns is not None
                    else None
                ),
                "previous_v6_point_c": previous.get("v6_point_c"),
                "v6_point_c": current.get("v6_point_c"),
                "v6_point_delta_c": point_delta,
                "probability_total_variation": total_variation,
                "probability_delta_json": json.dumps(
                    probability_delta, sort_keys=True
                ),
                "largest_probability_increase_bucket": increased_bucket,
                "largest_probability_increase": probability_delta[
                    increased_bucket
                ],
                "largest_probability_decrease_bucket": decreased_bucket,
                "largest_probability_decrease": probability_delta[
                    decreased_bucket
                ],
                "candidate_bucket": candidate_bucket,
                "candidate_bucket_model_probability": (
                    candidate_model_probability
                ),
                "candidate_bucket_probability_increase": (
                    candidate_probability_increase
                ),
                "candidate_model_probability_bin": (
                    _probability_bin(candidate_model_probability)
                ),
                "candidate_bucket_rank": candidate_bucket_rank,
                "candidate_bucket_rank_band": _rank_band(
                    candidate_bucket_rank
                ),
                "candidate_is_model_top": candidate_bucket_rank == 1,
                "candidate_probability_below_model_top": (
                    model_top_probability - candidate_model_probability
                ),
                "model_top_bucket": model_top_bucket,
                "model_top_bucket_probability": model_top_probability,
                "model_top_bucket_probability_increase": (
                    model_top_probability_increase
                ),
                "model_top_probability_bin": _probability_bin(
                    model_top_probability
                ),
                "model_top_weather_market_gap_t0": (
                    model_top_weather_market_gap
                ),
                "model_top_weather_market_gap_bin": _gap_bin(
                    model_top_weather_market_gap
                ),
                "market_distribution_t0_json": market_t0,
                "market_mid_mass_t0": sum(
                    _json_probabilities(
                        current.get("market_distribution_json")
                    ).values()
                ),
                "weather_market_gap_t0": weather_market_gap,
                "weather_market_gap_bin": _gap_bin(
                    weather_market_gap
                ),
                "material_model_update": material,
                "active_information_update": active_information,
                "candidate_lead_update": candidate_lead,
                "qualified_candidate_update": qualified_candidate,
                "qualified_positive_gap_update_state": (
                    qualified_positive_gap_state
                ),
                "modal_candidate_update": modal_candidate,
                "modal_positive_gap_update_state": (
                    modal_positive_gap_state
                ),
                "winning_bucket": winner,
                "settled": _truth(current.get("settled")) or bool(winner),
                "winning_bucket_probability_delta": (
                    probability_delta.get(winner) if winner else None
                ),
                "final_brier_before": previous_brier,
                "final_brier_after": current_brier,
                "final_brier_improvement": (
                    previous_brier - current_brier
                    if previous_brier is not None
                    and current_brier is not None
                    else None
                ),
                "update_improved_final_outcome": (
                    current_brier < previous_brier
                    if previous_brier is not None
                    and current_brier is not None
                    else None
                ),
                "live_orders_enabled": False,
            }
        )
    next_material_by_station_day: dict[
        tuple[str, str], int
    ] = {}
    for row in reversed(output):
        key = (str(row["station_id"]), str(row["local_date"]))
        row["next_material_update_wall_ns"] = (
            next_material_by_station_day.get(key)
        )
        if _truth(row.get("material_model_update")):
            timestamp_ns = _integer(row.get("received_wall_ns"))
            if timestamp_ns is not None:
                next_material_by_station_day[key] = timestamp_ns
    return output


def _refresh_forecast_trigger_cache(
    cfg: AppConfig,
    predictions: list[dict[str, Any]],
    cached: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_decision = {
        str(row.get("decision_id") or ""): row
        for row in cached
        if row.get("decision_id")
    }
    missing_by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in predictions:
        if str(row.get("decision_mode") or "") != "forecast_update":
            continue
        decision_id = str(row.get("decision_id") or "")
        run_id = str(row.get("run_id") or "")
        if decision_id and run_id and decision_id not in by_decision:
            missing_by_run[run_id].append(row)
    for run_id, missing in missing_by_run.items():
        run_dir = cfg.data_dir / "runs" / run_id
        if not run_dir.exists():
            continue
        wanted_sources = {
            str(row.get("trigger_source") or "")
            for row in missing
            if str(row.get("trigger_source") or "").startswith(
                "forecast_"
            )
        }
        raw_by_stream: defaultdict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for event in iter_events(run_dir, wanted_sources):
            if event.get("event_type") != "forecast_snapshot":
                continue
            source = str(event.get("source") or "")
            station_id = str(event.get("station_id") or "")
            timestamp_ns = _integer(event.get("received_wall_ns"))
            if not source or not station_id or timestamp_ns is None:
                continue
            payload = event.get("payload", {})
            raw_by_stream[(station_id, source)].append(
                {
                    "timestamp_ns": timestamp_ns,
                    "snapshot_kind": payload.get("snapshot_kind"),
                    "content_sha256": payload.get("content_sha256"),
                    "previous_content_sha256": payload.get(
                        "previous_content_sha256"
                    ),
                }
            )
        for values in raw_by_stream.values():
            values.sort(key=lambda item: item["timestamp_ns"])
        for row in missing:
            decision_id = str(row.get("decision_id") or "")
            station_id = str(row.get("station_id") or "")
            source = str(row.get("trigger_source") or "")
            timestamp_ns = _integer(row.get("received_wall_ns"))
            if timestamp_ns is None:
                continue
            candidates = raw_by_stream.get((station_id, source), [])
            closest = min(
                candidates,
                key=lambda item: abs(
                    int(item["timestamp_ns"]) - timestamp_ns
                ),
                default=None,
            )
            if (
                closest is None
                or abs(int(closest["timestamp_ns"]) - timestamp_ns)
                > 100_000
            ):
                continue
            by_decision[decision_id] = {
                "decision_id": decision_id,
                "run_id": run_id,
                "station_id": station_id,
                "trigger_source": source,
                "received_wall_ns": timestamp_ns,
                "raw_received_wall_ns": closest["timestamp_ns"],
                "forecast_snapshot_kind": closest["snapshot_kind"],
                "forecast_content_sha256": closest["content_sha256"],
                "forecast_previous_content_sha256": closest[
                    "previous_content_sha256"
                ],
            }
    return sorted(
        by_decision.values(),
        key=lambda row: (
            _integer(row.get("received_wall_ns")) or 0,
            str(row.get("decision_id") or ""),
        ),
    )


def _annotate_forecast_novelty(
    predictions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_decision = {
        str(row.get("decision_id") or ""): row for row in metadata
    }
    ordered = sorted(
        predictions,
        key=lambda row: (
            _integer(row.get("received_wall_ns")) or 0,
            str(row.get("decision_id") or ""),
        ),
    )
    previous_content: dict[tuple[str, str], str] = {}
    output: list[dict[str, Any]] = []
    for row in ordered:
        annotated = dict(row)
        if str(row.get("decision_mode") or "") == "forecast_update":
            item = metadata_by_decision.get(
                str(row.get("decision_id") or ""), {}
            )
            source = str(row.get("trigger_source") or "")
            stream = (str(row.get("station_id") or ""), source)
            content = str(
                item.get("forecast_content_sha256") or ""
            )
            previous = previous_content.get(stream)
            payload_previous = str(
                item.get("forecast_previous_content_sha256") or ""
            )
            changed = bool(
                content
                and content != previous
                and (
                    previous is not None
                    or not payload_previous
                    or content != payload_previous
                )
            )
            annotated.update(item)
            annotated["forecast_content_changed"] = changed
            if content:
                previous_content[stream] = content
        output.append(annotated)
    return output


def _refresh_market_snapshot_cache(
    cfg: AppConfig,
    events: list[dict[str, Any]],
    cached: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row.get("decision_id") or ""),
            _integer(row.get("horizon_seconds")) or 0,
        ): row
        for row in cached
    }
    targets_by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for event in events:
        if not _truth(event.get("material_model_update")):
            continue
        decision_id = str(event.get("decision_id") or "")
        run_id = str(event.get("run_id") or "")
        timestamp_ns = _integer(event.get("received_wall_ns"))
        if not decision_id or not run_id or timestamp_ns is None:
            continue
        for horizon in cfg.execution.markout_seconds:
            cache_key = (decision_id, int(horizon))
            if cache_key in by_key:
                continue
            targets_by_run[run_id].append(
                {
                    "cache_key": cache_key,
                    "decision_id": decision_id,
                    "run_id": run_id,
                    "station_id": event["station_id"],
                    "local_date": event["local_date"],
                    "horizon_seconds": int(horizon),
                    "target_timestamp_ns": (
                        timestamp_ns + int(horizon) * 1_000_000_000
                    ),
                }
            )
    for run_id, targets in targets_by_run.items():
        run_dir = cfg.data_dir / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not run_dir.exists() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not manifest.get("completed_at"):
            continue
        for row in _stream_run_market_snapshots(run_dir, targets):
            by_key[row["cache_key"]] = {
                key: value
                for key, value in row.items()
                if key != "cache_key"
            }
    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row.get("run_id") or ""),
            _integer(row.get("target_timestamp_ns")) or 0,
            str(row.get("decision_id") or ""),
        ),
    )


def _stream_run_market_snapshots(
    run_dir: Path,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        targets,
        key=lambda row: int(row["target_timestamp_ns"]),
    )
    latest: defaultdict[
        tuple[str, str], dict[str, tuple[Any, ...]]
    ] = defaultdict(dict)
    labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    context_last_ns: dict[tuple[str, str], int] = {}
    captured: list[tuple[dict[str, Any], dict[str, tuple[Any, ...]], int | None]] = []
    target_index = 0
    last_event_ns = 0

    def capture_before(timestamp_ns: int) -> None:
        nonlocal target_index
        while target_index < len(ordered):
            target = ordered[target_index]
            if int(target["target_timestamp_ns"]) >= timestamp_ns:
                break
            context = (
                str(target["station_id"]),
                str(target["local_date"]),
            )
            captured.append(
                (
                    target,
                    dict(latest.get(context, {})),
                    context_last_ns.get(context),
                )
            )
            target_index += 1

    for event in iter_events(run_dir, {"polymarket_book"}):
        if event.get("event_type") != "book_state":
            continue
        event_ns = _integer(event.get("received_wall_ns")) or 0
        if event_ns <= 0:
            continue
        last_event_ns = max(last_event_ns, event_ns)
        capture_before(event_ns)
        station_id = str(event.get("station_id") or "")
        payload = event.get("payload", {})
        event_date = str(payload.get("event_date") or "")
        label = str(payload.get("bucket_label") or "")
        if not station_id or not event_date or not label:
            continue
        context = (station_id, event_date)
        labels[context].add(label)
        latest[context][label] = (
            _finite(payload.get("mid")),
            _finite(payload.get("best_bid")),
            _finite(payload.get("best_ask")),
        )
        context_last_ns[context] = event_ns

    while (
        target_index < len(ordered)
        and int(ordered[target_index]["target_timestamp_ns"])
        <= last_event_ns
    ):
        target = ordered[target_index]
        context = (
            str(target["station_id"]),
            str(target["local_date"]),
        )
        captured.append(
            (
                target,
                dict(latest.get(context, {})),
                context_last_ns.get(context),
            )
        )
        target_index += 1

    output: list[dict[str, Any]] = []
    for target, state, snapshot_ns in captured:
        context = (
            str(target["station_id"]),
            str(target["local_date"]),
        )
        mid = {
            label: float(values[0])
            for label, values in state.items()
            if values[0] is not None
        }
        bid = {
            label: float(values[1])
            for label, values in state.items()
            if values[1] is not None
        }
        ask = {
            label: float(values[2])
            for label, values in state.items()
            if values[2] is not None
        }
        expected = labels.get(context, set())
        output.append(
            {
                **target,
                "snapshot_received_wall_ns": snapshot_ns,
                "market_distribution_json": json.dumps(
                    mid, sort_keys=True
                ),
                "market_bid_json": json.dumps(bid, sort_keys=True),
                "market_ask_json": json.dumps(ask, sort_keys=True),
                "market_mid_mass": sum(mid.values()),
                "market_quote_coverage": (
                    len(mid) / len(expected) if expected else 0.0
                ),
            }
        )
    return output


def _join_market_reactions(
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot_by_key = {
        (
            str(row.get("decision_id") or ""),
            _integer(row.get("horizon_seconds")) or 0,
        ): row
        for row in snapshots
    }
    output: list[dict[str, Any]] = []
    for event in events:
        if not _truth(event.get("material_model_update")):
            continue
        model_delta = _json_numbers(
            event.get("probability_delta_json")
        )
        t0_raw = _json_probabilities(
            event.get("market_distribution_t0_json")
        )
        t0_mass = sum(t0_raw.values())
        t0_valid = _valid_market_mass(t0_mass)
        t0 = _normalize(t0_raw) if t0_valid else {}
        target_bucket = str(
            event.get("candidate_bucket") or ""
        )
        model_top_bucket = str(
            event.get("model_top_bucket") or ""
        )
        for horizon in sorted(
            {
                int(row.get("horizon_seconds") or 0)
                for row in snapshots
                if str(row.get("decision_id") or "")
                == str(event.get("decision_id") or "")
            }
        ):
            snapshot = snapshot_by_key.get(
                (str(event.get("decision_id") or ""), horizon)
            )
            if snapshot is None:
                continue
            future_raw = _json_probabilities(
                snapshot.get("market_distribution_json")
            )
            future_mass = sum(future_raw.values())
            valid = bool(t0_valid and _valid_market_mass(future_mass))
            future = _normalize(future_raw) if valid else {}
            labels = sorted(
                set(model_delta) | set(t0) | set(future)
            )
            market_delta = {
                label: future.get(label, 0.0) - t0.get(label, 0.0)
                for label in labels
            }
            dot = (
                sum(
                    model_delta.get(label, 0.0)
                    * market_delta.get(label, 0.0)
                    for label in labels
                )
                if valid
                else None
            )
            cosine = (
                _cosine_similarity(model_delta, market_delta)
                if valid
                else None
            )
            target_delta = (
                market_delta.get(target_bucket) if valid else None
            )
            model_top_delta = (
                market_delta.get(model_top_bucket) if valid else None
            )
            target_timestamp_ns = _integer(
                snapshot.get("target_timestamp_ns")
            )
            next_update_ns = _integer(
                event.get("next_material_update_wall_ns")
            )
            uncontaminated = bool(
                target_timestamp_ns is not None
                and (
                    next_update_ns is None
                    or target_timestamp_ns < next_update_ns
                )
            )
            output.append(
                {
                    **event,
                    "horizon_seconds": horizon,
                    "target_timestamp_ns": snapshot.get(
                        "target_timestamp_ns"
                    ),
                    "market_snapshot_received_wall_ns": snapshot.get(
                        "snapshot_received_wall_ns"
                    ),
                    "market_snapshot_available": True,
                    "market_vector_valid": valid,
                    "uncontaminated_by_later_model_update": (
                        uncontaminated
                    ),
                    "market_mid_mass_future": future_mass,
                    "market_quote_coverage_future": snapshot.get(
                        "market_quote_coverage"
                    ),
                    "market_distribution_future_json": snapshot.get(
                        "market_distribution_json"
                    ),
                    "market_probability_delta_json": json.dumps(
                        market_delta, sort_keys=True
                    ),
                    "market_model_delta_dot_product": dot,
                    "market_model_delta_cosine": cosine,
                    "target_bucket_market_probability_t0": (
                        t0.get(target_bucket) if valid else None
                    ),
                    "target_bucket_market_probability_future": (
                        future.get(target_bucket) if valid else None
                    ),
                    "target_bucket_market_probability_delta": (
                        target_delta
                    ),
                    "market_followed_target_bucket": (
                        target_delta > 0.0
                        if target_delta is not None
                        else None
                    ),
                    "model_top_bucket_market_probability_t0": (
                        t0.get(model_top_bucket) if valid else None
                    ),
                    "model_top_bucket_market_probability_future": (
                        future.get(model_top_bucket) if valid else None
                    ),
                    "model_top_bucket_market_probability_delta": (
                        model_top_delta
                    ),
                    "market_followed_model_top_bucket": (
                        model_top_delta > 0.0
                        if model_top_delta is not None
                        else None
                    ),
                    "market_followed_full_distribution": (
                        dot > 0.0 if dot is not None else None
                    ),
                    "lead_confirmed": (
                        target_delta > 0.0
                        and (_finite(event.get("weather_market_gap_t0")) or 0.0)
                        > 0.0
                        if target_delta is not None
                        else None
                    ),
                    "modal_lead_confirmed": (
                        model_top_delta > 0.0
                        and (
                            _finite(
                                event.get(
                                    "model_top_weather_market_gap_t0"
                                )
                            )
                            or 0.0
                        )
                        > 0.0
                        if model_top_delta is not None
                        else None
                    ),
                    "live_orders_enabled": False,
                }
            )
    return output


def _group_summaries(
    reactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions: list[tuple[str, Any]] = [
        ("overall", lambda row: "all"),
        ("station_id", lambda row: str(row.get("station_id") or "")),
        (
            "decision_mode",
            lambda row: str(row.get("decision_mode") or ""),
        ),
        (
            "trigger_source",
            lambda row: str(row.get("trigger_source") or ""),
        ),
        (
            "station_source",
            lambda row: (
                f"{row.get('station_id')}|{row.get('trigger_source')}"
            ),
        ),
        (
            "local_hour_bin",
            lambda row: str(row.get("local_hour_bin") or ""),
        ),
        (
            "information_class",
            lambda row: str(row.get("information_class") or ""),
        ),
        (
            "activity_class",
            lambda row: (
                "active_information"
                if _truth(row.get("active_information_update"))
                else "passive_scheduled_control"
            ),
        ),
        (
            "candidate_class",
            lambda row: (
                "active_candidate_lead"
                if _truth(row.get("candidate_lead_update"))
                else (
                    "active_update_without_positive_gap"
                    if _truth(row.get("active_information_update"))
                    else "passive_control"
                )
            ),
        ),
        (
            "qualified_candidate_class",
            lambda row: (
                "qualified_active_candidate"
                if _truth(row.get("qualified_candidate_update"))
                else "not_qualified"
            ),
        ),
        (
            "modal_candidate_class",
            lambda row: (
                "qualified_modal_candidate"
                if _truth(row.get("modal_candidate_update"))
                else "not_qualified"
            ),
        ),
        (
            "weather_market_gap_bin",
            lambda row: str(row.get("weather_market_gap_bin") or ""),
        ),
        (
            "candidate_station_id",
            lambda row: (
                str(row.get("station_id") or "")
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "candidate_trigger_source",
            lambda row: (
                str(row.get("trigger_source") or "")
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "candidate_local_hour_bin",
            lambda row: (
                str(row.get("local_hour_bin") or "")
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "candidate_weather_market_gap_bin",
            lambda row: (
                str(row.get("weather_market_gap_bin") or "")
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "candidate_model_probability_bin",
            lambda row: (
                str(
                    row.get("candidate_model_probability_bin") or ""
                )
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "candidate_bucket_rank_band",
            lambda row: (
                str(row.get("candidate_bucket_rank_band") or "")
                if _truth(row.get("qualified_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "modal_station_id",
            lambda row: (
                str(row.get("station_id") or "")
                if _truth(row.get("modal_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "modal_trigger_source",
            lambda row: (
                str(row.get("trigger_source") or "")
                if _truth(row.get("modal_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "modal_local_hour_bin",
            lambda row: (
                str(row.get("local_hour_bin") or "")
                if _truth(row.get("modal_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "modal_weather_market_gap_bin",
            lambda row: (
                str(row.get("model_top_weather_market_gap_bin") or "")
                if _truth(row.get("modal_candidate_update"))
                else "__excluded__"
            ),
        ),
        (
            "modal_model_probability_bin",
            lambda row: (
                str(row.get("model_top_probability_bin") or "")
                if _truth(row.get("modal_candidate_update"))
                else "__excluded__"
            ),
        ),
    ]
    output: list[dict[str, Any]] = []
    for dimension, key_fn in definitions:
        grouped: defaultdict[
            tuple[str, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in reactions:
            grouped[
                (
                    key_fn(row),
                    int(row.get("horizon_seconds") or 0),
                )
            ].append(row)
        for (value, horizon), rows in sorted(grouped.items()):
            if value == "__excluded__":
                continue
            modal_group = dimension.startswith("modal_")
            target_delta_field = (
                "model_top_bucket_market_probability_delta"
                if modal_group
                else "target_bucket_market_probability_delta"
            )
            target_followed_field = (
                "market_followed_model_top_bucket"
                if modal_group
                else "market_followed_target_bucket"
            )
            valid = [
                row
                for row in rows
                if _truth(row.get("market_vector_valid"))
                and _truth(
                    row.get(
                        "uncontaminated_by_later_model_update"
                    )
                )
            ]
            day_target_deltas = _station_day_means(
                valid, target_delta_field
            )
            day_dots = _station_day_means(
                valid, "market_model_delta_dot_product"
            )
            resolved = [
                row
                for row in rows
                if _finite(row.get("final_brier_improvement")) is not None
            ]
            target_ci = _bootstrap_mean_ci(day_target_deltas)
            dot_ci = _bootstrap_mean_ci(day_dots)
            output.append(
                {
                    "group_dimension": dimension,
                    "group_value": value,
                    "horizon_seconds": horizon,
                    "material_updates": len(rows),
                    "independent_station_days": len(
                        {
                            (
                                str(row.get("station_id") or ""),
                                str(row.get("local_date") or ""),
                            )
                            for row in rows
                        }
                    ),
                    "valid_market_reactions": len(valid),
                    "market_reactions_before_causal_censoring": sum(
                        _truth(row.get("market_vector_valid"))
                        for row in rows
                    ),
                    "mean_target_bucket_market_delta": _mean(
                        [
                            value
                            for row in valid
                            if (
                                value := _finite(
                                    row.get(
                                        target_delta_field
                                    )
                                )
                            )
                            is not None
                        ]
                    ),
                    "station_day_mean_target_delta": _mean(
                        day_target_deltas
                    ),
                    "station_day_target_delta_ci95_low": (
                        target_ci["low"] if target_ci else None
                    ),
                    "station_day_target_delta_ci95_high": (
                        target_ci["high"] if target_ci else None
                    ),
                    "market_followed_target_fraction": _truth_fraction(
                        valid, target_followed_field
                    ),
                    "mean_full_distribution_dot_product": _mean(
                        [
                            value
                            for row in valid
                            if (
                                value := _finite(
                                    row.get(
                                        "market_model_delta_dot_product"
                                    )
                                )
                            )
                            is not None
                        ]
                    ),
                    "station_day_mean_dot_product": _mean(day_dots),
                    "station_day_dot_product_ci95_low": (
                        dot_ci["low"] if dot_ci else None
                    ),
                    "station_day_dot_product_ci95_high": (
                        dot_ci["high"] if dot_ci else None
                    ),
                    "mean_full_distribution_cosine": _mean(
                        [
                            value
                            for row in valid
                            if (
                                value := _finite(
                                    row.get(
                                        "market_model_delta_cosine"
                                    )
                                )
                            )
                            is not None
                        ]
                    ),
                    "lead_confirmed_fraction": _truth_fraction(
                        valid, "lead_confirmed"
                    ),
                    "resolved_updates": len(resolved),
                    "final_outcome_improved_fraction": (
                        sum(
                            _truth(
                                row.get("update_improved_final_outcome")
                            )
                            for row in resolved
                        )
                        / len(resolved)
                        if resolved
                        else None
                    ),
                    "mean_final_brier_improvement": _mean(
                        [
                            float(row["final_brier_improvement"])
                            for row in resolved
                        ]
                    ),
                }
            )
    return output


def _candidate_episodes(
    events: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    material = sorted(
        (
            row
            for row in events
            if _truth(row.get("material_model_update"))
        ),
        key=lambda row: (
            str(row.get("station_id") or ""),
            str(row.get("local_date") or ""),
            _integer(row.get("received_wall_ns")) or 0,
        ),
    )
    episodes: list[dict[str, Any]] = []
    open_by_station_day: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    for row in material:
        station_day = (
            str(row.get("station_id") or ""),
            str(row.get("local_date") or ""),
        )
        timestamp_ns = _integer(row.get("received_wall_ns")) or 0
        target = str(
            row.get("model_top_bucket") or ""
        )
        positive_state = _truth(
            row.get("modal_positive_gap_update_state")
        )
        current = open_by_station_day.get(station_day)
        if (
            current is not None
            and positive_state
            and target == current["target_bucket"]
        ):
            current["confirmation_count"] += 1
            current["active_confirmation_count"] += int(
                _truth(row.get("active_information_update"))
            )
            current["last_confirmation_wall_ns"] = timestamp_ns
            current["maximum_weather_market_gap"] = max(
                float(current["maximum_weather_market_gap"]),
                float(row["model_top_weather_market_gap_t0"]),
            )
            continue
        if current is not None:
            current["episode_end_wall_ns"] = timestamp_ns
            current["episode_end_reason"] = (
                "target_changed"
                if positive_state
                else "positive_gap_ended"
            )
            open_by_station_day.pop(station_day, None)
        if not _truth(row.get("modal_candidate_update")):
            continue
        episode = {
            "episode_id": (
                f"{row.get('decision_id')}|{target}"
            ),
            "start_decision_id": row.get("decision_id"),
            "run_id": row.get("run_id"),
            "evaluation_phase": row.get("evaluation_phase"),
            "station_id": row.get("station_id"),
            "local_date": row.get("local_date"),
            "local_hour": row.get("local_hour"),
            "local_hour_bin": row.get("local_hour_bin"),
            "trigger_source": row.get("trigger_source"),
            "information_class": row.get("information_class"),
            "weather_market_gap_bin": row.get(
                "model_top_weather_market_gap_bin"
            ),
            "signal_variant": "model_modal_bucket",
            "target_bucket": target,
            "episode_start_wall_ns": timestamp_ns,
            "processed_at_ns": row.get("processed_at_ns"),
            "processing_lag_seconds": row.get(
                "processing_lag_seconds"
            ),
            "last_confirmation_wall_ns": timestamp_ns,
            "episode_end_wall_ns": None,
            "episode_end_reason": "still_open_at_last_observation",
            "confirmation_count": 1,
            "active_confirmation_count": 1,
            "initial_weather_market_gap": row.get(
                "model_top_weather_market_gap_t0"
            ),
            "maximum_weather_market_gap": row.get(
                "model_top_weather_market_gap_t0"
            ),
            "initial_probability_increase": row.get(
                "model_top_bucket_probability_increase"
            ),
            "initial_model_probability": row.get(
                "model_top_bucket_probability"
            ),
            "winning_bucket": row.get("winning_bucket"),
            "settled": row.get("settled"),
            "target_won": (
                target == str(row.get("winning_bucket") or "")
                if row.get("winning_bucket")
                else None
            ),
            "live_orders_enabled": False,
        }
        episodes.append(episode)
        open_by_station_day[station_day] = episode
    for episode in episodes:
        start_ns = _integer(episode.get("episode_start_wall_ns"))
        last_ns = _integer(
            episode.get("last_confirmation_wall_ns")
        )
        end_ns = _integer(episode.get("episode_end_wall_ns"))
        episode["observed_persistence_seconds"] = (
            (last_ns - start_ns) / 1e9
            if start_ns is not None and last_ns is not None
            else None
        )
        episode["closed_duration_seconds"] = (
            (end_ns - start_ns) / 1e9
            if start_ns is not None and end_ns is not None
            else None
        )

    reactions_by_decision: defaultdict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    for row in reactions:
        reactions_by_decision[
            str(row.get("decision_id") or "")
        ].append(row)
    episode_reactions: list[dict[str, Any]] = []
    for episode in episodes:
        end_ns = _integer(episode.get("episode_end_wall_ns"))
        for reaction in reactions_by_decision.get(
            str(episode.get("start_decision_id") or ""), []
        ):
            target_ns = _integer(reaction.get("target_timestamp_ns"))
            survives = bool(
                target_ns is not None
                and (end_ns is None or target_ns < end_ns)
            )
            modal_delta = _finite(
                reaction.get(
                    "model_top_bucket_market_probability_delta"
                )
            )
            episode_reactions.append(
                {
                    **reaction,
                    **episode,
                    "target_bucket_market_probability_t0": (
                        reaction.get(
                            "model_top_bucket_market_probability_t0"
                        )
                    ),
                    "target_bucket_market_probability_future": (
                        reaction.get(
                            "model_top_bucket_market_probability_future"
                        )
                    ),
                    "target_bucket_market_probability_delta": modal_delta,
                    "market_followed_target_bucket": (
                        modal_delta > 0.0
                        if modal_delta is not None
                        else None
                    ),
                    "lead_confirmed": reaction.get(
                        "modal_lead_confirmed"
                    ),
                    "episode_survives_horizon": survives,
                }
            )
    return episodes, episode_reactions


def _episode_group_summaries(
    reactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions: list[tuple[str, Any]] = [
        ("overall", lambda row: "all"),
        ("station_id", lambda row: str(row.get("station_id") or "")),
        (
            "trigger_source",
            lambda row: str(row.get("trigger_source") or ""),
        ),
        (
            "local_hour_bin",
            lambda row: str(row.get("local_hour_bin") or ""),
        ),
        (
            "weather_market_gap_bin",
            lambda row: str(row.get("weather_market_gap_bin") or ""),
        ),
        (
            "initial_model_probability_bin",
            lambda row: _probability_bin(
                _finite(row.get("initial_model_probability")) or 0.0
            ),
        ),
        (
            "confirmation_band",
            lambda row: _confirmation_band(
                _integer(row.get("active_confirmation_count")) or 0
            ),
        ),
    ]
    output: list[dict[str, Any]] = []
    for dimension, key_fn in definitions:
        grouped: defaultdict[
            tuple[str, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in reactions:
            grouped[
                (
                    key_fn(row),
                    int(row.get("horizon_seconds") or 0),
                )
            ].append(row)
        for (value, horizon), rows in sorted(grouped.items()):
            valid = [
                row
                for row in rows
                if _truth(row.get("market_vector_valid"))
                and _truth(row.get("episode_survives_horizon"))
            ]
            day_deltas = _station_day_means(
                valid, "target_bucket_market_probability_delta"
            )
            ci = _bootstrap_mean_ci(day_deltas)
            resolved = [
                row
                for row in rows
                if row.get("target_won") not in {None, ""}
            ]
            output.append(
                {
                    "group_dimension": dimension,
                    "group_value": value,
                    "horizon_seconds": horizon,
                    "episodes": len(rows),
                    "independent_station_days": len(
                        {
                            (
                                str(row.get("station_id") or ""),
                                str(row.get("local_date") or ""),
                            )
                            for row in rows
                        }
                    ),
                    "valid_episode_reactions": len(valid),
                    "mean_target_bucket_market_delta": _mean(
                        [
                            parsed
                            for row in valid
                            if (
                                parsed := _finite(
                                    row.get(
                                        "target_bucket_market_probability_delta"
                                    )
                                )
                            )
                            is not None
                        ]
                    ),
                    "station_day_mean_target_delta": _mean(day_deltas),
                    "station_day_target_delta_ci95_low": (
                        ci["low"] if ci else None
                    ),
                    "station_day_target_delta_ci95_high": (
                        ci["high"] if ci else None
                    ),
                    "market_followed_target_fraction": _truth_fraction(
                        valid, "market_followed_target_bucket"
                    ),
                    "resolved_episodes": len(resolved),
                    "target_win_fraction": _truth_fraction(
                        resolved, "target_won"
                    ),
                }
            )
    return output


def _summary(
    events: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    episode_reactions: list[dict[str, Any]],
    episode_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    material = [
        row for row in events if _truth(row.get("material_model_update"))
    ]
    active = [
        row for row in material if _truth(row.get("active_information_update"))
    ]
    passive = [
        row
        for row in material
        if not _truth(row.get("active_information_update"))
    ]
    candidate_leads = [
        row for row in material if _truth(row.get("candidate_lead_update"))
    ]
    qualified_candidates = [
        row
        for row in material
        if _truth(row.get("qualified_candidate_update"))
    ]
    modal_candidates = [
        row
        for row in material
        if _truth(row.get("modal_candidate_update"))
    ]
    resolved = [
        row
        for row in active
        if _finite(row.get("final_brier_improvement")) is not None
    ]
    prospective_modal = [
        row
        for row in modal_candidates
        if str(row.get("evaluation_phase") or "") == "prospective"
    ]
    prospective_modal_lags = [
        value
        for row in prospective_modal
        if (value := _finite(row.get("processing_lag_seconds"))) is not None
    ]
    active_station_days = {
        (str(row["station_id"]), str(row["local_date"]))
        for row in active
    }
    candidate_station_days = {
        (str(row["station_id"]), str(row["local_date"]))
        for row in qualified_candidates
    }
    modal_station_days = {
        (str(row["station_id"]), str(row["local_date"]))
        for row in modal_candidates
    }
    overall_groups = [
        row
        for row in groups
        if row["group_dimension"] == "overall"
    ]
    active_groups = [
        row
        for row in groups
        if row["group_dimension"] == "activity_class"
        and row["group_value"] == "active_information"
    ]
    passive_groups = [
        row
        for row in groups
        if row["group_dimension"] == "activity_class"
        and row["group_value"] == "passive_scheduled_control"
    ]
    candidate_groups = [
        row
        for row in groups
        if row["group_dimension"] == "qualified_candidate_class"
        and row["group_value"] == "qualified_active_candidate"
    ]
    modal_groups = [
        row
        for row in groups
        if row["group_dimension"] == "modal_candidate_class"
        and row["group_value"] == "qualified_modal_candidate"
    ]
    candidate_rank_groups = [
        row
        for row in groups
        if row["group_dimension"] == "candidate_bucket_rank_band"
    ]
    episode_overall_groups = [
        row
        for row in episode_groups
        if row["group_dimension"] == "overall"
    ]
    episode_station_days = {
        (str(row["station_id"]), str(row["local_date"]))
        for row in episodes
    }
    resolved_episodes = [
        row
        for row in episodes
        if row.get("target_won") not in {None, ""}
    ]
    open_unresolved_episodes = [
        row
        for row in episodes
        if str(row.get("episode_end_reason") or "")
        == "still_open_at_last_observation"
        and not _truth(row.get("settled"))
    ]
    thirty_minute = next(
        (
            row
            for row in modal_groups
            if int(row["horizon_seconds"]) == 1800
        ),
        None,
    )
    enough_days = (
        len(modal_station_days) >= MINIMUM_INDEPENDENT_STATION_DAYS
    )
    timing_status = "insufficient_independent_station_days"
    if enough_days and thirty_minute is not None:
        ci_low = _finite(
            thirty_minute.get(
                "station_day_target_delta_ci95_low"
            )
        )
        timing_status = (
            "positive_repricing_evidence"
            if ci_low is not None and ci_low > 0.0
            else "no_robust_positive_repricing_evidence"
        )
    episode_thirty_minute = next(
        (
            row
            for row in episode_overall_groups
            if int(row["horizon_seconds"]) == 1800
        ),
        None,
    )
    episode_timing_status = "insufficient_independent_station_days"
    if (
        len(episode_station_days)
        >= MINIMUM_INDEPENDENT_STATION_DAYS
        and episode_thirty_minute is not None
    ):
        episode_ci_low = _finite(
            episode_thirty_minute.get(
                "station_day_target_delta_ci95_low"
            )
        )
        episode_timing_status = (
            "positive_persistent_repricing_evidence"
            if episode_ci_low is not None and episode_ci_low > 0.0
            else "no_robust_persistent_repricing_evidence"
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis": "station_maximum_v6_update_alpha",
        "purpose": (
            "Measure whether externally triggered V6 probability changes "
            "precede same-direction Polymarket repricing, separately from "
            "whether the update ultimately improves the resolved forecast."
        ),
        "method": {
            "signal_definition": (
                "Current V6 distribution minus the immediately preceding "
                "accepted V6 distribution for the same station-day."
            ),
            "materiality": {
                "absolute_point_delta_c": POINT_MATERIALITY_C,
                "probability_total_variation": (
                    PROBABILITY_TOTAL_VARIATION_MATERIALITY
                ),
            },
            "candidate_qualification": {
                "minimum_model_probability": (
                    MINIMUM_CANDIDATE_MODEL_PROBABILITY
                ),
                "bucket_selection": (
                    "Among buckets whose probability increased, choose the "
                    "largest positive V6-minus-market probability gap. This "
                    "is retained as a diagnostic comparator, not as the "
                    "primary tradable signal."
                ),
            },
            "modal_candidate_qualification": {
                "minimum_model_probability": (
                    MINIMUM_CANDIDATE_MODEL_PROBABILITY
                ),
                "bucket_selection": (
                    "Use only the V6 modal bucket when new external "
                    "information increases its probability and V6 still "
                    "prices it above the normalized market."
                ),
            },
            "market_reaction": (
                "Normalized CLOB midpoint distribution at 5/15/30/60/300/"
                "1800 seconds minus the decision-time distribution. Positive "
                "dot product means the complete market distribution moved in "
                "the model update's direction."
            ),
            "clock_scope": (
                "Market markouts currently start at source receipt time and "
                "therefore estimate information timing. Actual scoring lag is "
                "reported separately on prospective rows; operational "
                "markouts must not be inferred from replay rows."
            ),
            "causal_censoring": (
                "A horizon is excluded from attribution if another material "
                "V6 update for the same station-day arrived first."
            ),
            "persistent_episode": (
                "A primary candidate starts when new external information "
                "increases the V6 modal bucket while V6 still prices it above "
                "the market. Consecutive material states may reinforce the "
                "same modal bucket; the episode ends when the positive gap "
                "disappears or the modal target changes."
            ),
            "outcome_test": (
                "Brier change toward the official winning bucket, joined only "
                "after signal construction."
            ),
            "control": (
                "Scheduled/timer changes are retained as a passive-time "
                "control and are not treated as new external information."
            ),
            "independence": (
                "Confidence intervals bootstrap station-day means, not "
                "individual intraday updates."
            ),
        },
        "prediction_rows_with_previous_state": len(events),
        "material_updates": len(material),
        "active_information_updates": len(active),
        "passive_control_updates": len(passive),
        "passive_scheduled_controls": sum(
            str(row.get("decision_mode") or "") == "scheduled"
            for row in passive
        ),
        "duplicate_forecast_poll_controls": sum(
            str(row.get("decision_mode") or "") == "forecast_update"
            for row in passive
        ),
        "active_candidate_lead_updates": len(candidate_leads),
        "qualified_candidate_updates": len(qualified_candidates),
        "modal_candidate_updates": len(modal_candidates),
        "active_independent_station_days": len(active_station_days),
        "candidate_lead_independent_station_days": len(
            candidate_station_days
        ),
        "modal_candidate_independent_station_days": len(
            modal_station_days
        ),
        "prospective_modal_candidate_updates": len(prospective_modal),
        "prospective_modal_processing_lag_seconds": {
            "median": _percentile(prospective_modal_lags, 0.50),
            "p90": _percentile(prospective_modal_lags, 0.90),
            "maximum": max(prospective_modal_lags)
            if prospective_modal_lags
            else None,
        },
        "resolved_active_updates": len(resolved),
        "active_updates_improving_final_outcome": sum(
            _truth(row.get("update_improved_final_outcome"))
            for row in resolved
        ),
        "market_reaction_rows": len(reactions),
        "valid_market_reaction_rows": sum(
            _truth(row.get("market_vector_valid"))
            and _truth(
                row.get("uncontaminated_by_later_model_update")
            )
            for row in reactions
        ),
        "timing_evidence_status": timing_status,
        "candidate_episodes": len(episodes),
        "episode_independent_station_days": len(
            episode_station_days
        ),
        "resolved_candidate_episodes": len(resolved_episodes),
        "winning_candidate_episodes": sum(
            _truth(row.get("target_won")) for row in resolved_episodes
        ),
        "open_unresolved_modal_candidates": len(
            open_unresolved_episodes
        ),
        "episode_reaction_rows": len(episode_reactions),
        "episode_timing_evidence_status": episode_timing_status,
        "minimum_independent_station_days_for_timing_claim": (
            MINIMUM_INDEPENDENT_STATION_DAYS
        ),
        "alpha_demonstrated": False,
        "economic_pnl_evaluated": False,
        "operational_market_timing_evaluated": False,
        "warning": (
            "Same-direction repricing from source receipt is evidence about "
            "information timing, not proof that the current batch pipeline "
            "could act before repricing. Ask, spread, fills and fees are "
            "deliberately excluded from this diagnostic."
        ),
        "overall_by_horizon": overall_groups,
        "active_information_by_horizon": active_groups,
        "active_candidate_lead_by_horizon": candidate_groups,
        "qualified_candidate_by_horizon": candidate_groups,
        "qualified_candidate_by_rank_and_horizon": candidate_rank_groups,
        "modal_candidate_by_horizon": modal_groups,
        "passive_control_by_horizon": passive_groups,
        "candidate_episode_by_horizon": episode_overall_groups,
        "live_orders_enabled": False,
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# V6 update-alpha audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        (
            f"Material external updates: "
            f"{summary['active_information_updates']} across "
            f"{summary['active_independent_station_days']} independent "
            "station-days."
        ),
        (
            f"Diagnostic max-gap updates (model probability at least 20% "
            f"and positive model-market gap): "
            f"{summary['qualified_candidate_updates']} across "
            f"{summary['candidate_lead_independent_station_days']} "
            "station-days."
        ),
        (
            f"Primary modal-bucket updates: "
            f"{summary['modal_candidate_updates']} across "
            f"{summary['modal_candidate_independent_station_days']} "
            "station-days."
        ),
        (
            "Prospective modal updates with measurable real scoring lag: "
            f"{summary['prospective_modal_candidate_updates']}."
        ),
        (
            f"Passive scheduled controls: "
            f"{summary['passive_scheduled_controls']}."
        ),
        (
            f"Persistent modal-bucket episodes: "
            f"{summary['candidate_episodes']}; resolved: "
            f"{summary['resolved_candidate_episodes']}; winning target: "
            f"{summary['winning_candidate_episodes']}."
        ),
        (
            f"Resolved external updates: "
            f"{summary['resolved_active_updates']}; updates improving the "
            f"final Brier score: "
            f"{summary['active_updates_improving_final_outcome']}."
        ),
        "",
        f"Timing status: **{summary['timing_evidence_status']}**.",
        "",
        "| Horizon | Valid reactions | Target followed | Mean target delta | "
        "Station-day target delta | 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["modal_candidate_by_horizon"]:
        low = row.get("station_day_target_delta_ci95_low")
        high = row.get("station_day_target_delta_ci95_high")
        ci = (
            f"[{low:.5f}, {high:.5f}]"
            if low is not None and high is not None
            else "n/a"
        )
        followed = row.get("market_followed_target_fraction")
        mean_delta = row.get("mean_target_bucket_market_delta")
        day_delta = row.get("station_day_mean_target_delta")
        lines.append(
            f"| {row['horizon_seconds']}s | "
            f"{row['valid_market_reactions']} | "
            f"{followed:.1%} | "
            f"{mean_delta:.5f} | "
            f"{day_delta:.5f} | {ci} |"
            if followed is not None
            and mean_delta is not None
            and day_delta is not None
            else (
                f"| {row['horizon_seconds']}s | "
                f"{row['valid_market_reactions']} | n/a | n/a | n/a | {ci} |"
            )
        )
    lines.extend(
        [
            "",
            "## Why max-gap is only a diagnostic",
            "",
            (
                "The largest positive model-market gap can belong to a "
                "secondary tail bucket. The primary signal therefore requires "
                "the target to be the V6 modal bucket; rank-stratified max-gap "
                "results remain available in update_alpha_groups.csv."
            ),
            "",
            "## Persistent episodes",
            "",
            "| Horizon | Valid episodes | Target followed | "
            "Station-day target delta | 95% CI |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["candidate_episode_by_horizon"]:
        low = row.get("station_day_target_delta_ci95_low")
        high = row.get("station_day_target_delta_ci95_high")
        ci = (
            f"[{low:.5f}, {high:.5f}]"
            if low is not None and high is not None
            else "n/a"
        )
        followed = row.get("market_followed_target_fraction")
        day_delta = row.get("station_day_mean_target_delta")
        lines.append(
            f"| {row['horizon_seconds']}s | "
            f"{row['valid_episode_reactions']} | "
            f"{followed:.1%} | {day_delta:.5f} | {ci} |"
            if followed is not None and day_delta is not None
            else (
                f"| {row['horizon_seconds']}s | "
                f"{row['valid_episode_reactions']} | n/a | n/a | {ci} |"
            )
        )
    lines.extend(
        [
            "",
            (
                "This report does not estimate P&L. It tests whether the model "
                "information changes before the market and whether those "
                "changes later help the resolved forecast. The markout clock "
                "starts at source receipt, not at completion of the batch "
                "scoring cycle."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _information_class(
    mode: str, trigger: str, forecast_content_changed: Any = None
) -> str:
    if mode == "scheduled":
        return "passive_clock_control"
    if trigger.startswith("forecast_"):
        return (
            "forecast_provider_revision"
            if _truth(forecast_content_changed)
            else "forecast_poll_no_content_change"
        )
    if trigger == "knmi_10m":
        return "official_station_observation"
    if trigger in {
        "awc_metar",
        "nws_tgftp_metar",
        "wis2_metar",
    }:
        return "aviation_station_observation"
    return "other_external_update"


def _active_information_update(
    mode: str, forecast_content_changed: Any
) -> bool:
    if mode == "scheduled":
        return False
    if mode == "forecast_update":
        return _truth(forecast_content_changed)
    return True


def _local_hour_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    hour = int(value)
    if hour < 6:
        return "00-05"
    if hour < 9:
        return "06-08"
    if hour < 12:
        return "09-11"
    if hour < 15:
        return "12-14"
    if hour < 18:
        return "15-17"
    return "18-23"


def _gap_bin(value: float) -> str:
    if value <= 0.0:
        return "<=0"
    if value < 0.02:
        return "0-2%"
    if value < 0.05:
        return "2-5%"
    if value < 0.10:
        return "5-10%"
    if value < 0.20:
        return "10-20%"
    return ">=20%"


def _confirmation_band(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _rank_band(value: int) -> str:
    if value <= 1:
        return "1_modal"
    if value == 2:
        return "2_second"
    return "3+_tail"


def _probability_bin(value: float) -> str:
    if value < 0.20:
        return "<20%"
    if value < 0.35:
        return "20-35%"
    if value < 0.50:
        return "35-50%"
    if value < 0.70:
        return "50-70%"
    return ">=70%"


def _station_day_means(
    rows: list[dict[str, Any]], field: str
) -> list[float]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(
        list
    )
    for row in rows:
        value = _finite(row.get(field))
        if value is not None:
            grouped[
                (
                    str(row.get("station_id") or ""),
                    str(row.get("local_date") or ""),
                )
            ].append(value)
    return [
        sum(values) / len(values)
        for values in grouped.values()
        if values
    ]


def _bucket_brier(
    distribution: dict[str, float], winner: str
) -> float | None:
    if not distribution or not winner:
        return None
    labels = set(distribution) | {winner}
    return sum(
        (
            distribution.get(label, 0.0)
            - (1.0 if label == winner else 0.0)
        )
        ** 2
        for label in labels
    )


def _valid_market_mass(value: float) -> bool:
    return MINIMUM_MARKET_MID_MASS <= value <= MAXIMUM_MARKET_MID_MASS


def _cosine_similarity(
    left: dict[str, float], right: dict[str, float]
) -> float | None:
    labels = set(left) | set(right)
    left_norm = math.sqrt(
        sum(left.get(label, 0.0) ** 2 for label in labels)
    )
    right_norm = math.sqrt(
        sum(right.get(label, 0.0) ** 2 for label in labels)
    )
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return sum(
        left.get(label, 0.0) * right.get(label, 0.0)
        for label in labels
    ) / (left_norm * right_norm)


def _truth_fraction(
    rows: list[dict[str, Any]], field: str
) -> float | None:
    values = [
        row.get(field)
        for row in rows
        if row.get(field) not in {None, ""}
    ]
    if not values:
        return None
    return sum(_truth(value) for value in values) / len(values)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(
        len(ordered) - 1,
        max(0, round(quantile * (len(ordered) - 1))),
    )
    return ordered[position]


def _json_probabilities(value: Any) -> dict[str, float]:
    return {
        label: parsed
        for label, parsed in _json_numbers(value).items()
        if parsed >= 0.0
    }


def _json_numbers(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(label): parsed
        for label, value in raw.items()
        if (parsed := _finite(value)) is not None
    }


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
