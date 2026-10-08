from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig
from ..parsing.buckets import parse_temperature_bucket
from .model import _bootstrap_mean_ci
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import _load_validation_rows
from .station_adaptive_alpha import (
    STATION_ADAPTIVE_LANES,
    _recent_dates_by_station,
    _weather_centered_locked_probability,
)
from .station_specific_alpha import (
    _load_historical_snapshots,
    _truth,
    evaluate_station_policy_snapshot,
)

EDGE_BANDS = (
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.12, 0.20),
    (0.14, 0.20),
    (0.15, 0.20),
    (0.16, 0.20),
    (0.18, 0.20),
    (0.15, 0.22),
    (0.15, 0.25),
    (0.10, 0.20),
    (0.20, 1.00),
    (0.15, 1.00),
)
MINIMUM_PRIOR_LANE_SIGNALS = (0, 5, 10, 15, 20)
FORECAST_SPREAD_LIMITS_C = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


def analyze_station_tail_diagnostics(cfg: AppConfig) -> dict[str, Path]:

    snapshots = _load_historical_snapshots(cfg)
    split_dates = _recent_dates_by_station(snapshots)
    validation_rows, validation_paths = _load_validation_rows(cfg)
    feature_rows = _load_feature_rows(cfg)
    candidates = _build_candidates(
        cfg,
        snapshots,
        split_dates,
        validation_rows,
        feature_rows,
    )

    fixed_trades = _replay(
        candidates,
        probability_field="fixed_lane_probability",
        edge_field="fixed_edge",
        minimum_edge=0.15,
        maximum_edge=0.20,
    )
    chronological_trades = _replay(
        candidates,
        probability_field="chronological_lane_probability",
        edge_field="chronological_edge",
        minimum_edge=0.15,
        maximum_edge=0.20,
        minimum_prior_signals=20,
    )

    sensitivity_rows = _edge_sensitivity(candidates)
    spread_rows = _spread_sensitivity(candidates)
    trade_diagnostics = _trade_diagnostics(fixed_trades, cfg)
    warning_prevalence = _warning_prevalence(trade_diagnostics)
    live_updates = _live_update_inventory(cfg)

    output_dir = cfg.reports_dir / "station_tail_diagnostics_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_path = output_dir / "fixed_15_20_trade_diagnostics.csv"
    edge_path = output_dir / "edge_parameter_sensitivity.csv"
    spread_path = output_dir / "forecast_spread_sensitivity.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"

    _write_csv_atomic(trade_path, trade_diagnostics)
    _write_csv_atomic(edge_path, sensitivity_rows)
    _write_csv_atomic(spread_path, spread_rows)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis": "station_tail_diagnostics_v3",
        "purpose": (
            "Diagnose whether upper-bucket misses were recognizable point-in-time "
            "model failures or irreducible residual uncertainty. This report does "
            "not optimize a rule to make the known losses disappear."
        ),
        "historical_snapshots": len(snapshots),
        "eligible_lane_candidates": len(candidates),
        "fixed_full_sample_15_20": _summarize_trades(fixed_trades),
        "strictly_chronological_15_20_minimum_20_prior_lane_signals": (
            _summarize_trades(chronological_trades)
        ),
        "fixed_replay_warning": (
            "The fixed lane probabilities use the complete historical sample. "
            "They are valid for the current prospective deployment, but their "
            "historical replay contains look-ahead and is not an alpha estimate."
        ),
        "chronological_replay_definition": (
            "For each date, the lane probability is a weak weather-centered Beta "
            "estimate using only earlier eligible, resolved dates in that lane."
        ),
        "known_loss_diagnostics": [
            row for row in trade_diagnostics if not _truth(row.get("won"))
        ],
        "warning_prevalence": warning_prevalence,
        "configured_maximum_forecast_spread_c": (
            cfg.paper_trading.maximum_forecast_spread_c
        ),
        "quality_gate_gap": (
            "The frozen station-adaptive/edge-band layer consumes challenger rows "
            "but does not currently inherit upstream rejection reasons such as "
            "forecast_consensus_too_wide. It remains unchanged as a control."
        ),
        "live_update_pipeline": live_updates,
        "validation_files": [str(path) for path in validation_paths],
        "live_orders_enabled": False,
    }
    _write_json_atomic(summary_path, summary)
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {
        "summary": summary_path,
        "report": report_path,
        "trade_diagnostics": trade_path,
        "edge_sensitivity": edge_path,
        "spread_sensitivity": spread_path,
    }


def _build_candidates(
    cfg: AppConfig,
    snapshots: list[dict[str, Any]],
    split_dates: dict[str, set[str]],
    validation_rows: dict[tuple[str, str, str, str], dict[str, str]],
    feature_rows: dict[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    station_units = {
        station.station_id: station.market_unit for station in cfg.stations.values()
    }
    candidates_by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for station_id, lanes in STATION_ADAPTIVE_LANES.items():
        for lane in lanes:
            for snapshot in snapshots:
                if (
                    snapshot["station_id"] != station_id
                    or snapshot["local_hour"] != lane["target_local_hour"]
                ):
                    continue
                evaluation = evaluate_station_policy_snapshot(
                    lane,
                    snapshot["weather_distribution"],
                    snapshot["market_distribution"],
                )
                if not evaluation["eligible"]:
                    continue
                key = (
                    station_id,
                    str(snapshot["local_date"]),
                    "historical_scheduled",
                    str(snapshot["received_wall_ns"]),
                )
                validation = validation_rows.get(key, {})
                features = feature_rows.get(key, {})
                unit = station_units[station_id]
                candidate_bucket = str(evaluation["candidate_bucket"])
                winner = str(snapshot["winning_bucket"])
                candidate_high = _bucket_high(candidate_bucket, unit)
                forecast_point = _safe_float(
                    validation.get("forecast_consensus_prediction")
                )
                model_point = _safe_float(validation.get("point_prediction"))
                current_max = _safe_float(validation.get("current_running_max"))
                market_upper_tail = _upper_tail_probability(
                    snapshot["market_distribution"], candidate_bucket, unit
                )
                weather_upper_tail = _upper_tail_probability(
                    snapshot["weather_distribution"], candidate_bucket, unit
                )
                fixed_cost = float(lane["fixed_cost_per_share"])
                configured_peer_count = len(
                    next(
                        station.peer_station_ids
                        for station in cfg.stations.values()
                        if station.station_id == station_id
                    )
                )
                peer_available_count = _safe_float(
                    features.get("peer_available_count")
                )
                row = {
                    "station_id": station_id,
                    "local_date": str(snapshot["local_date"]),
                    "split": (
                        "recent_diagnostic"
                        if snapshot["local_date"] in split_dates[station_id]
                        else "development_diagnostic"
                    ),
                    "policy_id": str(lane["policy_id"]),
                    "local_hour": int(snapshot["local_hour"]),
                    "candidate_bucket": candidate_bucket,
                    "winning_bucket": winner,
                    "won": candidate_bucket == winner,
                    "adjacent_upper_bucket_miss": _is_adjacent_upper_bucket(
                        candidate_bucket, winner, unit
                    ),
                    "candidate_price_proxy": float(
                        evaluation["candidate_price_proxy"]
                    ),
                    "fixed_cost_per_share": fixed_cost,
                    "synthetic_cost_per_share": (
                        float(evaluation["candidate_price_proxy"]) + fixed_cost
                    ),
                    "raw_snapshot_probability": float(
                        evaluation["candidate_weather_probability"]
                    ),
                    "market_upper_tail_probability": market_upper_tail,
                    "weather_upper_tail_probability": weather_upper_tail,
                    "market_minus_weather_upper_tail": (
                        market_upper_tail - weather_upper_tail
                    ),
                    "forecast_consensus_prediction": forecast_point,
                    "model_point_prediction": model_point,
                    "current_running_max": current_max,
                    "forecast_minus_candidate_high": _difference(
                        forecast_point, candidate_high
                    ),
                    "model_point_minus_candidate_high": _difference(
                        model_point, candidate_high
                    ),
                    "candidate_high_minus_current_max": _difference(
                        candidate_high, current_max
                    ),
                    "raw_probability_rise_again": _safe_float(
                        validation.get("raw_probability_rise_again")
                    ),
                    "calibrated_probability_rise_again": _safe_float(
                        validation.get("probability_rise_again")
                    ),
                    "forecast_spread_remaining_max_c": _safe_float(
                        features.get("historical_model_spread_remaining_max_c")
                    ),
                    "temperature_change_1h": _safe_float(
                        features.get("temperature_change_1h")
                    ),
                    "temperature_change_3h": _safe_float(
                        features.get("temperature_change_3h")
                    ),
                    "hours_since_running_max": _safe_float(
                        features.get("hours_since_running_max")
                    ),
                    "peer_temperature_change_1h_mean": _safe_float(
                        features.get("peer_temperature_change_1h_mean")
                    ),
                    "peer_temperature_mean_c": _safe_float(
                        features.get("peer_temperature_mean_c")
                    ),
                    "peer_temperature_spread_c": _safe_float(
                        features.get("peer_temperature_spread_c")
                    ),
                    "peer_running_max_mean_c": _safe_float(
                        features.get("peer_running_max_mean_c")
                    ),
                    "peer_age_minutes_mean": _safe_float(
                        features.get("peer_age_minutes_mean")
                    ),
                    "peer_available_count": peer_available_count,
                    "peer_station_coverage": (
                        peer_available_count / configured_peer_count
                        if peer_available_count is not None
                        and configured_peer_count
                        else None
                    ),
                    "peer_history_point_in_time": _truth(
                        features.get("peer_history_point_in_time")
                    ),
                    "temperature_c": _safe_float(
                        features.get("temperature_c")
                    ),
                    "day_of_year_sin": _safe_float(
                        features.get("day_of_year_sin")
                    ),
                    "day_of_year_cos": _safe_float(
                        features.get("day_of_year_cos")
                    ),
                    "local_time_sin": _safe_float(
                        features.get("local_time_sin")
                    ),
                    "local_time_cos": _safe_float(
                        features.get("local_time_cos")
                    ),
                    "minutes_from_solar_noon": _safe_float(
                        features.get("minutes_from_solar_noon")
                    ),
                    "solar_elevation_deg": _safe_float(
                        features.get("solar_elevation_deg")
                    ),
                    "received_wall_ns": int(snapshot["received_wall_ns"]),
                }
                candidates_by_lane[str(lane["policy_id"])].append(row)

    output: list[dict[str, Any]] = []
    for lane_id, rows in candidates_by_lane.items():
        rows.sort(key=lambda row: (str(row["local_date"]), int(row["received_wall_ns"])))
        fixed_probability = _weather_centered_locked_probability(
            [
                {
                    "won": row["won"],
                    "candidate_weather_probability": row[
                        "raw_snapshot_probability"
                    ],
                }
                for row in rows
            ]
        )
        prior: list[dict[str, Any]] = []
        for row in rows:
            chronological_probability = _prior_probability(
                prior, float(row["raw_snapshot_probability"])
            )
            row["fixed_lane_probability"] = fixed_probability
            row["chronological_lane_probability"] = chronological_probability
            row["chronological_prior_lane_signals"] = len(prior)
            row["fixed_edge"] = (
                fixed_probability - float(row["synthetic_cost_per_share"])
            )
            row["chronological_edge"] = (
                chronological_probability
                - float(row["synthetic_cost_per_share"])
            )
            row["lane_probability_minus_snapshot_probability"] = (
                fixed_probability - float(row["raw_snapshot_probability"])
            )
            row["forecast_above_candidate"] = bool(
                _positive(row.get("forecast_minus_candidate_high"))
            )
            row["market_upper_tail_above_weather"] = bool(
                _positive(row.get("market_minus_weather_upper_tail"))
            )
            row["forecast_spread_above_configured_limit"] = bool(
                (
                    spread := _safe_float(
                        row.get("forecast_spread_remaining_max_c")
                    )
                )
                is not None
                and spread > cfg.paper_trading.maximum_forecast_spread_c
            )
            row["lane_probability_materially_above_snapshot"] = bool(
                float(row["lane_probability_minus_snapshot_probability"])
                >= 0.05
            )
            row["low_absolute_lane_probability"] = fixed_probability < 0.50
            output.append(row)
            prior.append(row)
    return output


def _prior_probability(prior: list[dict[str, Any]], current_raw: float) -> float:
    if not prior:
        return current_raw
    wins = sum(_truth(row.get("won")) for row in prior)
    raw_center = sum(
        float(row["raw_snapshot_probability"]) for row in prior
    ) / len(prior)
    return (wins + 2.0 * raw_center) / (len(prior) + 2.0)


def _replay(
    candidates: list[dict[str, Any]],
    *,
    probability_field: str,
    edge_field: str,
    minimum_edge: float,
    maximum_edge: float,
    minimum_prior_signals: int = 0,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    by_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_day[(str(row["station_id"]), str(row["local_date"]))].append(row)

    trades: list[dict[str, Any]] = []
    for day_rows in by_day.values():
        eligible = [
            row
            for row in day_rows
            if int(row["chronological_prior_lane_signals"])
            >= minimum_prior_signals
            and minimum_edge <= float(row[edge_field]) < maximum_edge
            and (predicate is None or predicate(row))
        ]
        if not eligible:
            continue
        first_hour = min(int(row["local_hour"]) for row in eligible)
        chosen = max(
            (row for row in eligible if int(row["local_hour"]) == first_hour),
            key=lambda row: float(row[edge_field]),
        )
        trade = dict(chosen)
        trade["strategy_probability"] = float(chosen[probability_field])
        trade["strategy_edge"] = float(chosen[edge_field])
        trade["synthetic_value_per_share"] = (
            (1.0 if _truth(chosen["won"]) else 0.0)
            - float(chosen["synthetic_cost_per_share"])
        )
        trades.append(trade)
    return sorted(
        trades,
        key=lambda row: (str(row["local_date"]), str(row["station_id"])),
    )


def _edge_sensitivity(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for minimum_edge, maximum_edge in EDGE_BANDS:
        fixed = _replay(
            candidates,
            probability_field="fixed_lane_probability",
            edge_field="fixed_edge",
            minimum_edge=minimum_edge,
            maximum_edge=maximum_edge,
        )
        output.append(
            {
                "probability_mode": "fixed_complete_sample_diagnostic",
                "minimum_prior_lane_signals": 0,
                "minimum_edge_inclusive": minimum_edge,
                "maximum_edge_exclusive": maximum_edge,
                **_flat_summary(fixed),
            }
        )
        for minimum_prior in MINIMUM_PRIOR_LANE_SIGNALS:
            chronological = _replay(
                candidates,
                probability_field="chronological_lane_probability",
                edge_field="chronological_edge",
                minimum_edge=minimum_edge,
                maximum_edge=maximum_edge,
                minimum_prior_signals=minimum_prior,
            )
            output.append(
                {
                    "probability_mode": "strictly_chronological",
                    "minimum_prior_lane_signals": minimum_prior,
                    "minimum_edge_inclusive": minimum_edge,
                    "maximum_edge_exclusive": maximum_edge,
                    **_flat_summary(chronological),
                }
            )
    return output


def _spread_sensitivity(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    modes = (
        (
            "fixed_complete_sample_diagnostic",
            "fixed_lane_probability",
            "fixed_edge",
            0,
        ),
        (
            "strictly_chronological",
            "chronological_lane_probability",
            "chronological_edge",
            20,
        ),
    )
    for mode, probability_field, edge_field, minimum_prior in modes:
        baseline = _replay(
            candidates,
            probability_field=probability_field,
            edge_field=edge_field,
            minimum_edge=0.15,
            maximum_edge=0.20,
            minimum_prior_signals=minimum_prior,
        )
        output.append(
            {
                "probability_mode": mode,
                "minimum_prior_lane_signals": minimum_prior,
                "maximum_forecast_spread_c": "",
                "filter": "none",
                **_flat_summary(baseline),
            }
        )
        for maximum_spread in FORECAST_SPREAD_LIMITS_C:
            trades = _replay(
                candidates,
                probability_field=probability_field,
                edge_field=edge_field,
                minimum_edge=0.15,
                maximum_edge=0.20,
                minimum_prior_signals=minimum_prior,
                predicate=lambda row, limit=maximum_spread: (
                    (
                        value := _safe_float(
                            row.get("forecast_spread_remaining_max_c")
                        )
                    )
                    is not None
                    and value <= limit
                ),
            )
            output.append(
                {
                    "probability_mode": mode,
                    "minimum_prior_lane_signals": minimum_prior,
                    "maximum_forecast_spread_c": maximum_spread,
                    "filter": "point_in_time_forecast_quality",
                    **_flat_summary(trades),
                }
            )
    return output


def _trade_diagnostics(
    trades: list[dict[str, Any]], cfg: AppConfig
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trade in trades:
        causes: list[str] = []
        if trade["low_absolute_lane_probability"]:
            causes.append("low_probability_disagreement_expected_variance")
        if trade["forecast_spread_above_configured_limit"]:
            causes.append("forecast_provider_spread_above_quality_limit")
        if trade["lane_probability_materially_above_snapshot"]:
            causes.append("lane_average_overrode_snapshot_uncertainty")
        if trade["forecast_above_candidate"]:
            causes.append("forecast_consensus_above_candidate")
        if trade["market_upper_tail_above_weather"]:
            causes.append("market_carried_more_upper_tail_than_weather")
        if trade["adjacent_upper_bucket_miss"] and not (
            trade["low_absolute_lane_probability"]
            or trade["forecast_spread_above_configured_limit"]
            or trade["lane_probability_materially_above_snapshot"]
        ):
            causes.append("plausible_adjacent_bucket_residual")
        row = dict(trade)
        row["diagnostic_cause_tags"] = "|".join(causes)
        row["configured_maximum_forecast_spread_c"] = (
            cfg.paper_trading.maximum_forecast_spread_c
        )
        output.append(row)
    return output


def _warning_prevalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if _truth(row.get("won"))]
    losses = [row for row in rows if not _truth(row.get("won"))]
    flags = (
        "forecast_above_candidate",
        "market_upper_tail_above_weather",
        "forecast_spread_above_configured_limit",
        "lane_probability_materially_above_snapshot",
        "low_absolute_lane_probability",
    )
    return {
        flag: {
            "wins_flagged": sum(_truth(row.get(flag)) for row in wins),
            "wins_total": len(wins),
            "losses_flagged": sum(_truth(row.get(flag)) for row in losses),
            "losses_total": len(losses),
        }
        for flag in flags
    }


def _summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_available": _summary_block(trades),
        "development_diagnostic": _summary_block(
            [
                row
                for row in trades
                if row["split"] == "development_diagnostic"
            ]
        ),
        "recent_diagnostic": _summary_block(
            [
                row for row in trades if row["split"] == "recent_diagnostic"
            ]
        ),
        "by_station": {
            station_id: _summary_block(
                [
                    row
                    for row in trades
                    if row["station_id"] == station_id
                ]
            )
            for station_id in STATION_ADAPTIVE_LANES
        },
    }


def _summary_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["synthetic_value_per_share"]) for row in rows]
    stake = sum(float(row["synthetic_cost_per_share"]) for row in rows)
    total = sum(values)
    return {
        "trades": len(rows),
        "wins": sum(_truth(row.get("won")) for row in rows),
        "hit_rate": (
            sum(_truth(row.get("won")) for row in rows) / len(rows)
            if rows
            else None
        ),
        "synthetic_stake": stake,
        "synthetic_value_total": total,
        "synthetic_return_on_stake": total / stake if stake else None,
        "mean_value_per_share": total / len(rows) if rows else None,
        "mean_value_ci95": _bootstrap_mean_ci(values) if values else None,
    }


def _flat_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _summary_block(rows)
    development = _summary_block(
        [
            row
            for row in rows
            if row["split"] == "development_diagnostic"
        ]
    )
    recent = _summary_block(
        [row for row in rows if row["split"] == "recent_diagnostic"]
    )
    return {
        "trades": overall["trades"],
        "wins": overall["wins"],
        "hit_rate": overall["hit_rate"],
        "synthetic_stake": overall["synthetic_stake"],
        "synthetic_value_total": overall["synthetic_value_total"],
        "synthetic_return_on_stake": overall["synthetic_return_on_stake"],
        "mean_value_ci95_low": (
            overall["mean_value_ci95"]["low"]
            if overall["mean_value_ci95"]
            else None
        ),
        "mean_value_ci95_high": (
            overall["mean_value_ci95"]["high"]
            if overall["mean_value_ci95"]
            else None
        ),
        "development_trades": development["trades"],
        "development_value": development["synthetic_value_total"],
        "recent_trades": recent["trades"],
        "recent_value": recent["synthetic_value_total"],
    }


def _live_update_inventory(cfg: AppConfig) -> dict[str, Any]:
    stations = list(cfg.stations.values())

    def enabled(source: str) -> list[str]:
        return sorted(
            station.station_id
            for station in stations
            if bool(getattr(station.sources, source))
        )

    return {
        "weather_observations": {
            "awc_metar": {
                "stations": enabled("awc"),
                "mode": "HTTP polling of target and configured peer METARs",
                "poll_seconds": cfg.collector.awc_poll_seconds,
            },
            "nws_tgftp_metar": {
                "stations": enabled("nws_tgftp"),
                "mode": "fallback/reconciliation HTTP polling",
                "poll_seconds": cfg.collector.nws_tgftp_poll_seconds,
            },
            "wis2_metar": {
                "stations": enabled("wis2"),
                "mode": "MQTT push parser; currently disabled in station configuration",
            },
            "knmi_10m": {
                "stations": enabled("knmi"),
                "mode": (
                    "MQTT file-created notification plus download"
                    if os.getenv("KNMI_NOTIFICATION_API_KEY", "").strip()
                    else "Open Data polling"
                ),
                "reconciliation_poll_seconds": cfg.collector.knmi_poll_seconds,
            },
        },
        "forecast_updates": {
            "poll_seconds": cfg.collector.forecast_poll_seconds,
            "providers_by_station": {
                station.station_id: list(station.forecast_providers)
                for station in stations
                if station.sources.forecasts
            },
            "snapshot_rule": "persist only when provider payload content changes",
            "revision_rule": (
                "emit a revision when today's or remaining hourly maximum changes"
            ),
        },
        "resolution_page": {
            "stations": enabled("wunderground"),
            "poll_seconds": cfg.collector.wunderground_poll_seconds,
            "actual_use": (
                "final resolution label and point-in-time resolution-running-max "
                "feature used by the current paper model"
            ),
        },
        "market": {
            "mode": "continuous Polymarket CLOB WebSocket",
            "routine_persistence": (
                f"top {cfg.market.compact_depth_levels} levels per asset every "
                f"{cfg.market.compact_interval_seconds:g} seconds"
            ),
            "forecast_revision_capture_window_seconds": {
                "before": cfg.market.burst_before_seconds,
                "after": cfg.market.burst_after_seconds,
            },
        },
        "decision_rows_created": {
            "event_driven": "on each new non-backfill primary weather observation",
            "forecast_update": "on each changed provider forecast snapshot",
            "scheduled": (
                f"every {cfg.predictive.scheduled_interval_minutes} minutes"
            ),
        },
        "current_edge_band_behavior": {
            "accepted_decision_modes": ["scheduled"],
            "entry_hours": {
                station_id: sorted(
                    {int(lane["target_local_hour"]) for lane in lanes}
                )
                for station_id, lanes in STATION_ADAPTIVE_LANES.items()
            },
            "maximum_entries_per_station_day": 1,
            "after_entry": (
                "record and hold the paper position to resolution; later updates "
                "do not resize, exit, reverse, or open a second station-day trade"
            ),
            "live_orders_enabled": False,
        },
    }


def _load_feature_rows(
    cfg: AppConfig,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    paths = sorted(
        cfg.reports_dir.glob("*/predictive_features.csv"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    records: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                timestamp = _timestamp_text(row.get("received_wall_ns"))
                if not timestamp:
                    continue
                key = (
                    str(row.get("station_id") or ""),
                    str(row.get("local_date") or ""),
                    str(row.get("decision_mode") or ""),
                    timestamp,
                )
                records[key] = row
    return records


def _timestamp_text(value: Any) -> str:
    try:
        return str(int(Decimal(str(value))))
    except (InvalidOperation, TypeError, ValueError):
        return ""


def _upper_tail_probability(
    distribution: dict[str, float],
    candidate: str,
    unit: str,
) -> float:
    candidate_high = _bucket_high(candidate, unit)
    if candidate_high is None:
        return 0.0
    total = 0.0
    for label, probability in distribution.items():
        try:
            bucket = parse_temperature_bucket(str(label), unit)
        except ValueError:
            continue
        if bucket.low is not None and bucket.low > candidate_high:
            total += float(probability)
    return total


def _bucket_high(label: str, unit: str) -> float | None:
    try:
        return parse_temperature_bucket(label, unit).high
    except ValueError:
        return None


def _is_adjacent_upper_bucket(candidate: str, winner: str, unit: str) -> bool:
    try:
        candidate_bucket = parse_temperature_bucket(candidate, unit)
        winner_bucket = parse_temperature_bucket(winner, unit)
    except ValueError:
        return False
    if candidate_bucket.high is None or winner_bucket.low is None:
        return False
    width = (
        candidate_bucket.high - candidate_bucket.low + 1.0
        if candidate_bucket.low is not None
        else 1.0
    )
    gap = winner_bucket.low - candidate_bucket.high
    return 0.0 < gap <= max(width, 2.0)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _positive(value: Any) -> bool:
    parsed = _safe_float(value)
    return parsed is not None and parsed > 0.0


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _markdown_report(summary: dict[str, Any]) -> str:
    fixed = summary["fixed_full_sample_15_20"]["all_available"]
    chronological = summary[
        "strictly_chronological_15_20_minimum_20_prior_lane_signals"
    ]["all_available"]
    losses = summary["known_loss_diagnostics"]
    lines = [
        "# Station tail diagnostics V3",
        "",
        "This is a causal/robustness diagnostic, not a rule fitted to erase four losses.",
        "",
        "## Backtest distinction",
        "",
        (
            f"- Fixed complete-sample replay: {fixed['trades']} trades, "
            f"{fixed['wins']} wins, synthetic value "
            f"{fixed['synthetic_value_total']:.3f}."
        ),
        (
            f"- Strictly chronological replay with at least 20 prior lane signals: "
            f"{chronological['trades']} trades, {chronological['wins']} wins, "
            f"synthetic value {chronological['synthetic_value_total']:.3f}."
        ),
        "",
        "The first number contains look-ahead in the lane probability. The second is the "
        "appropriate historical robustness check.",
        "",
        "## Known losses",
        "",
    ]
    for row in losses:
        lines.append(
            f"- {row['station_id']} {row['local_date']}: "
            f"{row['candidate_bucket']} -> {row['winning_bucket']}; "
            f"{row['diagnostic_cause_tags']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Forecast-above-candidate and market-more-upper-tail warnings also occur on "
            "most winning trades, so using either as a hard veto would be overfit. A "
            "forecast-spread quality gate is physically defensible and already exists "
            "upstream, but the frozen edge-band layer does not inherit that rejection.",
            "",
            "The next model should estimate conditional upper-tail probability per snapshot "
            "and preserve the frozen paper strategy as a control.",
            "",
            "No live order is sent by any component described here.",
            "",
        ]
    )
    return "\n".join(lines)
