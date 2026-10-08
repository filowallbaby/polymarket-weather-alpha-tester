from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from ..parsing.buckets import c_to_f, parse_temperature_bucket
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import _probabilities, _timestamp_ns
from .station_adaptive_alpha import STATION_ADAPTIVE_LANES
from .station_specific_alpha import (
    _load_csv,
    _read_json,
    _settle_trades,
    _truth,
    evaluate_station_policy_snapshot,
)
from .station_tail_diagnostics import _load_feature_rows
from .station_tail_paper_v3 import (
    HARD_SOURCE_REJECTIONS,
    _historical_candidates,
)


MODEL_SCHEMA_VERSION = 1
STRATEGY_NAME = "station_tail_hierarchical_market_offset_v4"
MINIMUM_GLOBAL_CANDIDATES = 60
MINIMUM_STATION_DAYS = 20
MINIMUM_LANE_DAYS = 8
MINIMUM_STATION_WINS = 5
MINIMUM_STATION_LOSSES = 5
RECENCY_HALF_LIFE_DAYS = 120.0
GLOBAL_INTERCEPT_PENALTY = 12.0
GLOBAL_SLOPE_PENALTY = 45.0
STATION_INTERCEPT_SHRINKAGE = 18.0
STATION_SLOPE_SHRINKAGE = 55.0
LANE_INTERCEPT_SHRINKAGE = 20.0
LOWER_CONFIDENCE_Z = 1.2815515655446004
MINIMUM_PEER_COVERAGE = 0.25
ALLOWED_DECISION_MODES = {"event_driven", "forecast_update", "scheduled"}


FEATURE_NAMES = (
    "weather_market_logit_gap",
    "forecast_above_high_c",
    "forecast_below_low_c",
    "forecast_inside_margin_c",
    "upper_headroom_c",
    "distance_to_low_c",
    "peer_slope_1h_c",
    "peer_temperature_delta_c",
    "temperature_slope_1h_c",
    "hours_since_running_max",
    "day_of_year_sin",
    "day_of_year_cos",
    "local_time_sin",
    "local_time_cos",
)
FEATURE_SIGNS = (
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    0,
    0,
    0,
    0,
)
EDGE_BIN_BOUNDS = (-1.0, -0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 1.0)


@dataclass(slots=True)
class OffsetModel:
    coefficients: Any
    covariance: Any
    independent_days: int
    effective_days: float
    wins: int
    losses: int


@dataclass(slots=True)
class LaneOffset:
    offset: float
    variance: float
    independent_days: int


@dataclass(slots=True)
class StationTailV4Bundle:
    schema_version: int
    deployed_at_ns: int
    trained_through_local_date: str
    training_candidates: int
    input_sha256: str
    feature_names: tuple[str, ...]
    feature_means: Any
    feature_scales: Any
    global_model: OffsetModel
    station_models: dict[str, OffsetModel]
    lane_offsets: dict[str, LaneOffset]
    station_day_counts: dict[str, int]
    lane_day_counts: dict[str, int]
    station_outcomes: dict[str, tuple[int, int]]


def evaluate_station_tail_v4(cfg: AppConfig) -> dict[str, Path]:

    candidates = _historical_candidates_v4(cfg)
    input_hash = _historical_input_hash(cfg)
    full_bundle = _fit_bundle(candidates, 0, input_hash)
    predictions, skipped = _rolling_predictions(candidates)
    trades = _replay_lower_bound_strategy(predictions)
    scores = _probability_scores(predictions)
    calibration = _calibration_rows(predictions)
    edge_bins = _edge_bin_rows(predictions)
    station_scores = {
        station_id: _probability_scores(
            [row for row in predictions if row["station_id"] == station_id]
        )
        for station_id in sorted(STATION_ADAPTIVE_LANES)
    }
    station_trades = {
        station_id: _trade_summary(
            [row for row in trades if row["station_id"] == station_id]
        )
        for station_id in sorted(STATION_ADAPTIVE_LANES)
    }
    market_brier = scores["market_candidate_probability"]["brier"]
    v4_brier = scores["v4_candidate_probability"]["brier"]
    monotonic = _edge_monotonicity(edge_bins)
    historical_gate = bool(
        market_brier is not None
        and v4_brier is not None
        and v4_brier <= market_brier
        and monotonic["comparable_adjacent_bins"] > 0
        and monotonic["nondecreasing_fraction"] >= 0.5
    )

    generated_at_ns = time.time_ns()
    run_id = (
        datetime.fromtimestamp(generated_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{generated_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    output_dir = cfg.reports_dir / "station_tail_v4" / "artifacts" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "rolling_predictions.csv"
    skipped_path = output_dir / "rolling_skipped.csv"
    trades_path = output_dir / "lower_bound_trades.csv"
    calibration_path = output_dir / "calibration_bins.csv"
    edge_path = output_dir / "edge_monotonicity.csv"
    summary_path = output_dir / "summary.json"
    _write_csv_atomic(predictions_path, predictions)
    _write_csv_atomic(skipped_path, skipped)
    _write_csv_atomic(trades_path, trades)
    _write_csv_atomic(calibration_path, calibration)
    _write_csv_atomic(edge_path, edge_bins)

    summary = {
        "generated_at_utc": datetime.fromtimestamp(
            generated_at_ns / 1e9, tz=UTC
        ).isoformat(),
        "run_id": run_id,
        "strategy": STRATEGY_NAME,
        "status": (
            "historical_gate_passed"
            if historical_gate
            else "historical_gate_failed_research_only"
        ),
        "alpha_demonstrated": False,
        "alpha_warning": (
            "All available historical outcomes have informed development. "
            "This replay is diagnostic; only post-deployment station-days are prospective."
        ),
        "input_sha256": input_hash,
        "candidate_rows": len(candidates),
        "rolling_predictions": len(predictions),
        "rolling_skipped": len(skipped),
        "training_rule": {
            "chronology": "strictly earlier resolved local dates only",
            "independence": "weights normalized to one per station-day",
            "minimum_global_candidates": MINIMUM_GLOBAL_CANDIDATES,
            "minimum_station_days": MINIMUM_STATION_DAYS,
            "minimum_lane_days": MINIMUM_LANE_DAYS,
            "minimum_station_wins": MINIMUM_STATION_WINS,
            "minimum_station_losses": MINIMUM_STATION_LOSSES,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        },
        "model": {
            "market_offset": (
                "logit(p_v4) = logit(p_market) + shrunken station residual"
            ),
            "partial_pooling": (
                "global residual model, airport coefficients shrunk toward global, "
                "lane intercepts shrunk toward zero"
            ),
            "features": list(FEATURE_NAMES),
            "physical_sign_constraints": dict(
                zip(FEATURE_NAMES, FEATURE_SIGNS, strict=True)
            ),
            "uncertainty": (
                "forecast spread and peer coverage shrink the weather correction; "
                "trades require a one-sided 90% lower probability bound above cost"
            ),
        },
        "peer_feature_audit": _peer_audit(candidates),
        "probability_scores": scores,
        "probability_scores_by_station": station_scores,
        "predeclared_monthly_blocks": _monthly_block_rows(predictions),
        "frozen_full_history_model_audit": _bundle_audit(full_bundle),
        "edge_monotonicity": monotonic,
        "lower_bound_strategy": _trade_summary(trades),
        "lower_bound_strategy_by_station": station_trades,
        "historical_deployment_gate_passed": historical_gate,
        "economic_warning": (
            "Historical prices are probability proxies plus the frozen lane cost; "
            "the report is not a claim of executable fill P&L."
        ),
        "live_orders_enabled": False,
    }
    _write_json_atomic(summary_path, summary)
    manifest = _artifact_manifest(output_dir, input_hash)
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    latest_path = cfg.reports_dir / "station_tail_v4" / "latest.json"
    _write_json_atomic(
        latest_path,
        {
            "run_id": run_id,
            "artifact_directory": str(output_dir.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "summary_sha256": _sha256_file(summary_path),
        },
    )
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "predictions": predictions_path,
        "skipped": skipped_path,
        "trades": trades_path,
        "calibration": calibration_path,
        "edge_monotonicity": edge_path,
        "latest": latest_path,
    }


def deploy_station_tail_v4(cfg: AppConfig) -> dict[str, Path]:

    diagnostics = evaluate_station_tail_v4(cfg)
    diagnostic_summary = json.loads(
        diagnostics["summary"].read_text(encoding="utf-8")
    )
    candidates = _historical_candidates_v4(cfg)
    deployed_at_ns = time.time_ns()
    input_hash = _historical_input_hash(cfg)
    bundle = _fit_bundle(candidates, deployed_at_ns, input_hash)
    deployment_id = (
        datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{deployed_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    root = cfg.reports_dir / "paper_trading_station_tail_v4"
    deployment_dir = root / "deployments" / deployment_id
    deployment_dir.mkdir(parents=True, exist_ok=False)
    model_path = deployment_dir / "model_bundle.pkl"
    model_path.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    model_hash = _sha256_file(model_path)
    deployment_path = deployment_dir / "deployment.json"
    _write_json_atomic(
        deployment_path,
        {
            "deployment_id": deployment_id,
            "strategy": STRATEGY_NAME,
            "schema_version": MODEL_SCHEMA_VERSION,
            "deployed_at_ns": deployed_at_ns,
            "deployed_at_utc": datetime.fromtimestamp(
                deployed_at_ns / 1e9, tz=UTC
            ).isoformat(),
            "trained_through_local_date": bundle.trained_through_local_date,
            "training_candidates": bundle.training_candidates,
            "historical_input_sha256": input_hash,
            "model_sha256": model_hash,
            "source_diagnostics": str(diagnostics["summary"].resolve()),
            "source_diagnostics_sha256": _sha256_file(diagnostics["summary"]),
            "historical_deployment_gate_passed": diagnostic_summary[
                "historical_deployment_gate_passed"
            ],
            "deployment_mode": "prospective_research_paper",
            "trade_rule": (
                "one-sided 90% probability lower bound minus synthetic cost > 0"
            ),
            "maximum_entries_per_station_day": 1,
            "accepted_decision_modes": sorted(ALLOWED_DECISION_MODES),
            "hard_source_rejections": sorted(HARD_SOURCE_REJECTIONS),
            "live_orders_enabled": False,
        },
    )
    state_path = deployment_dir / "state.json"
    _write_json_atomic(
        state_path,
        {
            "deployment_id": deployment_id,
            "deployed_at_ns": deployed_at_ns,
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    _write_csv_atomic(deployment_dir / "paper_audit.csv", [])
    _write_csv_atomic(deployment_dir / "paper_trades.csv", [])
    immutable_manifest_path = deployment_dir / "manifest.json"
    _write_json_atomic(
        immutable_manifest_path,
        _artifact_manifest(
            deployment_dir,
            input_hash,
            exclude={
                "state.json",
                "paper_audit.csv",
                "paper_trades.csv",
                "paper_summary.json",
            },
        ),
    )
    active_path = root / "active_deployment.json"
    _write_json_atomic(
        active_path,
        {
            "deployment_id": deployment_id,
            "deployment_directory": str(deployment_dir.resolve()),
            "deployment_sha256": _sha256_file(deployment_path),
            "model_sha256": model_hash,
            "live_orders_enabled": False,
        },
    )
    return {
        "active": active_path,
        "deployment": deployment_path,
        "model": model_path,
        "state": state_path,
        "diagnostics": diagnostics["summary"],
        "manifest": immutable_manifest_path,
    }


def run_station_tail_v4_paper_cycle(cfg: AppConfig) -> dict[str, Path]:

    root = cfg.reports_dir / "paper_trading_station_tail_v4"
    active_path = root / "active_deployment.json"
    if not active_path.exists():
        raise FileNotFoundError("Missing V4 deployment; run deploy-station-tail-v4")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    deployment_dir = Path(active["deployment_directory"])
    deployment_path = deployment_dir / "deployment.json"
    model_path = deployment_dir / "model_bundle.pkl"
    if _sha256_file(model_path) != active["model_sha256"]:
        raise RuntimeError("Active V4 model hash mismatch")
    if _sha256_file(deployment_path) != active["deployment_sha256"]:
        raise RuntimeError("Active V4 deployment hash mismatch")
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    bundle = pickle.loads(model_path.read_bytes())
    if not isinstance(bundle, StationTailV4Bundle):
        raise TypeError("Invalid V4 model bundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise RuntimeError("V4 model schema is incompatible; redeploy it")

    source_path = (
        cfg.reports_dir
        / "paper_trading_challenger"
        / "paper_signal_audit.csv"
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}; run paper-challenger-cycle")
    state_path = deployment_dir / "state.json"
    audit_path = deployment_dir / "paper_audit.csv"
    trades_path = deployment_dir / "paper_trades.csv"
    summary_path = deployment_dir / "paper_summary.json"
    state = _read_json(
        state_path,
        {
            "deployment_id": deployment["deployment_id"],
            "deployed_at_ns": deployment["deployed_at_ns"],
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    audit_rows = _load_csv(audit_path)
    trade_rows = _load_csv(trades_path)
    processed = set(state.get("processed_decision_ids") or [])
    traded_days = set(state.get("traded_station_days") or [])
    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    lanes_by_station = {
        station_id: list(lanes)
        for station_id, lanes in STATION_ADAPTIVE_LANES.items()
    }
    feature_rows = _load_feature_rows(cfg)
    deployed_at_ns = int(deployment["deployed_at_ns"])
    cycle_started_ns = time.time_ns()

    for source_row in _load_csv(source_path):
        decision_id = str(source_row.get("decision_id") or "")
        timestamp_ns = _timestamp_ns(source_row.get("received_wall_ns"))
        station_id = str(source_row.get("station_id") or "")
        decision_mode = str(source_row.get("decision_mode") or "")
        if (
            not decision_id
            or decision_id in processed
            or timestamp_ns is None
            or timestamp_ns <= deployed_at_ns
            or station_id not in stations
            or decision_mode not in ALLOWED_DECISION_MODES
        ):
            continue
        processed.add(decision_id)
        station = stations[station_id]
        local_dt = datetime.fromtimestamp(
            timestamp_ns / 1e9, tz=UTC
        ).astimezone(ZoneInfo(station.timezone))
        local_date = local_dt.date().isoformat()
        day_key = f"{station_id}|{local_date}"
        weather = _probabilities(source_row.get("weather_distribution_json"))
        market = _probabilities(source_row.get("market_distribution_json"))
        feature_key = (
            station_id,
            local_date,
            decision_mode,
            str(timestamp_ns),
        )
        feature_row = feature_rows.get(feature_key, {})
        evaluations: list[dict[str, Any]] = []
        for lane in lanes_by_station.get(station_id, []):
            if local_dt.hour != int(lane["target_local_hour"]):
                continue
            evaluation = evaluate_station_policy_snapshot(lane, weather, market)
            reasons = list(evaluation["rejection_reasons"])
            if local_dt.minute > int(lane["latest_local_minute"]):
                reasons.append("after_station_entry_window")
            source_rejections = {
                item
                for item in str(source_row.get("rejection_reasons") or "").split("|")
                if item
            }
            reasons.extend(sorted(source_rejections & HARD_SOURCE_REJECTIONS))
            candidate = None
            prediction = None
            if evaluation.get("candidate_bucket"):
                candidate = _live_candidate(
                    source_row,
                    feature_row,
                    station,
                    lane,
                    evaluation,
                    weather,
                    market,
                    timestamp_ns,
                    local_date,
                )
                if candidate is not None:
                    prediction = _predict_candidate(bundle, candidate)
            if candidate is None or prediction is None:
                reasons.append("v4_candidate_features_unavailable")
            else:
                reasons.extend(prediction["warmup_rejections"])
                if float(candidate.get("peer_station_coverage") or 0.0) < (
                    MINIMUM_PEER_COVERAGE
                ):
                    reasons.append("insufficient_v4_peer_coverage")
                if float(prediction["lower_bound_edge"]) <= 0.0:
                    reasons.append("v4_lower_bound_edge_not_positive")
                if day_key in traded_days:
                    reasons.append("station_day_already_traded")
            evaluations.append(
                {
                    "lane": lane,
                    "evaluation": evaluation,
                    "candidate": candidate,
                    "prediction": prediction,
                    "reasons": sorted(set(reasons)),
                }
            )

        eligible = [item for item in evaluations if not item["reasons"]]
        chosen = (
            max(
                eligible,
                key=lambda item: float(
                    item["prediction"]["lower_bound_edge"]
                ),
            )
            if eligible
            else None
        )
        lag_seconds = max(0.0, (cycle_started_ns - timestamp_ns) / 1e9)
        if not evaluations:
            audit_rows.append(
                {
                    "decision_id": decision_id,
                    "station_id": station_id,
                    "local_date": local_date,
                    "decision_mode": decision_mode,
                    "received_wall_ns": timestamp_ns,
                    "processed_at_ns": cycle_started_ns,
                    "processing_lag_seconds": lag_seconds,
                    "paper_trade_opened": False,
                    "rejection_reasons": "outside_v4_station_entry_hour",
                    "live_orders_enabled": False,
                }
            )
            continue
        for item in evaluations:
            candidate = item["candidate"] or {}
            prediction = item["prediction"] or {}
            opened = item is chosen
            reasons = list(item["reasons"])
            if not opened and not reasons and chosen is not None:
                reasons.append("lower_confidence_edge_than_selected_lane")
            audit_rows.append(
                {
                    "decision_id": decision_id,
                    "station_id": station_id,
                    "local_date": local_date,
                    "decision_mode": decision_mode,
                    "trigger_source": source_row.get("trigger_source"),
                    "received_utc": source_row.get("received_utc"),
                    "received_wall_ns": timestamp_ns,
                    "processed_at_ns": cycle_started_ns,
                    "processing_lag_seconds": lag_seconds,
                    "policy_id": item["lane"]["policy_id"],
                    "candidate_bucket": item["evaluation"].get("candidate_bucket"),
                    "candidate_price_proxy": candidate.get("candidate_price_proxy"),
                    "synthetic_cost_per_share": candidate.get(
                        "synthetic_cost_per_share"
                    ),
                    "raw_snapshot_probability": candidate.get(
                        "raw_snapshot_probability"
                    ),
                    "v4_candidate_probability": prediction.get(
                        "candidate_probability"
                    ),
                    "v4_probability_lower_bound": prediction.get(
                        "probability_lower_bound"
                    ),
                    "v4_estimated_edge": prediction.get("estimated_edge"),
                    "v4_lower_bound_edge": prediction.get("lower_bound_edge"),
                    "weather_correction_confidence": prediction.get(
                        "weather_correction_confidence"
                    ),
                    "parameter_logit_standard_error": prediction.get(
                        "parameter_logit_standard_error"
                    ),
                    "peer_station_coverage": candidate.get(
                        "peer_station_coverage"
                    ),
                    "paper_trade_opened": opened,
                    "rejection_reasons": "|".join(sorted(set(reasons))),
                    "post_entry_monitor": day_key in traded_days,
                    "live_orders_enabled": False,
                }
            )
        if chosen is None:
            continue
        candidate = chosen["candidate"]
        prediction = chosen["prediction"]
        trade_rows.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "policy_id": chosen["lane"]["policy_id"],
                "decision_mode": decision_mode,
                "trigger_source": source_row.get("trigger_source"),
                "entry_received_utc": source_row.get("received_utc"),
                "entry_received_wall_ns": timestamp_ns,
                "processed_at_ns": cycle_started_ns,
                "processing_lag_seconds": lag_seconds,
                "bucket": chosen["evaluation"]["candidate_bucket"],
                "price_proxy": candidate["candidate_price_proxy"],
                "fixed_cost_per_share": candidate["fixed_cost_per_share"],
                "synthetic_cost_per_share": candidate["synthetic_cost_per_share"],
                "v4_candidate_probability": prediction["candidate_probability"],
                "v4_probability_lower_bound": prediction[
                    "probability_lower_bound"
                ],
                "estimated_synthetic_edge": prediction["estimated_edge"],
                "lower_bound_synthetic_edge": prediction["lower_bound_edge"],
                "winning_bucket": "",
                "won": "",
                "synthetic_value_per_share": "",
                "synthetic_value_10_shares": "",
                "settled": False,
                "live_orders_enabled": False,
            }
        )
        traded_days.add(day_key)

    _settle_trades(cfg, trade_rows)
    _write_csv_atomic(audit_path, audit_rows)
    _write_csv_atomic(trades_path, trade_rows)
    state.update(
        {
            "processed_decision_ids": sorted(processed)[-50_000:],
            "traded_station_days": sorted(traded_days),
            "last_cycle_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write_json_atomic(state_path, state)
    settled = [row for row in trade_rows if _truth(row.get("settled"))]
    _write_json_atomic(
        summary_path,
        {
            "strategy": STRATEGY_NAME,
            "deployment_id": deployment["deployment_id"],
            "deployed_at_ns": deployed_at_ns,
            "trained_through_local_date": bundle.trained_through_local_date,
            "historical_deployment_gate_passed": deployment[
                "historical_deployment_gate_passed"
            ],
            "alpha_demonstrated": False,
            "processed_decisions": len(processed),
            "audit_rows": len(audit_rows),
            "paper_trades": len(trade_rows),
            "settled_trades": len(settled),
            "wins": sum(_truth(row.get("won")) for row in settled),
            "synthetic_value_per_share_total": sum(
                float(row["synthetic_value_per_share"]) for row in settled
            ),
            "mean_processing_lag_seconds": _mean(
                [
                    _safe_float(row.get("processing_lag_seconds"))
                    for row in audit_rows
                ]
            ),
            "trade_rule": (
                "one-sided 90% probability lower bound exceeds frozen synthetic cost"
            ),
            "live_orders_enabled": False,
        },
    )
    return {
        "active": active_path,
        "audit": audit_path,
        "trades": trades_path,
        "summary": summary_path,
        "state": state_path,
    }


def _historical_candidates_v4(cfg: AppConfig) -> list[dict[str, Any]]:
    candidates = _historical_candidates(cfg)
    usable: list[dict[str, Any]] = []
    for row in candidates:
        if not _truth(row.get("peer_history_point_in_time")):
            continue
        coverage = _safe_float(row.get("peer_station_coverage"))
        if coverage is None or coverage < MINIMUM_PEER_COVERAGE:
            continue
        enriched = dict(row)
        enriched["weather_distribution"] = {}
        enriched["market_distribution"] = {}
        usable.append(enriched)
    if len(usable) < MINIMUM_GLOBAL_CANDIDATES:
        raise RuntimeError(
            "Insufficient point-in-time peer coverage for V4; run "
            "augment-historical-peers after the latest backfill"
        )
    return usable


def _fit_bundle(
    rows: list[dict[str, Any]],
    deployed_at_ns: int,
    input_hash: str,
) -> StationTailV4Bundle:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc
    if len(rows) < MINIMUM_GLOBAL_CANDIDATES:
        raise RuntimeError("Insufficient historical candidates for V4")
    records = [_feature_record(row) for row in rows]
    raw = np.asarray(records, dtype=float)
    means = np.nanmedian(raw, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(raw), raw, means)
    scales = np.nanstd(filled, axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    global_model = _fit_offset_model(
        rows,
        means,
        scales,
        prior=np.zeros(len(FEATURE_NAMES) + 1),
        intercept_penalty=GLOBAL_INTERCEPT_PENALTY,
        slope_penalty=GLOBAL_SLOPE_PENALTY,
    )
    station_models: dict[str, OffsetModel] = {}
    station_day_counts: dict[str, int] = {}
    station_outcomes: dict[str, tuple[int, int]] = {}
    for station_id in sorted({str(row["station_id"]) for row in rows}):
        station_rows = [
            row for row in rows if str(row["station_id"]) == station_id
        ]
        station_models[station_id] = _fit_offset_model(
            station_rows,
            means,
            scales,
            prior=global_model.coefficients,
            intercept_penalty=STATION_INTERCEPT_SHRINKAGE,
            slope_penalty=STATION_SLOPE_SHRINKAGE,
        )
        station_day_counts[station_id] = len(
            {str(row["local_date"]) for row in station_rows}
        )
        wins = sum(_truth(row["won"]) for row in station_rows)
        station_outcomes[station_id] = (wins, len(station_rows) - wins)
    lane_offsets: dict[str, LaneOffset] = {}
    lane_day_counts: dict[str, int] = {}
    for policy_id in sorted({str(row["policy_id"]) for row in rows}):
        lane_rows = [row for row in rows if str(row["policy_id"]) == policy_id]
        station_id = str(lane_rows[0]["station_id"])
        lane_offsets[policy_id] = _fit_lane_offset(
            lane_rows,
            station_models[station_id],
            means,
            scales,
        )
        lane_day_counts[policy_id] = len(
            {str(row["local_date"]) for row in lane_rows}
        )
    return StationTailV4Bundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        trained_through_local_date=max(str(row["local_date"]) for row in rows),
        training_candidates=len(rows),
        input_sha256=input_hash,
        feature_names=FEATURE_NAMES,
        feature_means=means,
        feature_scales=scales,
        global_model=global_model,
        station_models=station_models,
        lane_offsets=lane_offsets,
        station_day_counts=station_day_counts,
        lane_day_counts=lane_day_counts,
        station_outcomes=station_outcomes,
    )


def _fit_offset_model(
    rows: list[dict[str, Any]],
    means: Any,
    scales: Any,
    *,
    prior: Any,
    intercept_penalty: float,
    slope_penalty: float,
) -> OffsetModel:
    import numpy as np

    x, offset, target, weights = _design(rows, means, scales)
    penalties = np.asarray(
        [intercept_penalty] + [slope_penalty] * len(FEATURE_NAMES),
        dtype=float,
    )
    beta = np.asarray(prior, dtype=float).copy()
    identity = np.diag(penalties)
    for _ in range(80):
        eta = np.clip(offset + x @ beta, -25.0, 25.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        gradient = x.T @ (weights * (probability - target))
        gradient += penalties * (beta - prior)
        curvature = weights * probability * (1.0 - probability)
        hessian = x.T @ (x * curvature[:, None]) + identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        candidate = _project_coefficients(beta - step)
        if float(np.max(np.abs(candidate - beta))) < 1e-8:
            beta = candidate
            break
        beta = candidate
    eta = np.clip(offset + x @ beta, -25.0, 25.0)
    probability = 1.0 / (1.0 + np.exp(-eta))
    curvature = weights * probability * (1.0 - probability)
    hessian = x.T @ (x * curvature[:, None]) + identity
    covariance = np.linalg.pinv(hessian)
    day_weights = _date_weights(rows)
    effective_days = _effective_sample_size(list(day_weights.values()))
    wins = sum(_truth(row["won"]) for row in rows)
    return OffsetModel(
        coefficients=beta,
        covariance=covariance,
        independent_days=len(day_weights),
        effective_days=effective_days,
        wins=wins,
        losses=len(rows) - wins,
    )


def _fit_lane_offset(
    rows: list[dict[str, Any]],
    station_model: OffsetModel,
    means: Any,
    scales: Any,
) -> LaneOffset:
    import numpy as np

    x, market_offset, target, weights = _design(rows, means, scales)
    base = market_offset + x @ station_model.coefficients
    confidence = np.asarray(
        [_weather_confidence(row) for row in rows], dtype=float
    )
    delta = 0.0
    for _ in range(50):
        eta = np.clip(base + confidence * delta, -25.0, 25.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        gradient = float(
            np.sum(weights * confidence * (probability - target))
            + LANE_INTERCEPT_SHRINKAGE * delta
        )
        hessian = float(
            np.sum(
                weights
                * confidence**2
                * probability
                * (1.0 - probability)
            )
            + LANE_INTERCEPT_SHRINKAGE
        )
        update = gradient / max(hessian, 1e-9)
        delta -= update
        if abs(update) < 1e-8:
            break
    variance = 1.0 / max(hessian, 1e-9)
    return LaneOffset(
        offset=delta,
        variance=variance,
        independent_days=len({str(row["local_date"]) for row in rows}),
    )


def _design(
    rows: list[dict[str, Any]], means: Any, scales: Any
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    raw = np.asarray([_feature_record(row) for row in rows], dtype=float)
    filled = np.where(np.isfinite(raw), raw, means)
    standardized = (filled - means) / scales
    confidence = np.asarray(
        [_weather_confidence(row) for row in rows], dtype=float
    )
    x = np.column_stack([np.ones(len(rows)), standardized])
    x *= confidence[:, None]
    offset = np.asarray(
        [_logit(row["candidate_price_proxy"]) for row in rows], dtype=float
    )
    target = np.asarray([1.0 if _truth(row["won"]) else 0.0 for row in rows])
    row_weights = _row_weights(rows)
    weights = np.asarray(row_weights, dtype=float)
    return x, offset, target, weights


def _project_coefficients(coefficients: Any) -> Any:
    projected = coefficients.copy()
    for index, sign in enumerate(FEATURE_SIGNS, start=1):
        if sign > 0:
            projected[index] = max(0.0, projected[index])
        elif sign < 0:
            projected[index] = min(0.0, projected[index])
    return projected


def _predict_candidate(
    bundle: StationTailV4Bundle, row: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    station_id = str(row["station_id"])
    policy_id = str(row["policy_id"])
    station_model = bundle.station_models.get(
        station_id, bundle.global_model
    )
    lane = bundle.lane_offsets.get(
        policy_id, LaneOffset(0.0, 1.0 / LANE_INTERCEPT_SHRINKAGE, 0)
    )
    raw = np.asarray(_feature_record(row), dtype=float)
    filled = np.where(np.isfinite(raw), raw, bundle.feature_means)
    standardized = (filled - bundle.feature_means) / bundle.feature_scales
    confidence = _weather_confidence(row)
    x = np.concatenate([[1.0], standardized]) * confidence
    market_logit = _logit(row["candidate_price_proxy"])
    eta = float(
        market_logit
        + x @ station_model.coefficients
        + confidence * lane.offset
    )
    parameter_variance = float(
        x @ station_model.covariance @ x
        + confidence**2 * lane.variance
        + 1.0 / max(10.0, station_model.effective_days)
    )
    standard_error = math.sqrt(max(parameter_variance, 0.0))
    probability = _sigmoid(eta)
    lower_probability = _sigmoid(
        eta - LOWER_CONFIDENCE_Z * standard_error
    )
    cost = float(row["synthetic_cost_per_share"])
    warmup_rejections = _warmup_rejections(bundle, station_id, policy_id)
    return {
        "candidate_probability": probability,
        "probability_lower_bound": lower_probability,
        "estimated_edge": probability - cost,
        "lower_bound_edge": lower_probability - cost,
        "market_probability": float(row["candidate_price_proxy"]),
        "weather_correction_confidence": confidence,
        "parameter_logit_standard_error": standard_error,
        "station_training_days": bundle.station_day_counts.get(station_id, 0),
        "lane_training_days": bundle.lane_day_counts.get(policy_id, 0),
        "warmup_rejections": warmup_rejections,
    }


def _warmup_rejections(
    bundle: StationTailV4Bundle, station_id: str, policy_id: str
) -> list[str]:
    reasons: list[str] = []
    station_days = bundle.station_day_counts.get(station_id, 0)
    lane_days = bundle.lane_day_counts.get(policy_id, 0)
    wins, losses = bundle.station_outcomes.get(station_id, (0, 0))
    if station_days < MINIMUM_STATION_DAYS:
        reasons.append("insufficient_v4_station_days")
    if lane_days < MINIMUM_LANE_DAYS:
        reasons.append("insufficient_v4_lane_days")
    if wins < MINIMUM_STATION_WINS:
        reasons.append("insufficient_v4_station_wins")
    if losses < MINIMUM_STATION_LOSSES:
        reasons.append("insufficient_v4_station_losses")
    return reasons


def _rolling_predictions(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    dates = sorted({str(row["local_date"]) for row in candidates})
    input_hash = "rolling-diagnostic"
    for test_date in dates:
        training = [
            row for row in candidates if str(row["local_date"]) < test_date
        ]
        testing = [
            row for row in candidates if str(row["local_date"]) == test_date
        ]
        if len(training) < MINIMUM_GLOBAL_CANDIDATES:
            skipped.extend(
                {**row, "v4_skip_reason": "insufficient_global_candidates"}
                for row in testing
            )
            continue
        bundle = _fit_bundle(training, 0, input_hash)
        for row in testing:
            prediction = _predict_candidate(bundle, row)
            if prediction["warmup_rejections"]:
                skipped.append(
                    {
                        **row,
                        "v4_skip_reason": "|".join(
                            prediction["warmup_rejections"]
                        ),
                    }
                )
                continue
            value = (
                (1.0 if _truth(row["won"]) else 0.0)
                - float(row["synthetic_cost_per_share"])
            )
            predictions.append(
                {
                    **row,
                    "rolling_training_candidates": len(training),
                    "rolling_station_training_days": prediction[
                        "station_training_days"
                    ],
                    "rolling_lane_training_days": prediction[
                        "lane_training_days"
                    ],
                    "v4_candidate_probability": prediction[
                        "candidate_probability"
                    ],
                    "v4_probability_lower_bound": prediction[
                        "probability_lower_bound"
                    ],
                    "v4_estimated_edge": prediction["estimated_edge"],
                    "v4_lower_bound_edge": prediction["lower_bound_edge"],
                    "weather_correction_confidence": prediction[
                        "weather_correction_confidence"
                    ],
                    "parameter_logit_standard_error": prediction[
                        "parameter_logit_standard_error"
                    ],
                    "synthetic_value_per_share": value,
                }
            )
    return predictions, skipped


def _replay_lower_bound_strategy(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_day[(str(row["station_id"]), str(row["local_date"]))].append(row)
    trades: list[dict[str, Any]] = []
    for rows in by_day.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["received_wall_ns"]),
                -float(row["v4_lower_bound_edge"]),
            ),
        )
        first_timestamp = next(
            (
                int(row["received_wall_ns"])
                for row in ordered
                if float(row["v4_lower_bound_edge"]) > 0.0
            ),
            None,
        )
        if first_timestamp is None:
            continue
        eligible = [
            row
            for row in ordered
            if int(row["received_wall_ns"]) == first_timestamp
            and float(row["v4_lower_bound_edge"]) > 0.0
        ]
        trades.append(
            dict(max(eligible, key=lambda row: row["v4_lower_bound_edge"]))
        )
    return sorted(
        trades,
        key=lambda row: (str(row["local_date"]), str(row["station_id"])),
    )


def _feature_record(row: dict[str, Any]) -> list[float]:
    unit = _candidate_unit(str(row.get("candidate_bucket") or ""))
    try:
        bucket = parse_temperature_bucket(
            str(row["candidate_bucket"]), unit
        )
    except ValueError:
        bucket = None
    forecast = _safe_float(row.get("forecast_consensus_prediction"))
    current_max = _safe_float(row.get("current_running_max"))
    if unit == "F":
        forecast = _f_to_c(forecast)
        current_max = _f_to_c(current_max)
    low_c = _f_to_c(bucket.low) if bucket and unit == "F" else (
        bucket.low if bucket else None
    )
    high_c = _f_to_c(bucket.high) if bucket and unit == "F" else (
        bucket.high if bucket else None
    )
    forecast_above = (
        max(0.0, forecast - high_c)
        if forecast is not None and high_c is not None
        else 0.0
    )
    forecast_below = (
        max(0.0, low_c - forecast)
        if forecast is not None and low_c is not None
        else 0.0
    )
    inside_margin = (
        max(0.0, min(forecast - low_c, high_c - forecast))
        if forecast is not None and low_c is not None and high_c is not None
        else 0.0
    )
    upper_headroom = (
        max(0.0, high_c - current_max)
        if current_max is not None and high_c is not None
        else 0.0
    )
    distance_to_low = (
        max(0.0, low_c - current_max)
        if current_max is not None and low_c is not None
        else 0.0
    )
    peer_mean = _safe_float(row.get("peer_temperature_mean_c"))
    temperature_c = _safe_float(row.get("temperature_c"))
    peer_delta = (
        peer_mean - temperature_c
        if peer_mean is not None and temperature_c is not None
        else math.nan
    )
    market_logit = _logit(row.get("candidate_price_proxy"))
    weather_logit = _logit(row.get("raw_snapshot_probability"))
    seasonal = _seasonal_values(row)
    return [
        weather_logit - market_logit,
        forecast_above,
        forecast_below,
        inside_margin,
        upper_headroom,
        distance_to_low,
        _nan(row.get("peer_temperature_change_1h_mean")),
        peer_delta,
        _nan(row.get("temperature_change_1h")),
        _nan(row.get("hours_since_running_max")),
        seasonal["day_of_year_sin"],
        seasonal["day_of_year_cos"],
        seasonal["local_time_sin"],
        seasonal["local_time_cos"],
    ]


def _seasonal_values(row: dict[str, Any]) -> dict[str, float]:
    local_date = date.fromisoformat(str(row["local_date"]))
    day_angle = 2.0 * math.pi * (local_date.timetuple().tm_yday - 1) / 365.25
    local_hour = float(row.get("local_hour") or 0.0)
    hour_angle = 2.0 * math.pi * local_hour / 24.0
    return {
        "day_of_year_sin": _safe_float(row.get("day_of_year_sin"))
        or math.sin(day_angle),
        "day_of_year_cos": _safe_float(row.get("day_of_year_cos"))
        or math.cos(day_angle),
        "local_time_sin": _safe_float(row.get("local_time_sin"))
        or math.sin(hour_angle),
        "local_time_cos": _safe_float(row.get("local_time_cos"))
        or math.cos(hour_angle),
    }


def _weather_confidence(row: dict[str, Any]) -> float:
    spread = _safe_float(row.get("forecast_spread_remaining_max_c"))
    if spread is None:
        spread_factor = 0.50
    else:
        spread_factor = 1.0 / math.sqrt(1.0 + (max(spread, 0.0) / 3.0) ** 2)
    coverage = _safe_float(row.get("peer_station_coverage"))
    coverage_factor = 0.50 if coverage is None else 0.75 + 0.25 * min(
        1.0, max(0.0, coverage)
    )
    return max(0.15, min(1.0, spread_factor * coverage_factor))


def _row_weights(rows: list[dict[str, Any]]) -> list[float]:
    day_weights = _date_weights(rows)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["station_id"]), str(row["local_date"]))] += 1
    return [
        day_weights[(str(row["station_id"]), str(row["local_date"]))]
        / counts[(str(row["station_id"]), str(row["local_date"]))]
        for row in rows
    ]


def _date_weights(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    latest = max(date.fromisoformat(str(row["local_date"])) for row in rows)
    result: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["station_id"]), str(row["local_date"]))
        age = (latest - date.fromisoformat(key[1])).days
        result[key] = 0.5 ** (max(0, age) / RECENCY_HALF_LIFE_DAYS)
    return result


def _effective_sample_size(weights: list[float]) -> float:
    total = sum(weights)
    squared = sum(value * value for value in weights)
    return total * total / squared if squared else 0.0


def _live_candidate(
    source_row: dict[str, Any],
    feature_row: dict[str, Any],
    station: StationConfig,
    lane: dict[str, Any],
    evaluation: dict[str, Any],
    weather: dict[str, float],
    market: dict[str, float],
    timestamp_ns: int,
    local_date: str,
) -> dict[str, Any] | None:
    candidate_bucket = str(evaluation.get("candidate_bucket") or "")
    try:
        bucket = parse_temperature_bucket(candidate_bucket, station.market_unit)
    except ValueError:
        return None
    current_max = _safe_float(
        source_row.get("current_resolution_running_max")
    )
    if current_max is None:
        current_max = _safe_float(
            feature_row.get("resolution_running_max_market_unit")
        )
    forecast_c = _safe_float(feature_row.get("model_consensus_today_max_c"))
    forecast_market = (
        c_to_f(forecast_c)
        if forecast_c is not None and station.market_unit == "F"
        else forecast_c
    )
    if forecast_market is None:
        forecast_market = _safe_float(
            source_row.get("calibration_forecast_point")
        )
    spread = _safe_float(source_row.get("forecast_spread_c"))
    if spread is None:
        spread = _safe_float(
            feature_row.get("model_spread_remaining_max_c")
        )
    peer_change = _first_float(
        feature_row,
        "peer_mean_slope_60m_c_per_hour",
        "peer_temperature_change_1h_mean",
    )
    peer_count = _first_float(
        feature_row, "peer_station_count", "peer_available_count"
    )
    expected_peers = len(station.peer_station_ids)
    peer_coverage = _safe_float(feature_row.get("peer_station_coverage"))
    if peer_coverage is None and peer_count is not None and expected_peers:
        peer_coverage = peer_count / expected_peers
    local_dt = datetime.fromtimestamp(
        timestamp_ns / 1e9, tz=UTC
    ).astimezone(ZoneInfo(station.timezone))
    price = float(evaluation["candidate_price_proxy"])
    fixed_cost = float(lane["fixed_cost_per_share"])
    return {
        "station_id": station.station_id,
        "local_date": local_date,
        "local_hour": local_dt.hour + local_dt.minute / 60.0,
        "policy_id": str(lane["policy_id"]),
        "candidate_bucket": candidate_bucket,
        "candidate_price_proxy": price,
        "fixed_cost_per_share": fixed_cost,
        "synthetic_cost_per_share": price + fixed_cost,
        "raw_snapshot_probability": float(
            evaluation["candidate_weather_probability"]
        ),
        "forecast_consensus_prediction": forecast_market,
        "current_running_max": current_max,
        "forecast_spread_remaining_max_c": spread,
        "hours_since_running_max": _safe_float(
            feature_row.get("hours_since_running_max")
        ),
        "temperature_change_1h": _safe_float(
            feature_row.get("temperature_change_1h")
        ),
        "temperature_c": _safe_float(feature_row.get("temperature_c")),
        "peer_temperature_change_1h_mean": peer_change,
        "peer_temperature_mean_c": _safe_float(
            feature_row.get("peer_temperature_mean_c")
        ),
        "peer_station_coverage": peer_coverage,
        "peer_history_point_in_time": True,
        "day_of_year_sin": _safe_float(feature_row.get("day_of_year_sin")),
        "day_of_year_cos": _safe_float(feature_row.get("day_of_year_cos")),
        "local_time_sin": _safe_float(feature_row.get("local_time_sin")),
        "local_time_cos": _safe_float(feature_row.get("local_time_cos")),
        "weather_distribution": weather,
        "market_distribution": market,
        "candidate_low": bucket.low,
        "candidate_high": bucket.high,
    }


def _probability_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [1.0 if _truth(row["won"]) else 0.0 for row in rows]

    def score(field: str) -> dict[str, float | int | None]:
        probabilities = [
            min(1.0 - 1e-9, max(1e-9, float(row[field]))) for row in rows
        ]
        return {
            "rows": len(probabilities),
            "brier": (
                sum(
                    (probability - target) ** 2
                    for probability, target in zip(
                        probabilities, targets, strict=True
                    )
                )
                / len(probabilities)
                if probabilities
                else None
            ),
            "log_loss": (
                -sum(
                    target * math.log(probability)
                    + (1.0 - target) * math.log(1.0 - probability)
                    for probability, target in zip(
                        probabilities, targets, strict=True
                    )
                )
                / len(probabilities)
                if probabilities
                else None
            ),
        }

    return {
        "market_candidate_probability": score("candidate_price_proxy"),
        "raw_weather_candidate_probability": score(
            "raw_snapshot_probability"
        ),
        "v4_candidate_probability": score("v4_candidate_probability"),
    }


def _calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lower in [value / 10.0 for value in range(10)]:
        upper = lower + 0.1
        selected = [
            row
            for row in rows
            if lower <= float(row["v4_candidate_probability"]) < upper
            or (
                upper >= 1.0
                and float(row["v4_candidate_probability"]) == 1.0
            )
        ]
        if not selected:
            continue
        result.append(
            {
                "probability_lower_inclusive": lower,
                "probability_upper_exclusive": upper,
                "rows": len(selected),
                "independent_station_days": len(
                    {
                        (row["station_id"], row["local_date"])
                        for row in selected
                    }
                ),
                "mean_prediction": _mean(
                    [float(row["v4_candidate_probability"]) for row in selected]
                ),
                "observed_frequency": _mean(
                    [1.0 if _truth(row["won"]) else 0.0 for row in selected]
                ),
            }
        )
    return result


def _edge_bin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lower, upper in zip(
        EDGE_BIN_BOUNDS[:-1], EDGE_BIN_BOUNDS[1:], strict=True
    ):
        selected = [
            row
            for row in rows
            if lower <= float(row["v4_estimated_edge"]) < upper
        ]
        if not selected:
            continue
        result.append(
            {
                "edge_lower_inclusive": lower,
                "edge_upper_exclusive": upper,
                "rows": len(selected),
                "independent_station_days": len(
                    {
                        (row["station_id"], row["local_date"])
                        for row in selected
                    }
                ),
                "mean_estimated_edge": _mean(
                    [float(row["v4_estimated_edge"]) for row in selected]
                ),
                "mean_realized_value": _mean(
                    [float(row["synthetic_value_per_share"]) for row in selected]
                ),
                "hit_rate": _mean(
                    [1.0 if _truth(row["won"]) else 0.0 for row in selected]
                ),
            }
        )
    return result


def _edge_monotonicity(edge_bins: list[dict[str, Any]]) -> dict[str, Any]:
    populated = [row for row in edge_bins if int(row["rows"]) >= 3]
    comparisons = 0
    nondecreasing = 0
    for left, right in zip(populated[:-1], populated[1:]):
        comparisons += 1
        if float(right["mean_realized_value"]) >= float(
            left["mean_realized_value"]
        ):
            nondecreasing += 1
    return {
        "minimum_rows_per_compared_bin": 3,
        "comparable_adjacent_bins": comparisons,
        "nondecreasing_adjacent_bins": nondecreasing,
        "nondecreasing_fraction": (
            nondecreasing / comparisons if comparisons else 0.0
        ),
    }


def _monthly_block_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    months = sorted({str(row["local_date"])[:7] for row in rows})
    for month in months:
        selected = [
            row for row in rows if str(row["local_date"]).startswith(month)
        ]
        scores = _probability_scores(selected)
        trades = _replay_lower_bound_strategy(selected)
        blocks.append(
            {
                "month": month,
                "rows": len(selected),
                "independent_station_days": len(
                    {
                        (row["station_id"], row["local_date"])
                        for row in selected
                    }
                ),
                "market_brier": scores["market_candidate_probability"][
                    "brier"
                ],
                "v4_brier": scores["v4_candidate_probability"]["brier"],
                "paper_trades": len(trades),
                "paper_value": sum(
                    float(row["synthetic_value_per_share"]) for row in trades
                ),
            }
        )
    return blocks


def _bundle_audit(bundle: StationTailV4Bundle) -> dict[str, Any]:
    coefficient_names = ("intercept", *bundle.feature_names)
    return {
        "global": {
            "coefficients": {
                name: float(value)
                for name, value in zip(
                    coefficient_names,
                    bundle.global_model.coefficients,
                    strict=True,
                )
            },
            "effective_days": bundle.global_model.effective_days,
        },
        "airports": {
            station_id: {
                "coefficients": {
                    name: float(value)
                    for name, value in zip(
                        coefficient_names,
                        model.coefficients,
                        strict=True,
                    )
                },
                "independent_days": model.independent_days,
                "effective_days": model.effective_days,
                "wins": model.wins,
                "losses": model.losses,
            }
            for station_id, model in bundle.station_models.items()
        },
        "lanes": {
            policy_id: {
                "logit_offset": lane.offset,
                "variance": lane.variance,
                "independent_days": lane.independent_days,
            }
            for policy_id, lane in bundle.lane_offsets.items()
        },
    }


def _trade_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stake = sum(float(row["synthetic_cost_per_share"]) for row in rows)
    value = sum(float(row["synthetic_value_per_share"]) for row in rows)
    return {
        "trades": len(rows),
        "wins": sum(_truth(row.get("won")) for row in rows),
        "hit_rate": (
            sum(_truth(row.get("won")) for row in rows) / len(rows)
            if rows
            else None
        ),
        "synthetic_stake": stake,
        "synthetic_value_total": value,
        "synthetic_return_on_stake": value / stake if stake else None,
    }


def _peer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        station_id: {
            "candidate_rows": len(station_rows),
            "point_in_time_rows": sum(
                _truth(row.get("peer_history_point_in_time"))
                for row in station_rows
            ),
            "mean_coverage": _mean(
                [
                    _safe_float(row.get("peer_station_coverage"))
                    for row in station_rows
                ]
            ),
            "rows_with_peer_trend": sum(
                _safe_float(row.get("peer_temperature_change_1h_mean"))
                is not None
                for row in station_rows
            ),
        }
        for station_id in sorted({str(row["station_id"]) for row in rows})
        for station_rows in [
            [row for row in rows if str(row["station_id"]) == station_id]
        ]
    }


def _historical_input_hash(cfg: AppConfig) -> str:
    digest = hashlib.sha256()
    for path in (
        cfg.reports_dir / "historical" / "predictive_features.csv",
        cfg.reports_dir / "historical" / "resolved_events.json",
        cfg.project_root / "config" / "app.yaml",
        cfg.project_root / "config" / "stations.yaml",
    ):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _artifact_manifest(
    directory: Path,
    input_hash: str,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude or set()
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "immutable": True,
        "historical_input_sha256": input_hash,
        "files": {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path.name != "manifest.json"
            and path.name not in excluded
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_unit(label: str) -> str:
    return "F" if "F" in label.upper() else "C"


def _f_to_c(value: float | None) -> float | None:
    return (value - 32.0) * 5.0 / 9.0 if value is not None else None


def _logit(value: Any) -> float:
    probability = _safe_float(value)
    if probability is None:
        probability = 0.5
    probability = min(0.995, max(0.005, probability))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    value = min(30.0, max(-30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _nan(value: Any) -> float:
    parsed = _safe_float(value)
    return parsed if parsed is not None else math.nan


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None
