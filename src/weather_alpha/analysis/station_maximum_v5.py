from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from .model import (
    _bootstrap_mean_ci,
    _brier,
    _label_for_value,
    _normalize,
)
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import (
    _load_validation_rows,
    _probabilities,
    _timestamp_ns,
)
from .station_adaptive_alpha import _recent_dates_by_station
from .station_specific_alpha import _load_csv, _read_json, _truth
from .station_tail_diagnostics import _build_candidates, _load_feature_rows
from .station_tail_v4 import (
    MINIMUM_PEER_COVERAGE,
    _artifact_manifest,
    _edge_bin_rows,
    _edge_monotonicity,
    _probability_scores as _v4_probability_scores,
    _replay_lower_bound_strategy,
    _rolling_predictions,
    _sha256_file,
    _trade_summary,
)


STRATEGY_NAME = "station_maximum_multiyear_physical_stack_v5"
NCEI_HOLDOUT_START_YEAR = 2023
MINIMUM_RECENT_CALIBRATION_DAYS = 20
MINIMUM_HOUR_CALIBRATION_DAYS = 12
FORECAST_WEIGHT_RIDGE = 10.0
RECENT_RESIDUAL_WEIGHT = 4
SEASONAL_RESIDUAL_WINDOW_DAYS = 60
NCEI_MIN_SAMPLES_LEAF = 8
NCEI_TREES_HOLDOUT = 160
NCEI_TREES_FINAL = 240
MODEL_SCHEMA_VERSION = 1
SHADOW_STRATEGY_NAME = "station_maximum_v5_prospective_shadow"
ALLOWED_DECISION_MODES = {"event_driven", "forecast_update", "scheduled"}
FEATURE_JOIN_TOLERANCE_NS = 100_000

BASE_FEATURES = (
    "temperature_c",
    "temperature_change_1h",
    "temperature_change_3h",
    "temperature_change_6h",
    "hours_since_running_max",
    "peer_temperature_mean_c",
    "peer_temperature_max_c",
    "peer_temperature_min_c",
    "peer_temperature_spread_c",
    "peer_running_max_mean_c",
    "peer_running_max_max_c",
    "peer_temperature_change_1h_mean",
    "peer_age_minutes_mean",
    "day_of_year_sin",
    "day_of_year_cos",
    "local_time_sin",
    "local_time_cos",
    "solar_elevation_deg",
    "solar_clear_sky_factor",
    "minutes_to_sunset",
    "minutes_from_solar_noon",
    "historical_observation_count",
)
INTERACTION_FEATURES = (
    "running_max_c",
    "cooling_from_max_c",
    "peer_mean_delta_c",
    "peer_warmest_delta_c",
    "peer_running_max_delta_c",
    "solar_remaining_h",
    "after_solar_noon_h",
    "before_solar_noon_h",
    "warming_1h",
    "cooling_1h",
    "peer_warming_1h",
    "peer_cooling_1h",
    "thermal_solar_interaction",
    "peer_solar_interaction",
    "cooling_after_noon_interaction",
    "season_hour_sin",
    "season_hour_cos",
)
MODEL_FEATURES = (*BASE_FEATURES, *INTERACTION_FEATURES)


@dataclass(slots=True)
class StationBaseModel:
    station_id: str
    unit: str
    imputer: Any
    model: Any
    holdout_residuals: list[dict[str, float]]
    training_rows: int
    training_days: int


@dataclass(slots=True)
class StationMaximumV5Bundle:
    schema_version: int
    deployed_at_ns: int
    trained_through_local_date: str
    input_sha256: str
    station_models: dict[str, StationBaseModel]
    recent_calibration: dict[str, list[dict[str, Any]]]


def evaluate_station_maximum_v5(cfg: AppConfig) -> dict[str, Path]:

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    ncei_path = (
        cfg.reports_dir / "ncei_history" / "pretraining_features.csv.gz"
    )
    recent_path = cfg.reports_dir / "historical" / "predictive_features.csv"
    comparison_path = (
        cfg.reports_dir
        / "calibration_challenger"
        / "chronological_comparison.csv"
    )
    if not ncei_path.exists() or not recent_path.exists():
        raise FileNotFoundError(
            "V5 requires NCEI pretraining and current historical features"
        )
    ncei = pd.read_csv(ncei_path, low_memory=False)
    recent = pd.read_csv(recent_path, low_memory=False)
    incumbent = pd.read_csv(comparison_path)
    incumbent = incumbent[
        incumbent["decision_mode"].astype(str).eq("historical_scheduled")
    ].copy()
    incumbent_by_key = {
        (
            str(row["station_id"]),
            str(row["test_date"]),
            int(float(row["received_wall_ns"])),
        ): row
        for row in incumbent.to_dict(orient="records")
    }

    predictions: list[dict[str, Any]] = []
    station_model_audit: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        station_ncei = ncei[
            ncei["station_id"].astype(str).eq(station.station_id)
        ].copy()
        station_recent = recent[
            recent["station_id"].astype(str).eq(station.station_id)
        ].copy()
        if station_ncei.empty or station_recent.empty:
            continue
        model = _fit_station_base_model(station_ncei, station)
        no_interactions_model = _fit_station_base_model(
            station_ncei, station, feature_names=BASE_FEATURES
        )
        station_predictions = _chronological_station_predictions(
            station_recent,
            station,
            model,
            incumbent_by_key,
        )
        no_interactions_predictions = _chronological_station_predictions(
            station_recent,
            station,
            no_interactions_model,
            incumbent_by_key,
        )
        no_interactions_by_key = {
            (
                row["station_id"],
                row["local_date"],
                row["received_wall_ns"],
            ): row
            for row in no_interactions_predictions
        }
        for row in station_predictions:
            ablation = no_interactions_by_key.get(
                (
                    row["station_id"],
                    row["local_date"],
                    row["received_wall_ns"],
                ),
                {},
            )
            row["no_interactions_brier"] = ablation.get("v5_brier")
        predictions.extend(station_predictions)
        ranked_features = sorted(
            zip(
                MODEL_FEATURES,
                model.model.feature_importances_,
                strict=True,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        station_model_audit.append(
            {
                "station_id": station.station_id,
                "ncei_training_rows": model.training_rows,
                "ncei_training_days": model.training_days,
                "ncei_holdout_residuals": len(model.holdout_residuals),
                "ncei_2023_2025_holdout_mae_c": _mean(
                    [
                        abs(float(row["residual_c"]))
                        for row in model.holdout_residuals
                    ]
                ),
                "ncei_2023_2025_no_interactions_mae_c": _mean(
                    [
                        abs(float(row["residual_c"]))
                        for row in no_interactions_model.holdout_residuals
                    ]
                ),
                "first_recent_prediction": min(
                    (
                        row["local_date"]
                        for row in station_predictions
                    ),
                    default=None,
                ),
                "recent_predictions": len(station_predictions),
                "top_feature_importance": [
                    {"feature": name, "importance": float(importance)}
                    for name, importance in ranked_features[:10]
                ],
            }
        )

    probability_scores = _distribution_scores(predictions)
    station_scores = {
        station_id: _distribution_scores(
            [
                row
                for row in predictions
                if row["station_id"] == station_id
            ]
        )
        for station_id in sorted({row["station_id"] for row in predictions})
    }
    hour_scores = _hour_score_rows(predictions)
    monthly_scores = _monthly_score_rows(predictions)
    improvements = _station_day_improvements(
        predictions, "incumbent_brier", "v5_brier"
    )
    interaction_improvements = _station_day_improvements(
        predictions, "no_interactions_brier", "v5_brier"
    )
    market_improvements = _station_day_improvements(
        predictions, "market_brier", "v5_brier"
    )
    candidate_result = _candidate_economic_replay(cfg, predictions)
    current_brier = probability_scores["incumbent"]["brier"]
    v5_brier = probability_scores["v5"]["brier"]
    improvement_ci = _bootstrap_mean_ci(improvements)
    meteorological_gate = bool(
        current_brier is not None
        and v5_brier is not None
        and v5_brier < current_brier
        and improvement_ci is not None
        and improvement_ci["low"] > 0.0
    )
    economic_scores = candidate_result["probability_scores"]
    economic_gate = bool(
        economic_scores["v4_candidate_probability"]["brier"] is not None
        and economic_scores["market_candidate_probability"]["brier"] is not None
        and economic_scores["v4_candidate_probability"]["brier"]
        < economic_scores["market_candidate_probability"]["brier"]
        and candidate_result["edge_monotonicity"][
            "nondecreasing_fraction"
        ]
        >= 0.5
        and candidate_result["trade_summary"]["trades"] > 0
        and candidate_result["trade_summary"]["synthetic_value_total"] > 0.0
    )

    generated_at_ns = time.time_ns()
    input_hash = _input_hash(cfg)
    run_id = (
        datetime.fromtimestamp(generated_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{generated_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    output_dir = (
        cfg.reports_dir / "station_maximum_v5" / "artifacts" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "chronological_predictions.csv"
    hour_path = output_dir / "scores_by_station_hour.csv"
    month_path = output_dir / "scores_by_month.csv"
    candidates_path = output_dir / "candidate_predictions.csv"
    trades_path = output_dir / "lower_bound_trades.csv"
    edge_path = output_dir / "edge_bins.csv"
    summary_path = output_dir / "summary.json"
    _write_csv_atomic(predictions_path, predictions)
    _write_csv_atomic(hour_path, hour_scores)
    _write_csv_atomic(month_path, monthly_scores)
    _write_csv_atomic(
        candidates_path, candidate_result["candidate_predictions"]
    )
    _write_csv_atomic(trades_path, candidate_result["trades"])
    _write_csv_atomic(edge_path, candidate_result["edge_bins"])
    summary = {
        "generated_at_utc": datetime.fromtimestamp(
            generated_at_ns / 1e9, tz=UTC
        ).isoformat(),
        "run_id": run_id,
        "strategy": STRATEGY_NAME,
        "input_sha256": input_hash,
        "alpha_demonstrated": False,
        "status": (
            "meteorological_gate_passed_economic_gate_passed"
            if meteorological_gate and economic_gate
            else (
                "meteorological_gate_passed_economic_gate_failed"
                if meteorological_gate
                else "meteorological_gate_failed"
            )
        ),
        "method": {
            "long_history": (
                "Per-airport remaining-maximum ExtraTrees trained directly on "
                "2010-2025 NCEI point-in-time observations."
            ),
            "source_separation": (
                "The NCEI model uses only fields present historically. Forecast "
                "consensus is stacked later with a weight learned from strictly "
                "earlier 2026 WU station-days."
            ),
            "interactions": list(INTERACTION_FEATURES),
            "uncertainty": (
                "2023-2025 held-out NCEI residuals plus strictly earlier recent "
                "WU residuals, matched by airport, hour and 60-day season window."
            ),
            "forecast_weight": (
                "Airport/hour convex weight with ridge toward the multiyear "
                "observational base; provider spread shrinks forecast influence."
            ),
        },
        "ncei_period": {
            "start": str(ncei["local_date"].min()),
            "end": str(ncei["local_date"].max()),
            "rows": len(ncei),
            "holdout_start_year": NCEI_HOLDOUT_START_YEAR,
        },
        "station_models": station_model_audit,
        "chronological_predictions": len(predictions),
        "independent_station_days": len(
            {
                (row["station_id"], row["local_date"])
                for row in predictions
            }
        ),
        "probability_scores": probability_scores,
        "probability_scores_by_station": station_scores,
        "v5_minus_incumbent_brier_improvement_station_day": {
            "mean": _mean(improvements),
            "ci95": improvement_ci,
        },
        "interaction_ablation_station_day": {
            "mean_v5_improvement_over_no_interactions": _mean(
                interaction_improvements
            ),
            "ci95": _bootstrap_mean_ci(interaction_improvements),
        },
        "v5_minus_market_brier_improvement_station_day": {
            "mean": _mean(market_improvements),
            "ci95": _bootstrap_mean_ci(market_improvements),
        },
        "meteorological_gate_passed": meteorological_gate,
        "candidate_economic_replay": {
            key: value
            for key, value in candidate_result.items()
            if key
            not in {
                "candidate_predictions",
                "trades",
                "edge_bins",
            }
        },
        "economic_gate_passed": economic_gate,
        "deployment_decision": (
            "eligible_for_new_paper_challenger"
            if meteorological_gate and economic_gate
            else "do_not_replace_active_v4"
        ),
        "warning": (
            "All 2026 outcomes are development data. Passing these gates would "
            "still require prospective confirmation and would not prove alpha."
        ),
        "live_orders_enabled": False,
    }
    _write_json_atomic(summary_path, summary)
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(
        manifest_path, _artifact_manifest(output_dir, input_hash)
    )
    latest_path = cfg.reports_dir / "station_maximum_v5" / "latest.json"
    _write_json_atomic(
        latest_path,
        {
            "run_id": run_id,
            "artifact_directory": str(output_dir.resolve()),
            "summary_sha256": _sha256_file(summary_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
    )
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "predictions": predictions_path,
        "hour_scores": hour_path,
        "monthly_scores": month_path,
        "candidates": candidates_path,
        "trades": trades_path,
        "edge_bins": edge_path,
        "latest": latest_path,
    }


def deploy_station_maximum_v5_shadow(cfg: AppConfig) -> dict[str, Path]:

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    diagnostics = evaluate_station_maximum_v5(cfg)
    summary = json.loads(diagnostics["summary"].read_text(encoding="utf-8"))
    if not summary["meteorological_gate_passed"]:
        raise RuntimeError(
            "V5 did not pass the meteorological gate; shadow deployment refused"
        )
    ncei = pd.read_csv(
        cfg.reports_dir
        / "ncei_history"
        / "pretraining_features.csv.gz",
        low_memory=False,
    )
    recent = pd.read_csv(
        cfg.reports_dir / "historical" / "predictive_features.csv",
        low_memory=False,
    )
    station_models: dict[str, StationBaseModel] = {}
    calibration: dict[str, list[dict[str, Any]]] = {}
    for station in cfg.stations.values():
        station_ncei = ncei[
            ncei["station_id"].astype(str).eq(station.station_id)
        ].copy()
        station_recent = recent[
            recent["station_id"].astype(str).eq(station.station_id)
        ].copy()
        if station_ncei.empty or station_recent.empty:
            continue
        model = _fit_station_base_model(station_ncei, station)
        station_models[station.station_id] = model
        calibration[station.station_id] = _resolved_calibration_rows(
            station_recent, station, model
        )
    if set(station_models) != set(cfg.selected_station_ids):
        raise RuntimeError("V5 shadow is missing one or more station models")

    deployed_at_ns = time.time_ns()
    input_hash = _input_hash(cfg)
    trained_through = max(
        row["local_date"]
        for rows in calibration.values()
        for row in rows
    )
    bundle = StationMaximumV5Bundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        trained_through_local_date=trained_through,
        input_sha256=input_hash,
        station_models=station_models,
        recent_calibration=calibration,
    )
    deployment_id = (
        datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{deployed_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    root = cfg.reports_dir / "paper_shadow_station_maximum_v5"
    deployment_dir = root / "deployments" / deployment_id
    deployment_dir.mkdir(parents=True, exist_ok=False)
    model_path = deployment_dir / "model_bundle.pkl"
    model_path.write_bytes(
        pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    )
    deployment_path = deployment_dir / "deployment.json"
    _write_json_atomic(
        deployment_path,
        {
            "deployment_id": deployment_id,
            "strategy": SHADOW_STRATEGY_NAME,
            "schema_version": MODEL_SCHEMA_VERSION,
            "deployed_at_ns": deployed_at_ns,
            "deployed_at_utc": datetime.fromtimestamp(
                deployed_at_ns / 1e9, tz=UTC
            ).isoformat(),
            "trained_through_local_date": trained_through,
            "historical_input_sha256": input_hash,
            "model_sha256": _sha256_file(model_path),
            "source_diagnostics": str(diagnostics["summary"].resolve()),
            "source_diagnostics_sha256": _sha256_file(
                diagnostics["summary"]
            ),
            "meteorological_gate_passed": True,
            "economic_gate_passed": False,
            "purpose": (
                "Prospective probability comparison only; never opens a paper "
                "or real position."
            ),
            "live_orders_enabled": False,
        },
    )
    _write_json_atomic(
        deployment_dir / "state.json",
        {
            "deployment_id": deployment_id,
            "processed_decision_ids": [],
        },
    )
    _write_csv_atomic(deployment_dir / "shadow_predictions.csv", [])
    manifest_path = deployment_dir / "manifest.json"
    _write_json_atomic(
        manifest_path,
        _artifact_manifest(
            deployment_dir,
            input_hash,
            exclude={"state.json", "shadow_predictions.csv", "summary.json"},
        ),
    )
    active_path = root / "active_deployment.json"
    _write_json_atomic(
        active_path,
        {
            "deployment_id": deployment_id,
            "deployment_directory": str(deployment_dir.resolve()),
            "deployment_sha256": _sha256_file(deployment_path),
            "model_sha256": _sha256_file(model_path),
            "live_orders_enabled": False,
        },
    )
    return {
        "active": active_path,
        "deployment": deployment_path,
        "model": model_path,
        "manifest": manifest_path,
        "diagnostics": diagnostics["summary"],
    }


def run_station_maximum_v5_shadow_cycle(cfg: AppConfig) -> dict[str, Path]:

    root = cfg.reports_dir / "paper_shadow_station_maximum_v5"
    active_path = root / "active_deployment.json"
    if not active_path.exists():
        raise FileNotFoundError(
            "Missing V5 shadow deployment; run deploy-station-maximum-v5-shadow"
        )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    deployment_dir = Path(active["deployment_directory"])
    model_path = deployment_dir / "model_bundle.pkl"
    deployment_path = deployment_dir / "deployment.json"
    if _sha256_file(model_path) != active["model_sha256"]:
        raise RuntimeError("Active V5 shadow model hash mismatch")
    if _sha256_file(deployment_path) != active["deployment_sha256"]:
        raise RuntimeError("Active V5 shadow deployment hash mismatch")
    bundle = pickle.loads(model_path.read_bytes())
    if not isinstance(bundle, StationMaximumV5Bundle):
        raise TypeError("Invalid V5 shadow bundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise RuntimeError("V5 shadow schema mismatch")
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    source_path = (
        cfg.reports_dir
        / "paper_trading_challenger"
        / "paper_signal_audit.csv"
    )
    feature_rows = _load_feature_rows(cfg)
    feature_timelines = _build_feature_timelines(feature_rows)
    state_path = deployment_dir / "state.json"
    predictions_path = deployment_dir / "shadow_predictions.csv"
    summary_path = deployment_dir / "summary.json"
    state = _read_json(
        state_path,
        {
            "deployment_id": deployment["deployment_id"],
            "processed_decision_ids": [],
        },
    )
    processed = set(state.get("processed_decision_ids") or [])
    predictions = _load_csv(predictions_path)
    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    deployed_at_ns = int(deployment["deployed_at_ns"])
    cycle_started_ns = time.time_ns()
    source_rows = _load_csv(source_path)
    label_vocabulary = _market_label_vocabulary(source_rows)
    for source_row in source_rows:
        decision_id = str(source_row.get("decision_id") or "")
        station_id = str(source_row.get("station_id") or "")
        decision_mode = str(source_row.get("decision_mode") or "")
        timestamp_ns = _timestamp_ns(source_row.get("received_wall_ns"))
        if (
            not decision_id
            or decision_id in processed
            or station_id not in stations
            or decision_mode not in ALLOWED_DECISION_MODES
            or timestamp_ns is None
            or timestamp_ns <= deployed_at_ns
        ):
            continue
        processed.add(decision_id)
        station = stations[station_id]
        local_dt = datetime.fromtimestamp(
            timestamp_ns / 1e9, tz=UTC
        ).astimezone(ZoneInfo(station.timezone))
        local_date = local_dt.date().isoformat()
        feature_row = _match_feature_row(
            feature_rows,
            feature_timelines,
            station_id,
            local_date,
            decision_mode,
            timestamp_ns,
        )
        merged = {
            **source_row,
            **feature_row,
            "market_labels_json": json.dumps(
                sorted(
                    label_vocabulary.get(
                        (station_id, local_date), set()
                    )
                )
            ),
        }
        result, rejection = _score_live_shadow(
            merged,
            station,
            bundle.station_models[station_id],
            bundle.recent_calibration[station_id],
            local_dt,
        )
        row = {
            "decision_id": decision_id,
            "station_id": station_id,
            "local_date": local_date,
            "local_hour": local_dt.hour + local_dt.minute / 60.0,
            "decision_mode": decision_mode,
            "trigger_source": source_row.get("trigger_source"),
            "received_utc": source_row.get("received_utc"),
            "received_wall_ns": timestamp_ns,
            "processed_at_ns": cycle_started_ns,
            "processing_lag_seconds": max(
                0.0, (cycle_started_ns - timestamp_ns) / 1e9
            ),
            "rejection_reason": rejection,
            "settled": False,
            "live_orders_enabled": False,
        }
        if result is not None:
            row.update(result)
        predictions.append(row)
    _settle_shadow_predictions(cfg, predictions)
    _write_csv_atomic(predictions_path, predictions)
    state.update(
        {
            "processed_decision_ids": sorted(processed)[-50_000:],
            "last_cycle_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write_json_atomic(state_path, state)
    settled = [
        row
        for row in predictions
        if _truth(row.get("settled"))
        and _finite(row.get("v5_brier")) is not None
    ]
    daily_improvements = _station_day_improvements(
        settled, "incumbent_brier", "v5_brier"
    )
    _write_json_atomic(
        summary_path,
        {
            "strategy": SHADOW_STRATEGY_NAME,
            "deployment_id": deployment["deployment_id"],
            "trained_through_local_date": bundle.trained_through_local_date,
            "processed_decisions": len(processed),
            "prediction_rows": len(predictions),
            "scored_rows": sum(
                not bool(row.get("rejection_reason")) for row in predictions
            ),
            "settled_rows": len(settled),
            "settled_station_days": len(
                {
                    (row["station_id"], row["local_date"])
                    for row in settled
                }
            ),
            "mean_v5_brier": _mean(
                [_finite(row.get("v5_brier")) for row in settled]
            ),
            "mean_incumbent_brier": _mean(
                [_finite(row.get("incumbent_brier")) for row in settled]
            ),
            "mean_station_day_brier_improvement": _mean(
                daily_improvements
            ),
            "brier_improvement_ci95": _bootstrap_mean_ci(
                daily_improvements
            ),
            "paper_trades": 0,
            "live_orders_enabled": False,
        },
    )
    return {
        "active": active_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "state": state_path,
    }


def _resolved_calibration_rows(
    station_recent: Any,
    station: StationConfig,
    model: StationBaseModel,
) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    physical = build_physical_interaction_frame(
        station_recent, station, ncei=False
    )
    feature_names = getattr(
        model.model, "_weather_alpha_feature_names", MODEL_FEATURES
    )
    x = model.imputer.transform(physical[list(feature_names)])
    base = physical["running_max_c"].to_numpy(dtype=float) + np.maximum(
        0.0, model.model.predict(x)
    )
    forecast = _coalesce_numeric(
        station_recent,
        (
            "historical_model_consensus_remaining_max_c",
            "historical_model_consensus_day1_max_c",
        ),
    )
    dates = pd.to_datetime(station_recent["local_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for index, source_index in enumerate(station_recent.index):
        target = _finite(physical.loc[source_index, "target_c"])
        forecast_value = _finite(forecast.loc[source_index])
        running = _finite(physical.loc[source_index, "running_max_c"])
        hour = _finite(station_recent.loc[source_index, "local_hour"])
        if None in (target, forecast_value, running, hour):
            continue
        rows.append(
            {
                "local_date": str(
                    station_recent.loc[source_index, "local_date"]
                ),
                "local_hour": float(hour),
                "day_of_year": int(dates.loc[source_index].dayofyear),
                "running_max_c": float(running),
                "base_point_c": float(base[index]),
                "forecast_point_c": max(
                    float(running), float(forecast_value)
                ),
                "target_c": float(target),
            }
        )
    return rows


def _build_feature_timelines(
    rows: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]]:
    timelines: dict[
        tuple[str, str, str], list[tuple[int, dict[str, Any]]]
    ] = defaultdict(list)
    for (station_id, local_date, mode, timestamp), row in rows.items():
        if (timestamp_ns := _timestamp_ns(timestamp)) is not None:
            timelines[(station_id, local_date, mode)].append(
                (timestamp_ns, row)
            )
    for timeline in timelines.values():
        timeline.sort(key=lambda item: item[0])
    return dict(timelines)


def _match_feature_row(
    rows: dict[tuple[str, str, str, str], dict[str, Any]],
    timelines: dict[
        tuple[str, str, str], list[tuple[int, dict[str, Any]]]
    ],
    station_id: str,
    local_date: str,
    mode: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    exact = rows.get(
        (station_id, local_date, mode, str(timestamp_ns))
    )
    if exact is not None:
        return exact
    timeline = timelines.get((station_id, local_date, mode), [])
    if not timeline:
        return {}
    timestamps = [item[0] for item in timeline]
    position = bisect_left(timestamps, timestamp_ns)
    candidates = timeline[max(0, position - 1) : position + 1]
    if not candidates:
        return {}
    closest_timestamp, closest_row = min(
        candidates, key=lambda item: abs(item[0] - timestamp_ns)
    )
    if abs(closest_timestamp - timestamp_ns) > FEATURE_JOIN_TOLERANCE_NS:
        return {}
    return closest_row


def _score_live_shadow(
    feature_row: dict[str, Any],
    station: StationConfig,
    model: StationBaseModel,
    calibration_rows: list[dict[str, Any]],
    local_dt: datetime,
) -> tuple[dict[str, Any] | None, str]:
    import pandas as pd

    frame = pd.DataFrame([feature_row])
    physical = build_physical_interaction_frame(
        frame, station, ncei=False
    )
    running = _finite(physical.iloc[0].get("running_max_c"))
    if running is None:
        return None, "missing_resolution_running_max"
    feature_names = getattr(
        model.model, "_weather_alpha_feature_names", MODEL_FEATURES
    )
    x = model.imputer.transform(physical[list(feature_names)])
    base_increment = max(0.0, float(model.model.predict(x)[0]))
    base_point = running + base_increment
    forecast = _first_finite(
        feature_row,
        "model_consensus_remaining_max_c",
        "model_consensus_today_max_c",
        "calibration_forecast_point",
    )
    if forecast is not None and station.market_unit == "F" and (
        forecast > 45.0
    ):
        forecast = (forecast - 32.0) * 5.0 / 9.0
    spread = _first_finite(
        feature_row,
        "model_spread_remaining_max_c",
        "model_spread_today_max_c",
        "forecast_spread_c",
    )
    nearest_hour = min(
        {int(float(row["local_hour"])) for row in calibration_rows},
        key=lambda hour: abs(hour - local_dt.hour),
    )
    calibration = pd.DataFrame(
        [
            row
            for row in calibration_rows
            if int(float(row["local_hour"])) == nearest_hour
        ]
    )
    raw_weight = _forecast_weight(calibration)
    spread_confidence = (
        0.5
        if spread is None
        else 1.0 / math.sqrt(1.0 + (max(0.0, spread) / 3.0) ** 2)
    )
    forecast_weight = raw_weight * spread_confidence
    center = (
        base_point
        if forecast is None
        else base_point + forecast_weight * (forecast - base_point)
    )
    center = max(running, center)
    day_of_year = local_dt.timetuple().tm_yday
    ncei_residuals = _seasonal_residuals(
        model.holdout_residuals, nearest_hour, day_of_year
    )
    recent_residuals = _recent_residuals(
        calibration, day_of_year, forecast_weight
    )
    residuals = [
        *ncei_residuals,
        *[
            residual
            for residual in recent_residuals
            for _ in range(RECENT_RESIDUAL_WEIGHT)
        ],
    ]
    incumbent = _probabilities(
        feature_row.get("weather_distribution_json")
    )
    market = _probabilities(feature_row.get("market_distribution_json"))
    labels = sorted(
        set(incumbent)
        | set(market)
        | set(_json_list(feature_row.get("market_labels_json")))
    )
    if not labels:
        return None, "missing_bucket_vocabulary"
    distribution = _distribution_from_residuals(
        center, running, residuals, labels, station.market_unit
    )
    if not distribution:
        return None, "v5_distribution_unavailable"
    point_market = (
        center * 9.0 / 5.0 + 32.0
        if station.market_unit == "F"
        else center
    )
    return (
        {
            "nearest_calibration_hour": nearest_hour,
            "running_max_c": running,
            "base_point_c": base_point,
            "forecast_point_c": forecast,
            "raw_forecast_weight": raw_weight,
            "forecast_spread_c": spread,
            "effective_forecast_weight": forecast_weight,
            "v5_point_prediction_market_unit": point_market,
            "v5_distribution_json": json.dumps(
                distribution, sort_keys=True
            ),
            "incumbent_distribution_json": json.dumps(
                incumbent, sort_keys=True
            ),
            "market_distribution_json": json.dumps(market, sort_keys=True),
            "ncei_residual_samples": len(ncei_residuals),
            "recent_residual_samples": len(recent_residuals),
        },
        "",
    )


def _settle_shadow_predictions(
    cfg: AppConfig, predictions: list[dict[str, Any]]
) -> None:
    ledger_path = (
        cfg.reports_dir / "resolutions" / "resolution_ledger.csv"
    )
    winners = {
        (
            str(row.get("station_id") or ""),
            str(row.get("event_date") or ""),
        ): str(row.get("official_winning_bucket") or "")
        for row in _load_csv(ledger_path)
    }
    for row in predictions:
        if _truth(row.get("settled")) or row.get("rejection_reason"):
            continue
        winner = winners.get(
            (str(row["station_id"]), str(row["local_date"]))
        )
        if not winner:
            continue
        v5 = _json_probabilities(row.get("v5_distribution_json"))
        incumbent = _json_probabilities(
            row.get("incumbent_distribution_json")
        )
        market = _json_probabilities(row.get("market_distribution_json"))
        row.update(
            {
                "winning_bucket": winner,
                "v5_brier": _brier(v5, winner),
                "incumbent_brier": _brier(incumbent, winner),
                "market_brier": (
                    _brier(_normalize(market), winner)
                    if sum(market.values()) >= 0.95
                    else ""
                ),
                "settled": True,
            }
        )


def build_physical_interaction_frame(
    frame: Any,
    station: StationConfig,
    *,
    ncei: bool,
) -> Any:

    import numpy as np
    import pandas as pd

    output = pd.DataFrame(index=frame.index)

    def numeric(name: str) -> Any:
        if name not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    for name in BASE_FEATURES:
        output[name] = numeric(name)
    if ncei:
        running_max = numeric("ncei_running_max_c")
        target = numeric("ncei_final_max_c")
    else:
        running_max = numeric("resolution_running_max_market_unit")
        target = numeric("final_wu_max")
        if station.market_unit == "F":
            running_max = (running_max - 32.0) * 5.0 / 9.0
            target = (target - 32.0) * 5.0 / 9.0
    temperature = numeric("temperature_c")
    output["running_max_c"] = running_max
    output["target_c"] = target
    output["remaining_increment_c"] = (target - running_max).clip(lower=0.0)
    output["cooling_from_max_c"] = temperature - running_max
    output["peer_mean_delta_c"] = (
        numeric("peer_temperature_mean_c") - temperature
    )
    output["peer_warmest_delta_c"] = (
        numeric("peer_temperature_max_c") - temperature
    )
    output["peer_running_max_delta_c"] = (
        numeric("peer_running_max_mean_c") - running_max
    )
    output["solar_remaining_h"] = (
        numeric("minutes_to_sunset").clip(lower=0.0) / 60.0
    )
    output["after_solar_noon_h"] = (
        numeric("minutes_from_solar_noon").clip(lower=0.0) / 60.0
    )
    output["before_solar_noon_h"] = (
        -numeric("minutes_from_solar_noon")
    ).clip(lower=0.0) / 60.0
    output["warming_1h"] = numeric("temperature_change_1h").clip(lower=0.0)
    output["cooling_1h"] = (
        -numeric("temperature_change_1h")
    ).clip(lower=0.0)
    output["peer_warming_1h"] = numeric(
        "peer_temperature_change_1h_mean"
    ).clip(lower=0.0)
    output["peer_cooling_1h"] = (
        -numeric("peer_temperature_change_1h_mean")
    ).clip(lower=0.0)
    output["thermal_solar_interaction"] = (
        output["warming_1h"] * numeric("solar_clear_sky_factor")
    )
    output["peer_solar_interaction"] = (
        output["peer_warming_1h"] * output["solar_remaining_h"]
    )
    output["cooling_after_noon_interaction"] = (
        output["cooling_1h"] * output["after_solar_noon_h"]
    )
    output["season_hour_sin"] = (
        numeric("day_of_year_sin") * numeric("local_time_sin")
    )
    output["season_hour_cos"] = (
        numeric("day_of_year_cos") * numeric("local_time_cos")
    )
    return output


def _fit_station_base_model(
    station_ncei: Any,
    station: StationConfig,
    *,
    feature_names: tuple[str, ...] = MODEL_FEATURES,
) -> StationBaseModel:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    features = build_physical_interaction_frame(
        station_ncei, station, ncei=True
    )
    dates = pd.to_datetime(station_ncei["local_date"], errors="coerce")
    target = features["remaining_increment_c"].astype(float)
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(features[list(feature_names)])
    old_mask = dates.dt.year < NCEI_HOLDOUT_START_YEAR
    holdout_mask = dates.dt.year >= NCEI_HOLDOUT_START_YEAR
    holdout_model = ExtraTreesRegressor(
        n_estimators=NCEI_TREES_HOLDOUT,
        min_samples_leaf=NCEI_MIN_SAMPLES_LEAF,
        max_features=0.8,
        n_jobs=-1,
        random_state=20260731,
    )
    holdout_model.fit(x[old_mask], target[old_mask])
    holdout_prediction = np.maximum(
        0.0, holdout_model.predict(x[holdout_mask])
    )
    holdout_rows = station_ncei.loc[holdout_mask]
    holdout_residuals = [
        {
            "local_hour": float(hour),
            "day_of_year": float(day_of_year),
            "residual_c": float(actual - prediction),
        }
        for hour, day_of_year, actual, prediction in zip(
            pd.to_numeric(holdout_rows["local_hour"], errors="coerce"),
            dates.loc[holdout_mask].dt.dayofyear,
            target[holdout_mask],
            holdout_prediction,
            strict=True,
        )
        if math.isfinite(float(hour))
        and math.isfinite(float(day_of_year))
        and math.isfinite(float(actual))
    ]
    latest = dates.max()
    age_days = (latest - dates).dt.days.clip(lower=0)
    half_life_days = 365.25 * 6.0
    weights = 0.5 ** (age_days.to_numpy(dtype=float) / half_life_days)
    final_model = ExtraTreesRegressor(
        n_estimators=NCEI_TREES_FINAL,
        min_samples_leaf=NCEI_MIN_SAMPLES_LEAF,
        max_features=0.8,
        n_jobs=-1,
        random_state=20260732,
    )
    final_model.fit(x, target, sample_weight=weights)
    final_model._weather_alpha_feature_names = tuple(feature_names)
    return StationBaseModel(
        station_id=station.station_id,
        unit=station.market_unit,
        imputer=imputer,
        model=final_model,
        holdout_residuals=holdout_residuals,
        training_rows=len(station_ncei),
        training_days=station_ncei["local_date"].astype(str).nunique(),
    )


def _chronological_station_predictions(
    station_recent: Any,
    station: StationConfig,
    model: StationBaseModel,
    incumbent_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    physical = build_physical_interaction_frame(
        station_recent, station, ncei=False
    )
    feature_names = getattr(
        model.model, "_weather_alpha_feature_names", MODEL_FEATURES
    )
    x = model.imputer.transform(physical[list(feature_names)])
    base_increment = np.maximum(0.0, model.model.predict(x))
    working = physical.copy()
    working["base_point_c"] = working["running_max_c"] + base_increment
    working["forecast_point_c"] = _coalesce_numeric(
        station_recent,
        (
            "historical_model_consensus_remaining_max_c",
            "historical_model_consensus_day1_max_c",
        ),
    )
    working["forecast_point_c"] = np.maximum(
        working["running_max_c"], working["forecast_point_c"]
    )
    working["forecast_spread_c"] = _coalesce_numeric(
        station_recent,
        (
            "historical_model_spread_remaining_max_c",
            "historical_model_spread_day1_max_c",
        ),
    )
    working["local_date"] = station_recent["local_date"].astype(str)
    working["local_hour"] = pd.to_numeric(
        station_recent["local_hour"], errors="coerce"
    )
    working["day_of_year"] = pd.to_datetime(
        working["local_date"], errors="coerce"
    ).dt.dayofyear
    working["received_wall_ns"] = pd.to_numeric(
        station_recent["received_wall_ns"], errors="coerce"
    )
    working["market_labels_json"] = station_recent["market_labels_json"]
    working["market_distribution_json"] = station_recent[
        "market_distribution_json"
    ]
    working["winning_bucket"] = station_recent["winning_bucket"].astype(str)
    output: list[dict[str, Any]] = []
    for test_date in sorted(working["local_date"].unique()):
        prior = working[working["local_date"] < test_date]
        testing = working[working["local_date"] == test_date]
        if prior["local_date"].nunique() < MINIMUM_RECENT_CALIBRATION_DAYS:
            continue
        for index, row in testing.iterrows():
            hour_prior = prior[
                prior["local_hour"].eq(row["local_hour"])
            ].dropna(
                subset=[
                    "forecast_point_c",
                    "base_point_c",
                    "target_c",
                ]
            )
            if (
                hour_prior["local_date"].nunique()
                < MINIMUM_HOUR_CALIBRATION_DAYS
            ):
                hour_prior = prior.dropna(
                    subset=[
                        "forecast_point_c",
                        "base_point_c",
                        "target_c",
                    ]
                )
            raw_forecast_weight = _forecast_weight(hour_prior)
            spread = _finite(row.get("forecast_spread_c"))
            spread_confidence = (
                0.5
                if spread is None
                else 1.0 / math.sqrt(1.0 + (max(0.0, spread) / 3.0) ** 2)
            )
            forecast_weight = raw_forecast_weight * spread_confidence
            base_point = float(row["base_point_c"])
            forecast_point = _finite(row.get("forecast_point_c"))
            center = (
                base_point
                if forecast_point is None
                else base_point
                + forecast_weight * (forecast_point - base_point)
            )
            center = max(float(row["running_max_c"]), center)
            ncei_residuals = _seasonal_residuals(
                model.holdout_residuals,
                float(row["local_hour"]),
                float(row["day_of_year"]),
            )
            recent_residuals = _recent_residuals(
                hour_prior,
                float(row["day_of_year"]),
                forecast_weight,
            )
            combined_residuals = [
                *ncei_residuals,
                *[
                    residual
                    for residual in recent_residuals
                    for _ in range(RECENT_RESIDUAL_WEIGHT)
                ],
            ]
            labels = _json_list(row.get("market_labels_json"))
            v5_distribution = _distribution_from_residuals(
                center,
                float(row["running_max_c"]),
                combined_residuals,
                labels,
                station.market_unit,
            )
            base_distribution = _distribution_from_residuals(
                base_point,
                float(row["running_max_c"]),
                ncei_residuals,
                labels,
                station.market_unit,
            )
            raw_market_distribution = _json_probabilities(
                row.get("market_distribution_json")
            )
            market_mass = sum(raw_market_distribution.values())
            market_distribution = (
                _normalize(raw_market_distribution)
                if market_mass >= 0.95
                else {}
            )
            winner = str(row["winning_bucket"])
            timestamp_ns = int(float(row["received_wall_ns"]))
            incumbent = incumbent_by_key.get(
                (station.station_id, test_date, timestamp_ns), {}
            )
            incumbent_distribution = _json_probabilities(
                incumbent.get("challenger_distribution_json")
            )
            current_brier = _finite(incumbent.get("challenger_brier"))
            if not v5_distribution or not winner or current_brier is None:
                continue
            point_market = (
                center * 9.0 / 5.0 + 32.0
                if station.market_unit == "F"
                else center
            )
            target_market = (
                float(row["target_c"]) * 9.0 / 5.0 + 32.0
                if station.market_unit == "F"
                else float(row["target_c"])
            )
            output.append(
                {
                    "station_id": station.station_id,
                    "local_date": test_date,
                    "local_hour": int(float(row["local_hour"])),
                    "received_wall_ns": timestamp_ns,
                    "winning_bucket": winner,
                    "base_point_c": base_point,
                    "forecast_point_c": forecast_point,
                    "raw_forecast_weight": raw_forecast_weight,
                    "forecast_spread_c": spread,
                    "forecast_spread_confidence": spread_confidence,
                    "effective_forecast_weight": forecast_weight,
                    "v5_point_prediction_market_unit": point_market,
                    "final_wu_max": target_market,
                    "v5_absolute_error": abs(point_market - target_market),
                    "ncei_residual_samples": len(ncei_residuals),
                    "recent_residual_samples": len(recent_residuals),
                    "base_distribution_json": json.dumps(
                        base_distribution, sort_keys=True
                    ),
                    "v5_distribution_json": json.dumps(
                        v5_distribution, sort_keys=True
                    ),
                    "incumbent_distribution_json": json.dumps(
                        incumbent_distribution, sort_keys=True
                    ),
                    "market_distribution_json": json.dumps(
                        market_distribution, sort_keys=True
                    ),
                    "base_brier": _brier(base_distribution, winner),
                    "v5_brier": _brier(v5_distribution, winner),
                    "incumbent_brier": current_brier,
                    "market_brier": (
                        _brier(market_distribution, winner)
                        if market_distribution
                        else None
                    ),
                }
            )
    return output


def _candidate_economic_replay(
    cfg: AppConfig, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    recent_rows = {
        (
            str(row["station_id"]),
            str(row["local_date"]),
            int(float(row["received_wall_ns"])),
        ): row
        for row in _load_historical_feature_rows(cfg)
    }
    snapshots: list[dict[str, Any]] = []
    for row in predictions:
        key = (
            str(row["station_id"]),
            str(row["local_date"]),
            int(row["received_wall_ns"]),
        )
        recent = recent_rows.get(key)
        if recent is None:
            continue
        snapshots.append(
            {
                "station_id": row["station_id"],
                "local_date": row["local_date"],
                "local_hour": row["local_hour"],
                "received_wall_ns": row["received_wall_ns"],
                "winning_bucket": row["winning_bucket"],
                "weather_distribution": _json_probabilities(
                    row["v5_distribution_json"]
                ),
                "market_distribution": _json_probabilities(
                    recent.get("market_distribution_json")
                ),
            }
        )
    validation_rows, _ = _load_validation_rows(cfg)
    candidates = _build_candidates(
        cfg,
        snapshots,
        _recent_dates_by_station(snapshots),
        validation_rows,
        _load_feature_rows(cfg),
    )
    candidates = [
        row
        for row in candidates
        if bool(row.get("peer_history_point_in_time"))
        and float(row.get("peer_station_coverage") or 0.0)
        >= MINIMUM_PEER_COVERAGE
    ]
    candidate_predictions, skipped = _rolling_predictions(candidates)
    trades = _replay_lower_bound_strategy(candidate_predictions)
    edge_bins = _edge_bin_rows(candidate_predictions)
    return {
        "candidates": len(candidates),
        "rolling_predictions": len(candidate_predictions),
        "rolling_skipped": len(skipped),
        "probability_scores": _v4_probability_scores(
            candidate_predictions
        ),
        "probability_scores_by_station": {
            station_id: _v4_probability_scores(
                [
                    row
                    for row in candidate_predictions
                    if row["station_id"] == station_id
                ]
            )
            for station_id in sorted(
                {
                    str(row["station_id"])
                    for row in candidate_predictions
                }
            )
        },
        "edge_monotonicity": _edge_monotonicity(edge_bins),
        "trade_summary": _trade_summary(trades),
        "forced_entry_sensitivity": _forced_entry_sensitivity(
            candidate_predictions
        ),
        "candidate_predictions": candidate_predictions,
        "trades": trades,
        "edge_bins": edge_bins,
    }


def _forced_entry_sensitivity(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        edge = _finite(row.get("v4_estimated_edge"))
        if edge is None:
            continue
        key = (str(row.get("station_id")), str(row.get("local_date")))
        incumbent = best.get(key)
        if incumbent is None or edge > float(
            incumbent["v4_estimated_edge"]
        ):
            best[key] = row
    output: list[dict[str, Any]] = []
    for threshold in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        selected = [
            row
            for row in best.values()
            if float(row["v4_estimated_edge"]) >= threshold
        ]
        values = [
            float(row["synthetic_value_per_share"]) for row in selected
        ]
        wins = sum(_truth(row.get("won")) for row in selected)
        output.append(
            {
                "minimum_estimated_edge": threshold,
                "station_day_entries": len(selected),
                "wins": wins,
                "hit_rate": wins / len(selected) if selected else None,
                "synthetic_value_total": sum(values),
                "synthetic_value_mean": _mean(values),
                "research_only": True,
            }
        )
    return output


def _distribution_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(field: str) -> dict[str, Any]:
        values = [
            float(row[field])
            for row in rows
            if _finite(row.get(field)) is not None
        ]
        return {
            "rows": len(values),
            "brier": _mean(values),
        }

    return {
        "long_history_base": score("base_brier"),
        "no_interactions": score("no_interactions_brier"),
        "v5": score("v5_brier"),
        "incumbent": score("incumbent_brier"),
        "market": score("market_brier"),
        "point_mae": {
            "rows": len(rows),
            "mae": _mean(
                [float(row["v5_absolute_error"]) for row in rows]
            ),
        },
    }


def _hour_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["station_id"]), int(row["local_hour"])) for row in rows}
    )
    for station_id, local_hour in keys:
        selected = [
            row
            for row in rows
            if row["station_id"] == station_id
            and int(row["local_hour"]) == local_hour
        ]
        scores = _distribution_scores(selected)
        output.append(
            {
                "station_id": station_id,
                "local_hour": local_hour,
                "rows": len(selected),
                "point_mae": scores["point_mae"]["mae"],
                "long_history_base_brier": scores[
                    "long_history_base"
                ]["brier"],
                "v5_brier": scores["v5"]["brier"],
                "incumbent_brier": scores["incumbent"]["brier"],
                "market_brier": scores["market"]["brier"],
                "mean_effective_forecast_weight": _mean(
                    [
                        float(row["effective_forecast_weight"])
                        for row in selected
                    ]
                ),
            }
        )
    return output


def _monthly_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for month in sorted({str(row["local_date"])[:7] for row in rows}):
        selected = [
            row
            for row in rows
            if str(row["local_date"]).startswith(month)
        ]
        scores = _distribution_scores(selected)
        output.append(
            {
                "month": month,
                "rows": len(selected),
                "station_days": len(
                    {
                        (row["station_id"], row["local_date"])
                        for row in selected
                    }
                ),
                "v5_brier": scores["v5"]["brier"],
                "incumbent_brier": scores["incumbent"]["brier"],
                "market_brier": scores["market"]["brier"],
                "point_mae": scores["point_mae"]["mae"],
            }
        )
    return output


def _station_day_improvements(
    rows: list[dict[str, Any]], baseline: str, challenger: str
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        baseline_value = _finite(row.get(baseline))
        challenger_value = _finite(row.get(challenger))
        if baseline_value is None or challenger_value is None:
            continue
        grouped.setdefault(
            (str(row["station_id"]), str(row["local_date"])), []
        ).append(baseline_value - challenger_value)
    return [
        sum(values) / len(values) for values in grouped.values() if values
    ]


def _forecast_weight(prior: Any) -> float:
    import numpy as np

    if prior.empty:
        return 0.0
    delta = (
        prior["forecast_point_c"].to_numpy(dtype=float)
        - prior["base_point_c"].to_numpy(dtype=float)
    )
    target = (
        prior["target_c"].to_numpy(dtype=float)
        - prior["base_point_c"].to_numpy(dtype=float)
    )
    denominator = float(delta @ delta) + FORECAST_WEIGHT_RIDGE
    return float(
        np.clip(float(delta @ target) / max(denominator, 1e-12), 0.0, 1.0)
    )


def _seasonal_residuals(
    residual_rows: list[dict[str, float]],
    local_hour: float,
    day_of_year: float,
) -> list[float]:
    selected = [
        row["residual_c"]
        for row in residual_rows
        if int(row["local_hour"]) == int(local_hour)
        and _circular_day_distance(row["day_of_year"], day_of_year)
        <= SEASONAL_RESIDUAL_WINDOW_DAYS
    ]
    if len(selected) >= 60:
        return selected
    return [
        row["residual_c"]
        for row in residual_rows
        if int(row["local_hour"]) == int(local_hour)
    ]


def _recent_residuals(
    prior: Any, day_of_year: float, forecast_weight: float
) -> list[float]:
    selected = prior[
        prior["day_of_year"].apply(
            lambda value: _circular_day_distance(
                float(value), day_of_year
            )
            <= SEASONAL_RESIDUAL_WINDOW_DAYS
        )
    ]
    if len(selected) < 8:
        selected = prior
    centers = selected["base_point_c"] + forecast_weight * (
        selected["forecast_point_c"] - selected["base_point_c"]
    )
    return [
        float(value)
        for value in (selected["target_c"] - centers).dropna()
    ]


def _distribution_from_residuals(
    center_c: float,
    current_max_c: float,
    residuals: list[float],
    labels: list[str],
    unit: str,
) -> dict[str, float]:
    if not residuals or not labels:
        return {}
    counts = {label: 0.25 for label in labels}
    for residual in residuals:
        sample_c = max(current_max_c, center_c + float(residual))
        sample_market = (
            sample_c * 9.0 / 5.0 + 32.0 if unit == "F" else sample_c
        )
        label = _label_for_value(sample_market, labels, unit)
        if label is not None:
            counts[label] += 1.0
    return _normalize(counts)


def _load_historical_feature_rows(cfg: AppConfig) -> list[dict[str, Any]]:
    import pandas as pd

    path = cfg.reports_dir / "historical" / "predictive_features.csv"
    return pd.read_csv(path, low_memory=False).to_dict(orient="records")


def _coalesce_numeric(frame: Any, names: tuple[str, ...]) -> Any:
    import numpy as np
    import pandas as pd

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame:
            result = result.fillna(
                pd.to_numeric(frame[name], errors="coerce")
            )
    return result


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _json_probabilities(value: Any) -> dict[str, float]:
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(label): probability
        for label, raw in parsed.items()
        if (probability := _finite(raw)) is not None
    }


def _market_label_vocabulary(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:

    output: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(row.get("station_id") or ""),
            str(row.get("local_date") or ""),
        )
        if not all(key):
            continue
        output[key].update(
            _json_list(row.get("market_labels_json"))
        )
        for field in (
            "market_distribution_json",
            "market_bid_json",
            "market_ask_json",
            "weather_distribution_json",
        ):
            output[key].update(_json_probabilities(row.get(field)))
    return dict(output)


def _circular_day_distance(left: float, right: float) -> float:
    raw = abs(left - right)
    return min(raw, 365.25 - raw)


def _input_hash(cfg: AppConfig) -> str:
    digest = hashlib.sha256()
    paths = (
        cfg.reports_dir
        / "ncei_history"
        / "pretraining_features.csv.gz",
        cfg.reports_dir / "historical" / "predictive_features.csv",
        cfg.reports_dir
        / "calibration_challenger"
        / "chronological_comparison.csv",
        cfg.project_root / "config" / "stations.yaml",
    )
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_finite(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if (value := _finite(row.get(name))) is not None:
            return value
    return None


def _mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None
