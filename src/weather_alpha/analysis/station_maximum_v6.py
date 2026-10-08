from __future__ import annotations

import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from .model import _bootstrap_mean_ci, _brier, _normalize
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import (
    _canonical_bucket,
    _timestamp_ns,
)
from .station_specific_alpha import _load_csv, _read_json, _truth
from .station_maximum_v5 import (
    ALLOWED_DECISION_MODES,
    RECENT_RESIDUAL_WEIGHT,
    SEASONAL_RESIDUAL_WINDOW_DAYS,
    StationBaseModel,
    StationMaximumV5Bundle,
    _build_feature_timelines,
    _candidate_economic_replay,
    _chronological_station_predictions,
    _circular_day_distance,
    _distribution_from_residuals,
    _finite,
    _fit_station_base_model,
    _forced_entry_sensitivity,
    _input_hash,
    _json_list,
    _json_probabilities,
    _match_feature_row,
    _market_label_vocabulary,
    _mean,
    _resolved_calibration_rows,
    _score_live_shadow,
    _seasonal_residuals,
)
from .station_tail_diagnostics import _load_feature_rows
from .station_tail_v4 import (
    _artifact_manifest,
    _sha256_file,
)


STRATEGY_NAME = "station_maximum_provider_expert_mixture_v6"
SHADOW_STRATEGY_NAME = "station_maximum_v6_prospective_shadow"
MODEL_SCHEMA_VERSION = 1
PROVIDER_STACK_RIDGE = 20.0
EXPERT_MIXTURE_RIDGE = 8.0
MINIMUM_PROVIDER_TRAINING_DAYS = 20
MINIMUM_EXPERT_MIXTURE_DAYS = 12

STATION_PROVIDER_NAMES = {
    "EGLC": ("best_match", "dwd_icon", "ecmwf"),
    "EHAM": ("best_match", "dwd_icon", "ecmwf"),
    "KLAX": ("best_match", "ecmwf", "gfs_hrrr"),
    "KLGA": ("best_match", "ecmwf", "gfs_hrrr"),
}


@dataclass(slots=True)
class ProviderHourModel:
    provider_names: tuple[str, ...]
    model: Any


@dataclass(slots=True)
class StationMaximumV6Bundle:
    schema_version: int
    deployed_at_ns: int
    trained_through_local_date: str
    input_sha256: str
    v5_bundle: StationMaximumV5Bundle
    provider_models: dict[str, dict[int, ProviderHourModel]]
    provider_oos_rows: dict[str, dict[int, list[dict[str, Any]]]]


def evaluate_station_maximum_v6(cfg: AppConfig) -> dict[str, Path]:

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
            "V6 requires NCEI pretraining and historical predictive features"
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
    station_audit: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        station_ncei = ncei[
            ncei["station_id"].astype(str).eq(station.station_id)
        ].copy()
        station_recent = recent[
            recent["station_id"].astype(str).eq(station.station_id)
        ].copy()
        if station_ncei.empty or station_recent.empty:
            continue
        base_model = _fit_station_base_model(station_ncei, station)
        v5_predictions = _chronological_station_predictions(
            station_recent,
            station,
            base_model,
            incumbent_by_key,
        )
        calibration_rows = _provider_calibration_rows(
            station_recent, station, base_model
        )
        station_predictions = _chronological_provider_mixture(
            v5_predictions,
            calibration_rows,
            station,
            base_model,
        )
        predictions.extend(station_predictions)
        provider_rows = [
            row
            for row in station_predictions
            if row["provider_point_c"] is not None
        ]
        mixture_rows = [
            row
            for row in station_predictions
            if float(row["provider_expert_weight"]) > 0.0
        ]
        station_audit.append(
            {
                "station_id": station.station_id,
                "providers": list(
                    STATION_PROVIDER_NAMES[station.station_id]
                ),
                "ncei_training_rows": base_model.training_rows,
                "calibration_rows": len(calibration_rows),
                "provider_oos_predictions": len(provider_rows),
                "mixture_predictions": len(mixture_rows),
                "mean_provider_expert_weight": _mean(
                    [
                        float(row["provider_expert_weight"])
                        for row in station_predictions
                    ]
                ),
                "v5_point_mae_c": _mean(
                    [float(row["v5_absolute_error_c"]) for row in predictions
                     if row["station_id"] == station.station_id]
                ),
                "provider_point_mae_c": _mean(
                    [
                        float(row["provider_absolute_error_c"])
                        for row in provider_rows
                    ]
                ),
                "v6_point_mae_c": _mean(
                    [
                        float(row["v6_absolute_error_c"])
                        for row in station_predictions
                    ]
                ),
            }
        )

    probability_scores = _probability_scores(predictions)
    station_scores = {
        station_id: _probability_scores(
            [
                row
                for row in predictions
                if row["station_id"] == station_id
            ]
        )
        for station_id in sorted({row["station_id"] for row in predictions})
    }
    point_improvements = _station_day_improvements(
        predictions, "v5_absolute_error_c", "v6_absolute_error_c"
    )
    probability_improvements = _station_day_improvements(
        predictions, "v5_brier", "v6_brier"
    )
    point_ci = _bootstrap_mean_ci(point_improvements)
    probability_ci = _bootstrap_mean_ci(probability_improvements)
    point_gate = bool(
        probability_scores["v6_point"]["mae_c"]
        < probability_scores["v5_point"]["mae_c"]
        and point_ci is not None
        and point_ci["low"] > 0.0
    )
    probability_gate = bool(
        probability_scores["v6"]["brier"]
        < probability_scores["v5"]["brier"]
        and probability_ci is not None
        and probability_ci["low"] > 0.0
    )
    candidate_result = _v6_candidate_replay(cfg, predictions)
    economic_scores = candidate_result["probability_scores"]
    economic_gate = bool(
        economic_scores["v4_candidate_probability"]["brier"] is not None
        and economic_scores["market_candidate_probability"]["brier"]
        is not None
        and economic_scores["v4_candidate_probability"]["brier"]
        < economic_scores["market_candidate_probability"]["brier"]
        and candidate_result["edge_monotonicity"][
            "nondecreasing_fraction"
        ]
        >= 0.5
        and candidate_result["trade_summary"]["trades"] > 0
        and candidate_result["trade_summary"]["synthetic_value_total"] > 0.0
    )
    timing_rows, timing_summary = _timing_analysis(predictions)

    generated_at_ns = time.time_ns()
    input_hash = _input_hash(cfg)
    run_id = (
        datetime.fromtimestamp(generated_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{generated_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    output_dir = (
        cfg.reports_dir / "station_maximum_v6" / "artifacts" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "chronological_predictions.csv"
    candidates_path = output_dir / "candidate_predictions.csv"
    trades_path = output_dir / "lower_bound_trades.csv"
    edge_path = output_dir / "edge_bins.csv"
    timing_path = output_dir / "temperature_market_lead.csv"
    summary_path = output_dir / "summary.json"
    _write_csv_atomic(predictions_path, predictions)
    _write_csv_atomic(
        candidates_path, candidate_result["candidate_predictions"]
    )
    _write_csv_atomic(trades_path, candidate_result["trades"])
    _write_csv_atomic(edge_path, candidate_result["edge_bins"])
    _write_csv_atomic(timing_path, timing_rows)
    summary = {
        "generated_at_utc": datetime.fromtimestamp(
            generated_at_ns / 1e9, tz=UTC
        ).isoformat(),
        "run_id": run_id,
        "strategy": STRATEGY_NAME,
        "input_sha256": input_hash,
        "alpha_demonstrated": False,
        "status": (
            "point_and_probability_gates_passed_economic_gate_passed"
            if point_gate and probability_gate and economic_gate
            else (
                "point_gate_passed_probability_or_economic_gate_failed"
                if point_gate
                else "point_gate_failed"
            )
        ),
        "method": {
            "physical_expert": (
                "The V5 airport-specific NCEI remaining-maximum model."
            ),
            "provider_expert": (
                "A station/hour ridge stack of individual provider deviations "
                "from the physical point, trained only on earlier resolved days."
            ),
            "mixture": (
                "A convex V5/provider weight estimated only from earlier "
                "out-of-sample provider predictions, with shrinkage toward V5."
            ),
            "timing": (
                "First scheduled point-in-time snapshot where V6 leads the "
                "market on the eventual bucket; not exact CLOB repricing time."
            ),
        },
        "parameters": {
            "provider_stack_ridge": PROVIDER_STACK_RIDGE,
            "expert_mixture_ridge": EXPERT_MIXTURE_RIDGE,
            "minimum_provider_training_days": (
                MINIMUM_PROVIDER_TRAINING_DAYS
            ),
            "minimum_expert_mixture_days": MINIMUM_EXPERT_MIXTURE_DAYS,
        },
        "chronological_predictions": len(predictions),
        "independent_station_days": len(
            {
                (row["station_id"], row["local_date"])
                for row in predictions
            }
        ),
        "station_audit": station_audit,
        "probability_scores": probability_scores,
        "probability_scores_by_station": station_scores,
        "v6_point_improvement_over_v5_station_day": {
            "mean_mae_c": _mean(point_improvements),
            "ci95": point_ci,
        },
        "v6_probability_improvement_over_v5_station_day": {
            "mean_brier": _mean(probability_improvements),
            "ci95": probability_ci,
        },
        "point_gate_passed": point_gate,
        "probability_gate_passed": probability_gate,
        "temperature_market_timing": timing_summary,
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
            "eligible_for_temperature_and_probability_shadow"
            if point_gate and probability_gate
            else (
                "eligible_for_temperature_only_shadow"
                if point_gate
                else "do_not_deploy"
            )
        ),
        "warning": (
            "All 2026 outcomes are development data. Timing uses scheduled "
            "snapshots and historical price proxies, never executable asks."
        ),
        "live_orders_enabled": False,
    }
    _write_json_atomic(summary_path, summary)
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(
        manifest_path, _artifact_manifest(output_dir, input_hash)
    )
    latest_path = cfg.reports_dir / "station_maximum_v6" / "latest.json"
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
        "candidates": candidates_path,
        "trades": trades_path,
        "edge_bins": edge_path,
        "timing": timing_path,
        "latest": latest_path,
    }


def deploy_station_maximum_v6_shadow(cfg: AppConfig) -> dict[str, Path]:

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    diagnostics = evaluate_station_maximum_v6(cfg)
    summary = json.loads(diagnostics["summary"].read_text(encoding="utf-8"))
    if not (
        summary["point_gate_passed"]
        and summary["probability_gate_passed"]
    ):
        raise RuntimeError(
            "V6 did not pass both meteorological gates; deployment refused"
        )
    v5_active_path = (
        cfg.reports_dir
        / "paper_shadow_station_maximum_v5"
        / "active_deployment.json"
    )
    if not v5_active_path.exists():
        raise FileNotFoundError(
            "V6 requires an active immutable V5 shadow deployment"
        )
    v5_active = json.loads(v5_active_path.read_text(encoding="utf-8"))
    v5_model_path = (
        Path(v5_active["deployment_directory"]) / "model_bundle.pkl"
    )
    if _sha256_file(v5_model_path) != v5_active["model_sha256"]:
        raise RuntimeError("Active V5 model hash mismatch")
    v5_bundle = pickle.loads(v5_model_path.read_bytes())
    if not isinstance(v5_bundle, StationMaximumV5Bundle):
        raise TypeError("Invalid V5 bundle for V6 deployment")
    input_hash = _input_hash(cfg)
    if v5_bundle.input_sha256 != input_hash:
        raise RuntimeError(
            "Active V5 bundle does not match current historical inputs"
        )

    recent = pd.read_csv(
        cfg.reports_dir / "historical" / "predictive_features.csv",
        low_memory=False,
    )
    evaluated = pd.read_csv(diagnostics["predictions"], low_memory=False)
    provider_models: dict[str, dict[int, ProviderHourModel]] = {}
    provider_oos_rows: dict[
        str, dict[int, list[dict[str, Any]]]
    ] = {}
    for station in cfg.stations.values():
        station_recent = recent[
            recent["station_id"].astype(str).eq(station.station_id)
        ].copy()
        calibration = _provider_calibration_rows(
            station_recent,
            station,
            v5_bundle.station_models[station.station_id],
        )
        models: dict[int, ProviderHourModel] = {}
        for local_hour in sorted(
            {int(float(row["local_hour"])) for row in calibration}
        ):
            hour_rows = [
                row
                for row in calibration
                if int(float(row["local_hour"])) == local_hour
            ]
            model = _fit_provider_stack(hour_rows)
            if model is not None:
                models[local_hour] = ProviderHourModel(
                    provider_names=STATION_PROVIDER_NAMES[
                        station.station_id
                    ],
                    model=model,
                )
        provider_models[station.station_id] = models
        station_evaluated = evaluated[
            evaluated["station_id"].astype(str).eq(station.station_id)
        ]
        by_hour: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in station_evaluated.to_dict(orient="records"):
            provider_point = _finite(row.get("provider_point_c"))
            if provider_point is None:
                continue
            local_hour = int(float(row["local_hour"]))
            local_date = str(row["local_date"])
            by_hour[local_hour].append(
                {
                    "local_date": local_date,
                    "day_of_year": date.fromisoformat(
                        local_date
                    ).timetuple().tm_yday,
                    "v5_point_c": float(row["v5_point_c"]),
                    "provider_point_c": provider_point,
                    "v6_point_c": float(row["v6_point_c"]),
                    "target_c": float(row["target_c"]),
                }
            )
        provider_oos_rows[station.station_id] = dict(by_hour)
    if any(
        not provider_models.get(station_id)
        for station_id in cfg.selected_station_ids
    ):
        raise RuntimeError("V6 is missing one or more provider models")

    deployed_at_ns = time.time_ns()
    trained_through = max(
        row["local_date"]
        for stations in provider_oos_rows.values()
        for rows in stations.values()
        for row in rows
    )
    bundle = StationMaximumV6Bundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        trained_through_local_date=trained_through,
        input_sha256=input_hash,
        v5_bundle=v5_bundle,
        provider_models=provider_models,
        provider_oos_rows=provider_oos_rows,
    )
    deployment_id = (
        datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        + f"{deployed_at_ns % 1_000_000_000:09d}Z-{input_hash[:12]}"
    )
    root = cfg.reports_dir / "paper_shadow_station_maximum_v6"
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
            "source_diagnostics": str(
                diagnostics["summary"].resolve()
            ),
            "source_diagnostics_sha256": _sha256_file(
                diagnostics["summary"]
            ),
            "point_gate_passed": True,
            "probability_gate_passed": True,
            "economic_gate_passed": False,
            "purpose": (
                "Prospective temperature/probability and timing comparison; "
                "never creates a paper or real position."
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


def run_station_maximum_v6_shadow_cycle(
    cfg: AppConfig, *, backfill_oos: bool = False
) -> dict[str, Path]:

    root = cfg.reports_dir / "paper_shadow_station_maximum_v6"
    active_path = root / "active_deployment.json"
    if not active_path.exists():
        raise FileNotFoundError(
            "Missing V6 deployment; run deploy-station-maximum-v6-shadow"
        )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    deployment_dir = Path(active["deployment_directory"])
    model_path = deployment_dir / "model_bundle.pkl"
    deployment_path = deployment_dir / "deployment.json"
    if _sha256_file(model_path) != active["model_sha256"]:
        raise RuntimeError("Active V6 model hash mismatch")
    if _sha256_file(deployment_path) != active["deployment_sha256"]:
        raise RuntimeError("Active V6 deployment hash mismatch")
    bundle = pickle.loads(model_path.read_bytes())
    if not isinstance(bundle, StationMaximumV6Bundle):
        raise TypeError("Invalid V6 shadow bundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise RuntimeError("V6 shadow schema mismatch")
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


    retryable_ids = {
        str(row.get("decision_id") or "")
        for row in predictions
        if row.get("rejection_reason") == "missing_resolution_running_max"
    }
    retryable_ids.discard("")
    if retryable_ids:
        predictions = [
            row
            for row in predictions
            if str(row.get("decision_id") or "") not in retryable_ids
        ]
        processed.difference_update(retryable_ids)
    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    deployed_at_ns = int(deployment["deployed_at_ns"])
    trained_through = date.fromisoformat(
        bundle.trained_through_local_date
    )
    cycle_started_ns = time.time_ns()
    source_rows = _load_csv(source_path)
    label_vocabulary = _market_label_vocabulary(source_rows)
    for source_row in source_rows:
        decision_id = str(source_row.get("decision_id") or "")
        station_id = str(source_row.get("station_id") or "")
        decision_mode = str(source_row.get("decision_mode") or "")
        timestamp_ns = _timestamp_ns(source_row.get("received_wall_ns"))
        local_date_text = str(source_row.get("local_date") or "")
        try:
            source_local_date = date.fromisoformat(local_date_text)
        except ValueError:
            source_local_date = None
        prospective = bool(
            timestamp_ns is not None and timestamp_ns > deployed_at_ns
        )
        predeployment_oos = bool(
            backfill_oos
            and timestamp_ns is not None
            and timestamp_ns <= deployed_at_ns
            and source_local_date is not None
            and source_local_date > trained_through
        )
        if (
            not decision_id
            or decision_id in processed
            or station_id not in stations
            or decision_mode not in ALLOWED_DECISION_MODES
            or timestamp_ns is None
            or not (prospective or predeployment_oos)
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
        result, rejection = _score_live_v6(
            merged, station, bundle, local_dt
        )
        row = {
            "decision_id": decision_id,
            "run_id": source_row.get("run_id"),
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
            "evaluation_phase": (
                "prospective"
                if prospective
                else "predeployment_out_of_sample_replay"
            ),
            "rejection_reason": rejection,
            "settled": False,
            "live_orders_enabled": False,
        }
        if result is not None:
            row.update(result)
        predictions.append(row)
    _settle_v6_predictions(cfg, predictions)
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
        and _finite(row.get("v6_brier")) is not None
    ]
    improvements = _station_day_improvements(
        settled, "v5_brier", "v6_brier"
    )
    _write_json_atomic(
        summary_path,
        {
            "strategy": SHADOW_STRATEGY_NAME,
            "deployment_id": deployment["deployment_id"],
            "trained_through_local_date": (
                bundle.trained_through_local_date
            ),
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
            "mean_v6_brier": _mean(
                [_finite(row.get("v6_brier")) for row in settled]
            ),
            "mean_v5_brier": _mean(
                [_finite(row.get("v5_brier")) for row in settled]
            ),
            "mean_station_day_brier_improvement": _mean(improvements),
            "brier_improvement_ci95": _bootstrap_mean_ci(improvements),
            "paper_trades": 0,
            "live_orders_enabled": False,
        },
    )
    paths = {
        "active": active_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "state": state_path,
    }
    from .station_maximum_v6_update_alpha import (
        refresh_station_maximum_v6_update_alpha,
    )

    paths.update(
        refresh_station_maximum_v6_update_alpha(
            cfg,
            deployment_dir=deployment_dir,
            predictions=predictions,
        )
    )
    return paths


def _score_live_v6(
    feature_row: dict[str, Any],
    station: StationConfig,
    bundle: StationMaximumV6Bundle,
    local_dt: datetime,
) -> tuple[dict[str, Any] | None, str]:
    station_id = station.station_id
    v5_result, rejection = _score_live_shadow(
        feature_row,
        station,
        bundle.v5_bundle.station_models[station_id],
        bundle.v5_bundle.recent_calibration[station_id],
        local_dt,
    )
    if v5_result is None:
        return None, rejection
    models = bundle.provider_models[station_id]
    nearest_hour = min(models, key=lambda hour: abs(hour - local_dt.hour))
    provider_values = _live_provider_values(feature_row, station_id)
    provider_point: float | None = None
    provider_weight = 0.0
    v5_point_c = _market_to_c(
        float(v5_result["v5_point_prediction_market_unit"]),
        station.market_unit,
    )
    v6_point_c = v5_point_c
    if len(provider_values) == len(STATION_PROVIDER_NAMES[station_id]):
        provider_point = _predict_provider_stack(
            models[nearest_hour].model,
            {
                "base_point_c": v5_result["base_point_c"],
                "running_max_c": v5_result["running_max_c"],
                "providers": provider_values,
            },
        )
        prior = bundle.provider_oos_rows[station_id].get(
            nearest_hour, []
        )
        provider_weight = _expert_mixture_weight(prior)
        v6_point_c = max(
            float(v5_result["running_max_c"]),
            v5_point_c
            + provider_weight * (provider_point - v5_point_c),
        )
    v5_distribution = _json_probabilities(
        v5_result["v5_distribution_json"]
    )
    market_distribution = _json_probabilities(
        v5_result["market_distribution_json"]
    )
    labels = sorted(set(v5_distribution) | set(market_distribution))
    prior_rows = bundle.provider_oos_rows[station_id].get(nearest_hour, [])
    if provider_weight <= 0.0 or not labels:
        v6_distribution = v5_distribution
        recent_residuals: list[float] = []
        ncei_residuals: list[float] = []
    else:
        day_of_year = local_dt.timetuple().tm_yday
        recent_residuals = _v6_recent_residuals(
            prior_rows, day_of_year
        )
        ncei_residuals = _seasonal_residuals(
            bundle.v5_bundle.station_models[station_id].holdout_residuals,
            nearest_hour,
            day_of_year,
        )
        residuals = [
            *ncei_residuals,
            *[
                residual
                for residual in recent_residuals
                for _ in range(RECENT_RESIDUAL_WEIGHT)
            ],
        ]
        v6_distribution = _distribution_from_residuals(
            v6_point_c,
            float(v5_result["running_max_c"]),
            residuals,
            labels,
            station.market_unit,
        )
    return (
        {
            **v5_result,
            "nearest_provider_hour": nearest_hour,
            "provider_values_json": json.dumps(provider_values),
            "provider_point_c": provider_point,
            "provider_spread_c": (
                max(provider_values) - min(provider_values)
                if provider_values
                else None
            ),
            "provider_expert_weight": provider_weight,
            "v5_point_c": v5_point_c,
            "v6_point_c": v6_point_c,
            "v6_point_prediction_market_unit": _c_to_market(
                v6_point_c, station.market_unit
            ),
            "v6_distribution_json": json.dumps(
                v6_distribution, sort_keys=True
            ),
            "recent_v6_residual_samples": len(recent_residuals),
            "v6_ncei_residual_samples": len(ncei_residuals),
        },
        "",
    )


def _settle_v6_predictions(
    cfg: AppConfig, predictions: list[dict[str, Any]]
) -> None:
    winners = {
        (
            str(row.get("station_id") or ""),
            str(row.get("event_date") or ""),
        ): str(row.get("official_winning_bucket") or "")
        for row in _load_csv(
            cfg.reports_dir / "resolutions" / "resolution_ledger.csv"
        )
    }
    for row in predictions:
        if _truth(row.get("settled")) or row.get("rejection_reason"):
            continue
        winner = winners.get(
            (str(row["station_id"]), str(row["local_date"]))
        )
        if not winner:
            continue
        v6 = _json_probabilities(row.get("v6_distribution_json"))
        v5 = _json_probabilities(row.get("v5_distribution_json"))
        market = _json_probabilities(row.get("market_distribution_json"))
        row.update(
            {
                "winning_bucket": winner,
                "v6_brier": _brier(v6, winner),
                "v5_brier": _brier(v5, winner),
                "market_brier": (
                    _brier(_normalize(market), winner)
                    if sum(market.values()) >= 0.95
                    else ""
                ),
                "settled": True,
            }
        )


def _live_provider_values(
    row: dict[str, Any], station_id: str
) -> list[float]:
    values: list[float] = []
    for provider in STATION_PROVIDER_NAMES[station_id]:
        value = next(
            (
                parsed
                for field in (
                    f"model_{provider}_remaining_max_c",
                    f"model_{provider}_today_max_c",
                )
                if (parsed := _finite(row.get(field))) is not None
            ),
            None,
        )
        if value is not None:
            values.append(value)
    return values


def _provider_calibration_rows(
    station_recent: Any,
    station: StationConfig,
    base_model: StationBaseModel,
) -> list[dict[str, Any]]:
    base_rows = _resolved_calibration_rows(
        station_recent, station, base_model
    )
    sources = {
        (
            str(row["local_date"]),
            int(float(row["local_hour"])),
        ): row
        for row in station_recent.to_dict(orient="records")
    }
    output: list[dict[str, Any]] = []
    for row in base_rows:
        key = (row["local_date"], int(float(row["local_hour"])))
        source = sources.get(key, {})
        providers = _historical_provider_values(source, station.station_id)
        if len(providers) != len(
            STATION_PROVIDER_NAMES[station.station_id]
        ):
            continue
        output.append(
            {
                **row,
                "providers": providers,
                "provider_spread_c": max(providers) - min(providers),
                "market_labels_json": source.get(
                    "market_labels_json", "[]"
                ),
            }
        )
    return output


def _chronological_provider_mixture(
    v5_predictions: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    station: StationConfig,
    base_model: StationBaseModel,
) -> list[dict[str, Any]]:
    calibration_by_hour: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        calibration_by_hour[int(float(row["local_hour"]))].append(row)
    for rows in calibration_by_hour.values():
        rows.sort(key=lambda row: row["local_date"])

    predictions_by_hour: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in v5_predictions:
        predictions_by_hour[int(float(row["local_hour"]))].append(row)
    output: list[dict[str, Any]] = []
    for local_hour, rows in predictions_by_hour.items():
        rows.sort(key=lambda row: row["local_date"])
        provider_oos: list[dict[str, Any]] = []
        for row in rows:
            local_date = str(row["local_date"])
            source = next(
                (
                    item
                    for item in calibration_by_hour[local_hour]
                    if item["local_date"] == local_date
                ),
                None,
            )
            if source is None:
                continue
            prior_calibration = [
                item
                for item in calibration_by_hour[local_hour]
                if item["local_date"] < local_date
            ]
            provider_model = _fit_provider_stack(prior_calibration)
            provider_point = (
                _predict_provider_stack(provider_model, source)
                if provider_model is not None
                else None
            )
            v5_point_c = _market_to_c(
                float(row["v5_point_prediction_market_unit"]),
                station.market_unit,
            )
            current_max_c = float(source["running_max_c"])
            provider_weight = (
                _expert_mixture_weight(provider_oos)
                if provider_point is not None
                else 0.0
            )
            v6_point_c = (
                v5_point_c
                if provider_point is None
                else v5_point_c
                + provider_weight * (provider_point - v5_point_c)
            )
            v6_point_c = max(current_max_c, v6_point_c)
            labels = _json_list(
                _source_value_for_prediction(
                    source, "market_labels_json", default="[]"
                )
            )
            v5_distribution = _json_probabilities(
                row["v5_distribution_json"]
            )
            if provider_weight <= 0.0 or not labels:
                v6_distribution = v5_distribution
                recent_residuals: list[float] = []
                ncei_residuals: list[float] = []
            else:
                day_of_year = date.fromisoformat(local_date).timetuple().tm_yday
                recent_residuals = _v6_recent_residuals(
                    provider_oos, day_of_year
                )
                ncei_residuals = _seasonal_residuals(
                    base_model.holdout_residuals,
                    float(local_hour),
                    float(day_of_year),
                )
                residuals = [
                    *ncei_residuals,
                    *[
                        residual
                        for residual in recent_residuals
                        for _ in range(RECENT_RESIDUAL_WEIGHT)
                    ],
                ]
                v6_distribution = _distribution_from_residuals(
                    v6_point_c,
                    current_max_c,
                    residuals,
                    labels,
                    station.market_unit,
                )
            target_c = float(source["target_c"])
            provider_error = (
                abs(provider_point - target_c)
                if provider_point is not None
                else None
            )
            winner = str(row["winning_bucket"])
            market_distribution = _json_probabilities(
                row["market_distribution_json"]
            )
            result = {
                **row,
                "v5_point_c": v5_point_c,
                "provider_point_c": provider_point,
                "provider_spread_c": source["provider_spread_c"],
                "provider_expert_weight": provider_weight,
                "v6_point_c": v6_point_c,
                "target_c": target_c,
                "v5_absolute_error_c": abs(v5_point_c - target_c),
                "provider_absolute_error_c": provider_error,
                "v6_absolute_error_c": abs(v6_point_c - target_c),
                "v6_point_prediction_market_unit": _c_to_market(
                    v6_point_c, station.market_unit
                ),
                "v6_distribution_json": json.dumps(
                    v6_distribution, sort_keys=True
                ),
                "v6_brier": _brier(v6_distribution, winner),
                "v5_brier": _brier(v5_distribution, winner),
                "market_brier": (
                    _brier(_normalize(market_distribution), winner)
                    if sum(market_distribution.values()) >= 0.95
                    else None
                ),
                "recent_v6_residual_samples": len(recent_residuals),
                "ncei_residual_samples": len(ncei_residuals),
            }
            output.append(result)
            if provider_point is not None:
                provider_oos.append(
                    {
                        "local_date": local_date,
                        "day_of_year": (
                            date.fromisoformat(local_date).timetuple().tm_yday
                        ),
                        "v5_point_c": v5_point_c,
                        "provider_point_c": provider_point,
                        "v6_point_c": v6_point_c,
                        "target_c": target_c,
                    }
                )
    return sorted(
        output,
        key=lambda row: (
            row["local_date"],
            row["station_id"],
            row["local_hour"],
        ),
    )


def _fit_provider_stack(
    rows: list[dict[str, Any]],
) -> Any | None:
    if len({row["local_date"] for row in rows}) < (
        MINIMUM_PROVIDER_TRAINING_DAYS
    ):
        return None
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    x = np.asarray(
        [
            [
                float(provider) - float(row["base_point_c"])
                for provider in row["providers"]
            ]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray(
        [
            float(row["target_c"]) - float(row["base_point_c"])
            for row in rows
        ],
        dtype=float,
    )
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=PROVIDER_STACK_RIDGE),
    )
    model.fit(x, y)
    return model


def _predict_provider_stack(model: Any, row: dict[str, Any]) -> float:
    base = float(row["base_point_c"])
    x = [
        [float(provider) - base for provider in row["providers"]]
    ]
    running_max = float(row.get("running_max_c", base))
    return max(running_max, base + float(model.predict(x)[0]))


def _expert_mixture_weight(rows: list[dict[str, Any]]) -> float:
    if len({row["local_date"] for row in rows}) < (
        MINIMUM_EXPERT_MIXTURE_DAYS
    ):
        return 0.0
    provider_delta = [
        float(row["provider_point_c"]) - float(row["v5_point_c"])
        for row in rows
    ]
    target_delta = [
        float(row["target_c"]) - float(row["v5_point_c"])
        for row in rows
    ]
    numerator = sum(
        provider * target
        for provider, target in zip(
            provider_delta, target_delta, strict=True
        )
    )
    denominator = (
        sum(value * value for value in provider_delta)
        + EXPERT_MIXTURE_RIDGE
    )
    return min(1.0, max(0.0, numerator / denominator))


def _v6_recent_residuals(
    rows: list[dict[str, Any]], day_of_year: int
) -> list[float]:
    selected = [
        row
        for row in rows
        if _circular_day_distance(
            float(row["day_of_year"]), float(day_of_year)
        )
        <= SEASONAL_RESIDUAL_WINDOW_DAYS
    ]
    if len(selected) < 8:
        selected = rows
    return [
        float(row["target_c"]) - float(row["v6_point_c"])
        for row in selected
    ]


def _v6_candidate_replay(
    cfg: AppConfig, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    proxy = [
        {
            **row,
            "v5_distribution_json": row["v6_distribution_json"],
        }
        for row in predictions
    ]
    result = _candidate_economic_replay(cfg, proxy)
    result["forced_entry_sensitivity"] = _forced_entry_sensitivity(
        result["candidate_predictions"]
    )
    return result


def _probability_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_field(field: str) -> dict[str, Any]:
        values = [
            value
            for row in rows
            if (value := _finite(row.get(field))) is not None
        ]
        return {"rows": len(values), "brier": _mean(values)}

    def point_field(field: str) -> dict[str, Any]:
        values = [
            value
            for row in rows
            if (value := _finite(row.get(field))) is not None
        ]
        return {"rows": len(values), "mae_c": _mean(values)}

    return {
        "v5": mean_field("v5_brier"),
        "v6": mean_field("v6_brier"),
        "market": mean_field("market_brier"),
        "v5_point": point_field("v5_absolute_error_c"),
        "provider_point": point_field("provider_absolute_error_c"),
        "v6_point": point_field("v6_absolute_error_c"),
    }


def _station_day_improvements(
    rows: list[dict[str, Any]], baseline: str, challenger: str
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        left = _finite(row.get(baseline))
        right = _finite(row.get(challenger))
        if left is None or right is None:
            continue
        grouped[(row["station_id"], row["local_date"])].append(left - right)
    return [_mean(values) or 0.0 for values in grouped.values()]


def _timing_analysis(
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["station_id"], row["local_date"])].append(row)
    events: list[dict[str, Any]] = []
    for (station_id, local_date), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["received_wall_ns"]))
        winner = _canonical_bucket(rows[0]["winning_bucket"])
        enriched: list[dict[str, Any]] = []
        for row in rows:
            model = _json_probabilities(row["v6_distribution_json"])
            market = _json_probabilities(row["market_distribution_json"])
            model_top, model_probability = _top(model)
            market_top, market_probability = _top(market)
            enriched.append(
                {
                    "timestamp_ns": int(row["received_wall_ns"]),
                    "local_hour": row["local_hour"],
                    "model_top": _canonical_bucket(model_top),
                    "model_top_probability": model_probability,
                    "model_correct": _canonical_bucket(model_top) == winner,
                    "model_top_market_probability": market.get(model_top),
                    "market_top": _canonical_bucket(market_top),
                    "market_top_probability": market_probability,
                    "market_correct": _canonical_bucket(market_top) == winner,
                }
            )
        first_model_correct = next(
            (
                row
                for row in enriched
                if row["model_correct"]
                and float(row["model_top_probability"] or 0.0) >= 0.20
            ),
            None,
        )
        first_market_correct = next(
            (row for row in enriched if row["market_correct"]),
            None,
        )
        lead_hours = (
            (
                first_market_correct["timestamp_ns"]
                - first_model_correct["timestamp_ns"]
            )
            / 3.6e12
            if first_model_correct is not None
            and first_market_correct is not None
            else None
        )
        signal = next(
            (
                row
                for row in enriched
                if row["model_top"]
                and row["model_top"] != row["market_top"]
                and float(row["model_top_probability"] or 0.0) >= 0.20
                and row["model_top_market_probability"] is not None
                and float(row["model_top_probability"])
                > float(row["model_top_market_probability"])
            ),
            None,
        )
        adoption = (
            next(
                (
                    row
                    for row in enriched
                    if row["timestamp_ns"] >= signal["timestamp_ns"]
                    and row["market_top"] == signal["model_top"]
                ),
                None,
            )
            if signal is not None
            else None
        )
        events.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "winning_bucket": winner,
                "first_v6_correct_hour": (
                    first_model_correct["local_hour"]
                    if first_model_correct is not None
                    else None
                ),
                "first_market_correct_hour": (
                    first_market_correct["local_hour"]
                    if first_market_correct is not None
                    else None
                ),
                "v6_lead_vs_market_hours": lead_hours,
                "v6_correct_before_market": (
                    lead_hours > 0 if lead_hours is not None else None
                ),
                "disagreement_signal": signal is not None,
                "disagreement_signal_correct": (
                    signal["model_correct"] if signal is not None else None
                ),
                "disagreement_signal_model_probability": (
                    signal["model_top_probability"]
                    if signal is not None
                    else None
                ),
                "disagreement_signal_market_probability": (
                    signal["model_top_market_probability"]
                    if signal is not None
                    else None
                ),
                "market_adoption_delay_hours": (
                    (
                        adoption["timestamp_ns"] - signal["timestamp_ns"]
                    )
                    / 3.6e12
                    if signal is not None and adoption is not None
                    else None
                ),
                "market_never_adopted_in_scheduled_window": (
                    signal is not None and adoption is None
                ),
                "scheduled_snapshots": len(enriched),
            }
        )
    paired_leads = [
        float(row["v6_lead_vs_market_hours"])
        for row in events
        if row["v6_lead_vs_market_hours"] is not None
    ]
    signals = [row for row in events if row["disagreement_signal"]]
    correct_signals = [
        row for row in signals if row["disagreement_signal_correct"]
    ]
    adoption_delays = [
        float(row["market_adoption_delay_hours"])
        for row in correct_signals
        if row["market_adoption_delay_hours"] is not None
    ]
    summary = {
        "station_days": len(events),
        "paired_first_correct_events": len(paired_leads),
        "v6_correct_before_market_events": sum(
            lead > 0 for lead in paired_leads
        ),
        "v6_correct_after_market_events": sum(
            lead < 0 for lead in paired_leads
        ),
        "v6_correct_tied_with_market_events": sum(
            lead == 0 for lead in paired_leads
        ),
        "median_v6_lead_vs_market_hours": (
            median(paired_leads) if paired_leads else None
        ),
        "nonoracle_disagreement_signals": len(signals),
        "correct_disagreement_signals": len(correct_signals),
        "disagreement_signal_accuracy": (
            len(correct_signals) / len(signals) if signals else None
        ),
        "correct_signals_with_market_adoption": len(adoption_delays),
        "median_market_adoption_delay_hours": (
            median(adoption_delays) if adoption_delays else None
        ),
        "warning": (
            "Three scheduled local-hour snapshots per day; lead is interval "
            "evidence, not the exact market repricing timestamp."
        ),
    }
    return events, summary


def _historical_provider_values(
    row: dict[str, Any], station_id: str
) -> list[float]:
    values: list[float] = []
    for provider in STATION_PROVIDER_NAMES[station_id]:
        value = _finite(
            row.get(
                f"historical_forecast_{provider}_day1_remaining_max_c"
            )
        )
        if value is not None:
            values.append(value)
    return values


def _source_value_for_prediction(
    row: dict[str, Any], name: str, *, default: Any = None
) -> Any:
    return row.get(name, default)


def _market_to_c(value: float, unit: str) -> float:
    return (value - 32.0) * 5.0 / 9.0 if unit == "F" else value


def _c_to_market(value: float, unit: str) -> float:
    return value * 9.0 / 5.0 + 32.0 if unit == "F" else value


def _top(probabilities: dict[str, float]) -> tuple[str, float | None]:
    if not probabilities:
        return "", None
    label, probability = max(
        probabilities.items(), key=lambda item: item[1]
    )
    return label, float(probability)
