from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig, PaperTradingConfig, StationConfig
from ..parsing.buckets import c_to_f
from .io import load_events_across_runs
from .master_model import (
    _calibration_residuals,
    conformal_final_distribution,
    forecast_anchored_conformal_distribution,
    remaining_max_distribution,
)
from .model import (
    NON_FEATURE_COLUMNS,
    _bootstrap_mean_ci,
    _brier,
    _fee_per_share,
    _label_for_value,
    _normalize,
    _read_market_quotes,
    _safe_float,
)
from .predictive import build_predictive_dataset

MODEL_SCHEMA_VERSION = 1
PERSISTENT_DISAGREEMENT_STRATEGY = "persistent_market_disagreement_v1"
PERSISTENT_DISAGREEMENT_RULES: dict[str, float | int | bool | str] = {
    "hypothesis_only": True,
    "minimum_market_mid": 0.15,
    "maximum_market_mid": 0.60,
    "maximum_executable_ask": 0.60,
    "minimum_weather_probability": 0.20,
    "minimum_weather_market_gap": 0.10,
    "minimum_provider_votes": 2,
    "maximum_ensemble_spread_c": 2.5,
    "edge_threshold": 0.08,
    "persistence_minutes": 30,
    "minimum_confirmations": 3,
}


@dataclass
class StationPaperModel:
    station_id: str
    market_unit: str
    feature_columns: list[str]
    imputer: Any
    rise_classifier: Any | None
    constant_rise_probability: float | None
    increment_regressor: Any | None
    fallback_positive_increments: list[float]
    calibration_history: list[dict[str, Any]]
    training_rows: int
    training_days: int
    trained_through_local_date: str


@dataclass
class PaperModelBundle:
    schema_version: int
    deployed_at_ns: int
    deployed_at_utc: str
    config_hash: str
    model_name: str
    stations: dict[str, StationPaperModel]
    training_files: list[str]
    validation_files: list[str]
    parameters: dict[str, Any]
    calibration_strategy: str = "legacy_oos_conformal"


def fit_paper_models(
    cfg: AppConfig,
    *,
    output_namespace: str = "paper_trading",
    calibration_strategy: str = "legacy_oos_conformal",
) -> dict[str, Path]:

    if calibration_strategy not in {
        "legacy_oos_conformal",
        "forecast_anchored_conformal_v2",
    }:
        raise ValueError(f"Unsupported calibration strategy: {calibration_strategy}")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies: pip install -e '.[analysis]'") from exc

    data, feature_files = _load_training_frame(cfg)
    calibration_by_station, validation_files = _load_calibration_history(cfg)
    station_models: dict[str, StationPaperModel] = {}
    metadata: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        frame = data[data["station_id"] == station.station_id].copy()
        if frame.empty:
            metadata.append(
                {"station_id": station.station_id, "status": "no_resolved_training_rows"}
            )
            continue
        frame = _add_canonical_features(frame, station)
        excluded = NON_FEATURE_COLUMNS | {
            "_target",
            "_current_max",
            "_remaining_increment",
            "_rise_again",
        }
        feature_columns: list[str] = []
        for column in frame.columns:
            if column in excluded:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() >= 4:
                frame[column] = converted
                feature_columns.append(column)
        if not feature_columns:
            metadata.append({"station_id": station.station_id, "status": "no_features"})
            continue

        imputer = SimpleImputer(strategy="median", keep_empty_features=False)
        x_train = imputer.fit_transform(frame[feature_columns])
        retained = list(imputer.get_feature_names_out(feature_columns))
        if not retained:
            metadata.append({"station_id": station.station_id, "status": "no_retained_features"})
            continue

        rows_per_day = frame.groupby("local_date")["local_date"].transform("count")
        weights = (1.0 / rows_per_day.astype(float)).to_numpy()
        latest_date = str(frame["local_date"].max())
        ages = (
            pd.Timestamp(latest_date) - pd.to_datetime(frame["local_date"], errors="coerce")
        ).dt.days.clip(lower=0)
        weights *= 0.5 ** (
            ages.to_numpy(dtype=float)
            / (365.25 * cfg.paper_trading.recency_half_life_years)
        )
        source_weights = (
            frame["target_source"]
            .astype(str)
            .str.lower()
            .eq("ncei_global_hourly")
            .map({True: cfg.paper_trading.ncei_source_weight, False: 1.0})
            .to_numpy(dtype=float)
        )
        weights *= source_weights

        rise = frame["_rise_again"].astype(int).to_numpy()
        increments = frame["_remaining_increment"].astype(float).to_numpy()
        classifier = None
        constant_probability: float | None = None
        if len(np.unique(rise)) == 1:
            constant_probability = float(rise[0])
        else:
            classifier = ExtraTreesClassifier(
                n_estimators=cfg.paper_trading.model_trees,
                min_samples_leaf=8,
                max_depth=20,
                max_features=0.8,
                n_jobs=1,
                random_state=142,
                class_weight="balanced",
            )
            classifier.fit(x_train, rise, sample_weight=weights)

        positive = increments > 1e-9
        positive_days = frame.loc[positive, "local_date"].nunique()
        regressor = None
        fallback = _one_increment_per_day(frame[positive])
        if positive.sum() >= 8 and positive_days >= 4:
            regressor = ExtraTreesRegressor(
                n_estimators=cfg.paper_trading.model_trees,
                min_samples_leaf=4,
                max_depth=20,
                max_features=0.8,
                n_jobs=1,
                random_state=143,
            )
            regressor.fit(
                x_train[positive],
                increments[positive],
                sample_weight=weights[positive],
            )
        if not fallback:
            fallback = [0.0]

        calibration = [
            item
            for item in calibration_by_station.get(station.station_id, [])
            if str(item["test_date"]) <= latest_date
        ]
        station_models[station.station_id] = StationPaperModel(
            station_id=station.station_id,
            market_unit=station.market_unit,
            feature_columns=retained,
            imputer=imputer,
            rise_classifier=classifier,
            constant_rise_probability=constant_probability,
            increment_regressor=regressor,
            fallback_positive_increments=fallback,
            calibration_history=calibration,
            training_rows=len(frame),
            training_days=int(frame["local_date"].nunique()),
            trained_through_local_date=latest_date,
        )
        calibration_days = len({str(item["test_date"]) for item in calibration})
        metadata.append(
            {
                "station_id": station.station_id,
                "status": "fitted",
                "training_rows": len(frame),
                "training_station_days": int(frame["local_date"].nunique()),
                "trained_through_local_date": latest_date,
                "feature_count": len(retained),
                "calibration_station_days": calibration_days,
                "probability_gate_ready": (
                    calibration_days >= cfg.paper_trading.minimum_calibration_days
                ),
                "rise_model": (
                    "extra_trees_classifier"
                    if classifier is not None
                    else f"constant_{constant_probability}"
                ),
                "increment_model": (
                    "extra_trees_regressor" if regressor is not None else "daily_empirical"
                ),
            }
        )

    missing = sorted(set(cfg.selected_station_ids) - set(station_models))
    if missing:
        raise RuntimeError(f"Paper models could not be fitted for: {', '.join(missing)}")

    deployed_at_ns = time.time_ns()
    bundle = PaperModelBundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        deployed_at_utc=datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC).isoformat(),
        config_hash=cfg.config_hash(),
        model_name=(
            "station_specific_two_stage_remaining_maximum_with_"
            + (
                "forecast_anchored_oos_conformal_calibration"
                if calibration_strategy == "forecast_anchored_conformal_v2"
                else "prior_oos_conformal_calibration"
            )
        ),
        stations=station_models,
        training_files=[str(path) for path in feature_files],
        validation_files=[str(path) for path in validation_files],
        parameters=cfg.paper_trading.model_dump(mode="json"),
        calibration_strategy=calibration_strategy,
    )
    output_dir = cfg.reports_dir / output_namespace
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "paper_model_bundle.pkl"
    _atomic_pickle(model_path, bundle)
    summary_path = output_dir / "paper_model_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "model": bundle.model_name,
                "calibration_strategy": bundle.calibration_strategy,
                "deployed_at_ns": deployed_at_ns,
                "deployed_at_utc": bundle.deployed_at_utc,
                "prospective_boundary": (
                    "Only feature rows received strictly after deployed_at_ns may open a paper trade."
                ),
                "config_hash": bundle.config_hash,
                "parameters": bundle.parameters,
                "stations": metadata,
                "training_files": bundle.training_files,
                "validation_files": bundle.validation_files,
                "live_orders_enabled": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model": model_path, "summary": summary_path}


def fit_calibration_challenger(cfg: AppConfig) -> dict[str, Path]:

    from .calibration_challenger import evaluate_calibration_challenger

    incumbent_path = cfg.reports_dir / "paper_trading" / "paper_model_bundle.pkl"
    if not incumbent_path.exists():
        raise FileNotFoundError(
            f"Missing {incumbent_path}; run fit-paper-model before the challenger"
        )
    incumbent = _load_bundle(incumbent_path)
    if incumbent.config_hash != cfg.config_hash():
        raise RuntimeError(
            "Configuration changed after the incumbent was frozen; "
            "a comparable challenger cannot be created"
        )
    calibration_by_station, validation_files = _load_calibration_history(cfg)
    validation_outputs = evaluate_calibration_challenger(cfg)
    validation_summary = json.loads(
        validation_outputs["summary"].read_text(encoding="utf-8")
    )
    enabled_stations = sorted(
        str(row["station_id"])
        for row in validation_summary.get("by_station", [])
        if _safe_float(
            (row.get("mean_brier_improvement_ci95") or {}).get("low")
        )
        is not None
        and float(row["mean_brier_improvement_ci95"]["low"]) > 0.0
    )
    stations = {
        station_id: replace(
            model,
            calibration_history=[
                item
                for item in calibration_by_station.get(station_id, [])
                if str(item["test_date"]) <= model.trained_through_local_date
            ],
        )
        for station_id, model in incumbent.stations.items()
    }
    deployed_at_ns = time.time_ns()
    bundle = PaperModelBundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        deployed_at_utc=datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC).isoformat(),
        config_hash=incumbent.config_hash,
        model_name=(
            "station_specific_two_stage_remaining_maximum_with_"
            "forecast_anchored_oos_conformal_calibration"
        ),
        stations=stations,
        training_files=list(incumbent.training_files),
        validation_files=[str(path) for path in validation_files],
        parameters={
            **dict(incumbent.parameters),
            "calibration_challenger_enabled_stations": enabled_stations,
            "calibration_challenger_station_gate": (
                "prior chronological Brier improvement CI95 lower bound > 0"
            ),
        },
        calibration_strategy="forecast_anchored_conformal_v2",
    )
    output_dir = cfg.reports_dir / "paper_trading_challenger"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "paper_model_bundle.pkl"
    _atomic_pickle(model_path, bundle)
    summary_path = output_dir / "paper_model_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "model": bundle.model_name,
                "calibration_strategy": bundle.calibration_strategy,
                "deployed_at_ns": bundle.deployed_at_ns,
                "deployed_at_utc": bundle.deployed_at_utc,
                "prospective_boundary": (
                    "Only feature rows received strictly after deployed_at_ns may open "
                    "challenger paper trades."
                ),
                "parent_incumbent_deployed_at_ns": incumbent.deployed_at_ns,
                "parent_incumbent_model": incumbent.model_name,
                "config_hash": bundle.config_hash,
                "parameters": bundle.parameters,
                "enabled_stations": enabled_stations,
                "station_gate": (
                    "Enabled only when chronological station-day Brier improvement "
                    "has a positive 95% bootstrap lower bound."
                ),
                "validation_summary": str(validation_outputs["summary"]),
                "stations": [
                    {
                        "station_id": station_id,
                        "training_rows": model.training_rows,
                        "training_station_days": model.training_days,
                        "trained_through_local_date": model.trained_through_local_date,
                        "calibration_station_days": len(
                            {
                                str(item["test_date"])
                                for item in model.calibration_history
                            }
                        ),
                    }
                    for station_id, model in sorted(bundle.stations.items())
                ],
                "training_files": bundle.training_files,
                "validation_files": bundle.validation_files,
                "live_orders_enabled": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model": model_path, "summary": summary_path}


def fit_persistent_disagreement_challenger(cfg: AppConfig) -> dict[str, Path]:

    parent_path = (
        cfg.reports_dir / "paper_trading_challenger" / "paper_model_bundle.pkl"
    )
    if not parent_path.exists():
        raise FileNotFoundError(
            f"Missing {parent_path}; run fit-paper-challenger first"
        )
    parent = _load_bundle(parent_path)
    if parent.config_hash != cfg.config_hash():
        raise RuntimeError(
            "Configuration changed after the calibration challenger was frozen; "
            "a comparable disagreement experiment cannot be created"
        )
    deployed_at_ns = time.time_ns()
    rules = dict(PERSISTENT_DISAGREEMENT_RULES)
    bundle = PaperModelBundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        deployed_at_utc=datetime.fromtimestamp(
            deployed_at_ns / 1e9, tz=UTC
        ).isoformat(),
        config_hash=parent.config_hash,
        model_name=(
            "station_specific_persistent_midprice_market_disagreement_"
            "hypothesis"
        ),
        stations=dict(parent.stations),
        training_files=list(parent.training_files),
        validation_files=list(parent.validation_files),
        parameters={
            **dict(parent.parameters),
            "persistent_disagreement_rules": rules,
            "underlying_calibration_strategy": parent.calibration_strategy,
            "hypothesis_only": True,
        },
        calibration_strategy=PERSISTENT_DISAGREEMENT_STRATEGY,
    )
    output_dir = cfg.reports_dir / "paper_trading_persistent_disagreement"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "paper_model_bundle.pkl"
    _atomic_pickle(model_path, bundle)
    summary_path = output_dir / "paper_model_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "model": bundle.model_name,
                "calibration_strategy": bundle.calibration_strategy,
                "deployed_at_ns": bundle.deployed_at_ns,
                "deployed_at_utc": bundle.deployed_at_utc,
                "prospective_boundary": (
                    "Only feature rows received strictly after deployed_at_ns may "
                    "open hypothesis-only paper trades."
                ),
                "parent_model": parent.model_name,
                "parent_deployed_at_ns": parent.deployed_at_ns,
                "config_hash": bundle.config_hash,
                "rules": rules,
                "outcome_blind_signal": True,
                "hypothesis_only": True,
                "maximum_entries_per_station_day": 1,
                "live_orders_enabled": False,
                "warnings": [
                    "Winning buckets are never used to create or persist a signal.",
                    "Weather probabilities are not claimed to beat the market.",
                    "Every entry is a shadow paper test of persistent disagreement.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model": model_path, "summary": summary_path}


def run_paper_cycle(
    cfg: AppConfig,
    run_id: str | None = None,
    *,
    rebuild_dataset: bool = True,
    output_namespace: str = "paper_trading",
    audit_only: bool = False,
) -> dict[str, Path]:

    if not cfg.paper_trading.enabled:
        raise RuntimeError("paper_trading.enabled is false")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies: pip install -e '.[analysis]'") from exc

    output_dir = cfg.reports_dir / output_namespace
    model_path = output_dir / "paper_model_bundle.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}; run fit-paper-model first")
    bundle = _load_bundle(model_path)
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise RuntimeError("Paper model schema is incompatible; refit the model")
    if bundle.config_hash != cfg.config_hash():
        raise RuntimeError(
            "Configuration changed after the prospective model was frozen; "
            "fit-paper-model is required to start a new declared experiment"
        )
    if rebuild_dataset:
        paths = build_predictive_dataset(cfg, run_id)
        feature_path = paths["features"]
    else:
        feature_path = _resolve_feature_path(cfg, run_id)
    frame = pd.read_csv(feature_path)
    if not frame.empty:
        frame["received_wall_ns"] = pd.to_numeric(
            frame["received_wall_ns"], errors="coerce"
        )
        frame = frame.dropna(subset=["station_id", "local_date", "received_wall_ns"])
        frame = frame[frame["received_wall_ns"] > bundle.deployed_at_ns]


        priority = {"forecast_update": 0, "scheduled": 1, "event_driven": 2}
        frame["_mode_priority"] = (
            frame.get("decision_mode", "event_driven")
            .astype(str)
            .map(priority)
            .fillna(3)
        )
        frame = (
            frame.sort_values(["received_wall_ns", "_mode_priority"])
            .drop_duplicates(["station_id", "local_date", "received_wall_ns"], keep="first")
            .drop(columns=["_mode_priority"])
        )
        station_frames = []
        for station in cfg.stations.values():
            subset = frame[frame["station_id"] == station.station_id].copy()
            if not subset.empty:
                station_frames.append(_add_canonical_features(subset, station))
        frame = pd.concat(station_frames, ignore_index=True) if station_frames else frame.iloc[0:0]

    audit_path = output_dir / "paper_signal_audit.csv"
    trades_path = output_dir / "paper_trades.csv"
    state_path = output_dir / "paper_state.json"
    summary_path = output_dir / "paper_summary.json"
    audit_rows = _read_csv(audit_path)
    trade_rows = _read_csv(trades_path)
    processed_ids = {str(row.get("decision_id") or "") for row in audit_rows}
    traded_days = {
        (str(row.get("station_id")), str(row.get("local_date"))) for row in trade_rows
    }
    state = _read_json(state_path, {"candidates": {}})
    candidates = state.setdefault("candidates", {})
    station_buffers = _station_execution_buffers(cfg)

    new_rows = 0
    new_trades = 0
    for record in frame.to_dict(orient="records"):
        station_id = str(record["station_id"])
        station = next(
            item for item in cfg.stations.values() if item.station_id == station_id
        )
        decision_id = _decision_id(record)
        if decision_id in processed_ids:
            continue
        evaluation = _evaluate_record(
            record,
            station,
            bundle.stations[station_id],
            cfg.paper_trading,
            station_buffers.get(station_id, cfg.paper_trading.default_execution_buffer),
            calibration_strategy=_bundle_station_strategy(bundle, station_id),
            strategy_parameters=_bundle_station_strategy_parameters(
                bundle, station_id
            ),
        )
        timestamp_ns = int(float(record["received_wall_ns"]))
        day_key = (station_id, str(record["local_date"]))
        persistence_minutes, minimum_confirmations = _bundle_persistence_rules(
            bundle, cfg.paper_trading
        )
        persistence = advance_persistence(
            candidates,
            station_id=station_id,
            local_date=str(record["local_date"]),
            timestamp_ns=timestamp_ns,
            bucket=evaluation.get("candidate_bucket"),
            eligible=not evaluation["rejection_reasons"],
            minimum_confirmations=minimum_confirmations,
            persistence_minutes=persistence_minutes,
        )
        audit = {
            "decision_id": decision_id,
            "run_id": record.get("run_id"),
            "station_id": station_id,
            "city": station.city,
            "local_date": record["local_date"],
            "received_wall_ns": timestamp_ns,
            "received_utc": datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC).isoformat(),
            "decision_mode": record.get("decision_mode"),
            "trigger_source": record.get("trigger_source"),
            "model_deployed_at_ns": bundle.deployed_at_ns,
            **evaluation,
            **persistence,
            "paper_trade_opened": False,
            "winning_bucket": "",
            "weather_brier": "",
            "market_brier": "",
            "brier_improvement": "",
        }
        can_open = bool(
            not audit_only
            and persistence["persistent"]
            and day_key not in traded_days
            and len(
                [
                    row
                    for row in trade_rows
                    if str(row.get("station_id")) == station_id
                    and str(row.get("local_date")) == str(record["local_date"])
                ]
            )
            < cfg.paper_trading.maximum_entries_per_station_day
        )
        if can_open:
            audit["paper_trade_opened"] = True
            trade = {
                "paper_trade_id": hashlib.sha256(
                    f"paper|{decision_id}".encode("utf-8")
                ).hexdigest()[:24],
                "decision_id": decision_id,
                "strategy_name": evaluation.get("strategy_name"),
                "hypothesis_only": evaluation.get("hypothesis_only", False),
                "station_id": station_id,
                "city": station.city,
                "local_date": record["local_date"],
                "entry_received_wall_ns": timestamp_ns,
                "entry_received_utc": audit["received_utc"],
                "bucket": evaluation["candidate_bucket"],
                "model_probability": evaluation["candidate_model_probability"],
                "market_mid": evaluation["candidate_market_mid"],
                "executable_ask": evaluation["candidate_ask"],
                "best_ask_size": evaluation["candidate_ask_size"],
                "requested_shares": cfg.paper_trading.requested_shares,
                "fee_per_share": evaluation["candidate_fee_per_share"],
                "additional_slippage_buffer": evaluation[
                    "candidate_additional_slippage_buffer"
                ],
                "conservative_cost_per_share": evaluation[
                    "candidate_conservative_cost"
                ],
                "estimated_edge_after_costs": evaluation["candidate_edge_after_costs"],
                "market_probability": evaluation.get(
                    "candidate_market_probability"
                ),
                "weather_market_gap": evaluation.get(
                    "candidate_weather_market_gap"
                ),
                "supporting_provider_votes": evaluation.get(
                    "candidate_provider_votes"
                ),
                "confirmations": persistence["confirmations"],
                "persistence_seconds": persistence["persistence_seconds"],
                "weather_distribution_json": evaluation[
                    "weather_distribution_json"
                ],
                "market_distribution_json": evaluation["market_distribution_json"],
                "status": "paper_open",
                "winning_bucket": "",
                "won": "",
                "realized_pnl_per_share": "",
                "realized_pnl_total": "",
                "live_order_sent": False,
            }
            trade_rows.append(trade)
            traded_days.add(day_key)
            new_trades += 1
        audit_rows.append(audit)
        processed_ids.add(decision_id)
        new_rows += 1

    final_maxima = _final_wu_maxima(cfg)
    _settle_rows(audit_rows, trade_rows, final_maxima, cfg)
    _write_csv_atomic(audit_path, audit_rows)
    _write_csv_atomic(trades_path, trade_rows)
    state.update(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_deployed_at_ns": bundle.deployed_at_ns,
            "last_cycle_at_utc": datetime.now(tz=UTC).isoformat(),
            "last_feature_file": str(feature_path),
            "last_new_audit_rows": new_rows,
            "last_new_paper_trades": new_trades,
            "audit_only": audit_only,
        }
    )
    _write_json_atomic(state_path, state)
    summary = _paper_summary(
        audit_rows,
        trade_rows,
        bundle,
        cfg.paper_trading,
        station_buffers,
    )
    _write_json_atomic(summary_path, summary)
    return {
        "audit": audit_path,
        "trades": trades_path,
        "state": state_path,
        "summary": summary_path,
        "features": feature_path,
    }


def advance_persistence(
    candidates: dict[str, Any],
    *,
    station_id: str,
    local_date: str,
    timestamp_ns: int,
    bucket: str | None,
    eligible: bool,
    minimum_confirmations: int,
    persistence_minutes: int,
) -> dict[str, Any]:

    key = f"{station_id}|{local_date}"
    if not eligible or not bucket:
        candidates.pop(key, None)
        return {
            "confirmations": 0,
            "persistence_seconds": 0.0,
            "persistent": False,
        }
    previous = candidates.get(key)
    if (
        previous
        and previous.get("bucket") == bucket
        and timestamp_ns > int(previous.get("last_seen_ns") or 0)
    ):
        first_seen_ns = int(previous["first_seen_ns"])
        confirmations = int(previous["confirmations"]) + 1
    elif previous and previous.get("bucket") == bucket:
        first_seen_ns = int(previous["first_seen_ns"])
        confirmations = int(previous["confirmations"])
    else:
        first_seen_ns = timestamp_ns
        confirmations = 1
    persistence_seconds = max(0.0, (timestamp_ns - first_seen_ns) / 1e9)
    candidates[key] = {
        "bucket": bucket,
        "first_seen_ns": first_seen_ns,
        "last_seen_ns": timestamp_ns,
        "confirmations": confirmations,
    }
    return {
        "confirmations": confirmations,
        "persistence_seconds": persistence_seconds,
        "persistent": bool(
            confirmations >= minimum_confirmations
            and persistence_seconds >= persistence_minutes * 60
        ),
    }


def _bundle_station_strategy(
    bundle: PaperModelBundle, station_id: str
) -> str:
    strategy = getattr(
        bundle, "calibration_strategy", "legacy_oos_conformal"
    )
    if strategy == PERSISTENT_DISAGREEMENT_STRATEGY:
        return strategy
    enabled = set(
        bundle.parameters.get(
            "calibration_challenger_enabled_stations", bundle.stations
        )
    )
    return strategy if station_id in enabled else "legacy_oos_conformal"


def _bundle_station_strategy_parameters(
    bundle: PaperModelBundle, station_id: str
) -> dict[str, Any]:
    if bundle.calibration_strategy != PERSISTENT_DISAGREEMENT_STRATEGY:
        return {}
    enabled = set(
        bundle.parameters.get(
            "calibration_challenger_enabled_stations", bundle.stations
        )
    )
    return {
        **dict(
            bundle.parameters.get("persistent_disagreement_rules")
            or PERSISTENT_DISAGREEMENT_RULES
        ),
        "weather_calibration_strategy": (
            bundle.parameters.get(
                "underlying_calibration_strategy",
                "forecast_anchored_conformal_v2",
            )
            if station_id in enabled
            else "legacy_oos_conformal"
        ),
    }


def _bundle_persistence_rules(
    bundle: PaperModelBundle, rules: PaperTradingConfig
) -> tuple[int, int]:
    if bundle.calibration_strategy != PERSISTENT_DISAGREEMENT_STRATEGY:
        return rules.persistence_minutes, rules.minimum_confirmations
    strategy = dict(
        bundle.parameters.get("persistent_disagreement_rules")
        or PERSISTENT_DISAGREEMENT_RULES
    )
    return (
        int(strategy["persistence_minutes"]),
        int(strategy["minimum_confirmations"]),
    )


def _evaluate_record(
    row: dict[str, Any],
    station: StationConfig,
    model: StationPaperModel,
    rules: PaperTradingConfig,
    station_p90_cost_from_mid: float,
    *,
    calibration_strategy: str = "legacy_oos_conformal",
    strategy_parameters: dict[str, Any] | None = None,
    weather_distribution_override: dict[str, float] | None = None,
    calibration_days_override: int | None = None,
) -> dict[str, Any]:
    strategy_parameters = dict(strategy_parameters or {})
    persistent_disagreement = (
        calibration_strategy == PERSISTENT_DISAGREEMENT_STRATEGY
    )
    weather_calibration_strategy = (
        str(
            strategy_parameters.get(
                "weather_calibration_strategy",
                "forecast_anchored_conformal_v2",
            )
        )
        if persistent_disagreement
        else calibration_strategy
    )
    reasons: list[str] = []
    if not _truth(row.get("day_history_complete")):
        reasons.append("incomplete_day_history")
    if not _truth(row.get("continuity_eligible")):
        reasons.append("continuity_warmup_or_gap")
    if not _truth(row.get("market_ws_healthy")):
        reasons.append("market_websocket_unhealthy")
    quote_coverage = _safe_float(row.get("market_quote_coverage")) or 0.0
    observation_age = _safe_float(row.get("minutes_since_last_observation"))
    if observation_age is None or observation_age > rules.maximum_observation_age_minutes:
        reasons.append("stale_primary_observation")

    provider_ages = [
        value
        for provider in station.forecast_providers
        if (
            value := _safe_float(row.get(f"model_{provider}_age_minutes"))
        )
        is not None
    ]
    provider_count = int(_safe_float(row.get("model_provider_count")) or len(provider_ages))
    if provider_count < rules.minimum_forecast_providers:
        reasons.append("insufficient_forecast_providers")
    if (
        len(provider_ages) < rules.minimum_forecast_providers
        or max(provider_ages, default=math.inf) > rules.maximum_forecast_age_minutes
    ):
        reasons.append("stale_forecast_consensus")
    forecast_spread = _safe_float(row.get("model_spread_remaining_max_c"))
    if forecast_spread is None:
        forecast_spread = _safe_float(row.get("model_spread_today_max_c"))
    if forecast_spread is None or forecast_spread > rules.maximum_forecast_spread_c:
        reasons.append("forecast_consensus_too_wide")
    peer_coverage = _safe_float(row.get("peer_station_coverage"))
    if peer_coverage is None or peer_coverage < rules.minimum_peer_coverage:
        reasons.append("insufficient_peer_station_coverage")

    (
        labels,
        market_mid_raw,
        market_asks,
        fee_rates,
        fee_exponents,
        fees_enabled,
    ) = _read_market_quotes(row)
    ask_sizes = _json_floats(row.get("market_ask_size_json"))
    market_mid_mass = sum(max(0.0, value) for value in market_mid_raw.values())
    market_probability_scoring_complete = bool(
        market_mid_mass >= rules.minimum_market_mid_mass
    )
    market_probs = (
        _normalize(market_mid_raw) if market_probability_scoring_complete else {}
    )
    current_max = _safe_float(row.get("resolution_running_max_market_unit"))
    if current_max is None:
        reasons.append("missing_resolution_provider_running_max")
    local_hour = _safe_float(row.get("local_hour"))
    mode = str(row.get("decision_mode") or "event_driven")
    rise_residuals, increment_residuals, final_residuals, calibration_days = (
        _calibration_residuals(model.calibration_history, local_hour, mode)
    )
    if calibration_days_override is not None:
        calibration_days = int(calibration_days_override)
    probability_calibrated = bool(
        calibration_days >= rules.minimum_calibration_days
        and (
            weather_distribution_override is not None
            or len(final_residuals) >= rules.minimum_calibration_days
        )
    )
    if not probability_calibrated:
        reasons.append("insufficient_prior_oos_calibration_days")

    weather_probs: dict[str, float] = {}
    calibration_diagnostics: dict[str, Any] = {
        "calibration_strategy": calibration_strategy,
        "weather_calibration_strategy": weather_calibration_strategy,
        "calibration_challenger_accepted": False,
    }
    raw_rise_probability: float | None = None
    positive_point: float | None = None
    if current_max is not None and labels:
        if weather_distribution_override is not None:
            weather_probs = _normalize(
                {
                    str(label): float(probability)
                    for label, probability in weather_distribution_override.items()
                    if _safe_float(probability) is not None
                }
            )
            calibration_diagnostics["distribution_source"] = (
                "chronological_oos_override"
            )
        else:
            raw_rise_probability, positive_point, tree_samples = (
                _predict_remaining_maximum(row, model)
            )
            raw_point_final = (
                current_max + raw_rise_probability * positive_point
            )
            legacy_weather_probs = (
                conformal_final_distribution(
                    current_max,
                    raw_point_final,
                    final_residuals,
                    labels,
                    station.market_unit,
                )
                if probability_calibrated
                else remaining_max_distribution(
                    current_max,
                    raw_rise_probability,
                    tree_samples
                    or increment_residuals
                    or model.fallback_positive_increments,
                    labels,
                    station.market_unit,
                )
            )
            weather_probs = legacy_weather_probs
            if weather_calibration_strategy == "forecast_anchored_conformal_v2":
                forecast_point = _safe_float(
                    row.get("canonical_forecast_today_max_market_unit")
                )
                challenger_probs, challenger_diagnostics = (
                    forecast_anchored_conformal_distribution(
                        current_max,
                        raw_point_final,
                        forecast_point,
                        model.calibration_history,
                        labels,
                        station.market_unit,
                        local_hour=local_hour,
                        decision_mode=mode,
                    )
                )
                calibration_diagnostics = {
                    "calibration_strategy": calibration_strategy,
                    "weather_calibration_strategy": (
                        weather_calibration_strategy
                    ),
                    "calibration_challenger_accepted": (
                        challenger_diagnostics.get(
                            "challenger_accepted", False
                        )
                    ),
                    "calibration_model_weight": challenger_diagnostics.get(
                        "model_weight"
                    ),
                    "calibration_raw_model_weight": (
                        challenger_diagnostics.get("raw_model_weight")
                    ),
                    "calibration_forecast_point": challenger_diagnostics.get(
                        "forecast_point_final"
                    ),
                    "calibration_blended_point": challenger_diagnostics.get(
                        "blended_point_final"
                    ),
                    "calibration_raw_rmse": challenger_diagnostics.get(
                        "raw_rmse"
                    ),
                    "calibration_blended_rmse": challenger_diagnostics.get(
                        "blended_rmse"
                    ),
                    "calibration_relative_rmse_improvement": (
                        challenger_diagnostics.get(
                            "relative_rmse_improvement"
                        )
                    ),
                }
                if (
                    challenger_diagnostics.get("challenger_accepted")
                    and challenger_probs
                ):
                    weather_probs = challenger_probs
    if not labels:
        reasons.append("missing_bucket_vocabulary")
    if not weather_probs:
        reasons.append("missing_weather_probability_distribution")
    if persistent_disagreement:
        if not market_probability_scoring_complete:
            reasons.append("incomplete_market_probability_state")
        strategy_spread = float(
            strategy_parameters.get(
                "maximum_ensemble_spread_c",
                PERSISTENT_DISAGREEMENT_RULES[
                    "maximum_ensemble_spread_c"
                ],
            )
        )
        if forecast_spread is None or forecast_spread > strategy_spread:
            reasons.append("disagreement_ensemble_spread_too_wide")

    candidates: list[dict[str, float | str]] = []
    provider_votes, consensus_bucket = _forecast_bucket_support(
        row, station, labels
    )
    for label, probability in weather_probs.items():
        ask = market_asks.get(label)


        mid = market_mid_raw.get(label)
        size = ask_sizes.get(label)
        if ask is None or not (rules.minimum_market_ask <= ask <= rules.maximum_market_ask):
            continue
        if probability < rules.minimum_model_probability:
            continue
        if label not in fees_enabled:
            continue
        fee = _fee_per_share(
            ask,
            fee_rates.get(label),
            fee_exponents.get(label),
            fees_enabled.get(label, 0.0) > 0,
        )
        if not math.isfinite(fee):
            continue
        observed_half_spread = max(0.0, ask - mid) if mid is not None else 0.0
        additional_buffer = max(
            0.0,
            station_p90_cost_from_mid - observed_half_spread,
        )
        conservative_cost = min(1.0, ask + fee + additional_buffer)
        market_probability = (
            market_probs.get(label)
            if market_probability_scoring_complete
            else mid
        )
        weather_market_gap = (
            probability - market_probability
            if market_probability is not None
            else math.nan
        )
        votes = provider_votes.get(label, 0)
        if persistent_disagreement:
            minimum_mid = float(
                strategy_parameters.get(
                    "minimum_market_mid",
                    PERSISTENT_DISAGREEMENT_RULES["minimum_market_mid"],
                )
            )
            maximum_mid = float(
                strategy_parameters.get(
                    "maximum_market_mid",
                    PERSISTENT_DISAGREEMENT_RULES["maximum_market_mid"],
                )
            )
            maximum_ask = float(
                strategy_parameters.get(
                    "maximum_executable_ask",
                    PERSISTENT_DISAGREEMENT_RULES[
                        "maximum_executable_ask"
                    ],
                )
            )
            minimum_probability = float(
                strategy_parameters.get(
                    "minimum_weather_probability",
                    PERSISTENT_DISAGREEMENT_RULES[
                        "minimum_weather_probability"
                    ],
                )
            )
            minimum_gap = float(
                strategy_parameters.get(
                    "minimum_weather_market_gap",
                    PERSISTENT_DISAGREEMENT_RULES[
                        "minimum_weather_market_gap"
                    ],
                )
            )
            minimum_votes = int(
                strategy_parameters.get(
                    "minimum_provider_votes",
                    PERSISTENT_DISAGREEMENT_RULES["minimum_provider_votes"],
                )
            )
            if (
                mid is None
                or not (minimum_mid <= mid <= maximum_mid)
                or ask > maximum_ask
                or probability < minimum_probability
                or not math.isfinite(weather_market_gap)
                or weather_market_gap < minimum_gap
                or votes < minimum_votes
                or consensus_bucket != label
            ):
                continue
        candidates.append(
            {
                "label": label,
                "probability": probability,
                "mid": mid if mid is not None else math.nan,
                "ask": ask,
                "size": size if size is not None else 0.0,
                "fee": fee,
                "buffer": additional_buffer,
                "cost": conservative_cost,
                "edge": probability - conservative_cost,
                "market_probability": (
                    market_probability
                    if market_probability is not None
                    else math.nan
                ),
                "weather_market_gap": weather_market_gap,
                "provider_votes": float(votes),
            }
        )
    selected = max(candidates, key=lambda item: float(item["edge"])) if candidates else None
    if selected is None:
        reasons.append(
            "no_persistent_midprice_disagreement"
            if persistent_disagreement
            else "no_costed_executable_candidate"
        )
    else:
        if float(selected["size"]) < rules.requested_shares:
            reasons.append("insufficient_best_ask_depth")
        edge_threshold = (
            float(
                strategy_parameters.get(
                    "edge_threshold",
                    PERSISTENT_DISAGREEMENT_RULES["edge_threshold"],
                )
            )
            if persistent_disagreement
            else rules.edge_threshold
        )
        if float(selected["edge"]) < edge_threshold:
            reasons.append("edge_below_threshold_after_costs")

    reasons = sorted(set(reasons))
    return {
        "model_trained_through_local_date": model.trained_through_local_date,
        "probability_calibrated": probability_calibrated,
        "calibration_station_days": calibration_days,
        "raw_probability_rise_again": raw_rise_probability,
        "conditional_positive_increment": positive_point,
        "current_resolution_running_max": current_max,
        "forecast_provider_count": provider_count,
        "forecast_max_age_minutes": max(provider_ages) if provider_ages else None,
        "forecast_spread_c": forecast_spread,
        "peer_station_coverage": peer_coverage,
        "market_quote_coverage": quote_coverage,
        "market_mid_mass": market_mid_mass,
        "market_probability_scoring_complete": market_probability_scoring_complete,
        "strategy_name": calibration_strategy,
        "hypothesis_only": bool(persistent_disagreement),
        "consensus_forecast_bucket": consensus_bucket,
        "weather_distribution_json": json.dumps(weather_probs, sort_keys=True),
        "market_distribution_json": json.dumps(market_probs, sort_keys=True),
        **calibration_diagnostics,
        "candidate_bucket": selected["label"] if selected else None,
        "candidate_model_probability": selected["probability"] if selected else None,
        "candidate_market_mid": selected["mid"] if selected else None,
        "candidate_ask": selected["ask"] if selected else None,
        "candidate_ask_size": selected["size"] if selected else None,
        "candidate_fee_per_share": selected["fee"] if selected else None,
        "candidate_additional_slippage_buffer": selected["buffer"] if selected else None,
        "candidate_conservative_cost": selected["cost"] if selected else None,
        "candidate_edge_after_costs": selected["edge"] if selected else None,
        "candidate_market_probability": (
            selected["market_probability"] if selected else None
        ),
        "candidate_weather_market_gap": (
            selected["weather_market_gap"] if selected else None
        ),
        "candidate_provider_votes": (
            int(float(selected["provider_votes"])) if selected else None
        ),
        "rejection_reasons": "|".join(reasons),
        "base_signal_eligible": not reasons,
    }


def _forecast_bucket_support(
    row: dict[str, Any], station: StationConfig, labels: list[str]
) -> tuple[Counter[str], str | None]:
    votes: Counter[str] = Counter()
    for provider in station.forecast_providers:
        forecast_c = _safe_float(
            row.get(f"model_{provider}_remaining_max_c")
        )
        if forecast_c is None:
            forecast_c = _safe_float(row.get(f"model_{provider}_today_max_c"))
        if forecast_c is None:
            continue
        represented = (
            c_to_f(forecast_c) if station.market_unit == "F" else forecast_c
        )
        if label := _label_for_value(represented, labels, station.market_unit):
            votes[label] += 1
    consensus_c = _safe_float(row.get("model_consensus_remaining_max_c"))
    if consensus_c is None:
        consensus_c = _safe_float(row.get("model_consensus_today_max_c"))
    if consensus_c is None:
        return votes, None
    represented = (
        c_to_f(consensus_c) if station.market_unit == "F" else consensus_c
    )
    return votes, _label_for_value(
        represented, labels, station.market_unit
    )


def _predict_remaining_maximum(
    row: dict[str, Any], model: StationPaperModel
) -> tuple[float, float, list[float]]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    values = pd.DataFrame(
        [{feature: _safe_float(row.get(feature)) for feature in model.feature_columns}]
    )
    x_value = model.imputer.transform(values)
    if model.rise_classifier is None:
        rise_probability = float(model.constant_rise_probability or 0.0)
    else:
        classes = list(model.rise_classifier.classes_)
        class_index = classes.index(1)
        rise_probability = float(
            model.rise_classifier.predict_proba(x_value)[0, class_index]
        )
    if model.increment_regressor is None:
        samples = list(model.fallback_positive_increments)
        positive_point = float(np.mean(samples))
    else:
        positive_point = max(0.0, float(model.increment_regressor.predict(x_value)[0]))
        samples = [
            max(0.0, float(tree.predict(x_value)[0]))
            for tree in model.increment_regressor.estimators_
        ]
    return min(1.0, max(0.0, rise_probability)), positive_point, samples


def _load_training_frame(cfg: AppConfig) -> tuple[Any, list[Path]]:
    import pandas as pd

    feature_files = sorted(cfg.reports_dir.glob("*/predictive_features.csv"))
    pretraining = cfg.reports_dir / "ncei_history" / "pretraining_features.csv.gz"
    if pretraining.exists():
        feature_files.append(pretraining)
    frames = []
    used_files: list[Path] = []
    for path in feature_files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)
        used_files.append(path)
    if not frames:
        raise FileNotFoundError("No predictive training features were found")
    data = pd.concat(frames, ignore_index=True).copy()
    if "final_temperature_target" not in data:
        data["final_temperature_target"] = data.get("final_wu_max")
    elif "final_wu_max" in data:
        data["final_temperature_target"] = data["final_temperature_target"].fillna(
            data["final_wu_max"]
        )
    if "target_source" not in data:
        data["target_source"] = "wunderground"
    else:
        data["target_source"] = data["target_source"].fillna("wunderground")
    data["_target"] = pd.to_numeric(data["final_temperature_target"], errors="coerce")
    resolution_max = pd.to_numeric(
        data.get("resolution_running_max_market_unit"), errors="coerce"
    )
    running_max = pd.to_numeric(data.get("running_max_market_unit"), errors="coerce")
    data["_current_max"] = resolution_max.fillna(running_max)
    data = data.dropna(subset=["station_id", "local_date", "_target", "_current_max"])
    if "day_history_complete" in data:
        data = data[_truth_series(data["day_history_complete"])]
    if "continuity_eligible" in data:
        data = data[_truth_series(data["continuity_eligible"])]
    if "wu_target_matches_resolved_winner" in data:
        raw_match = data["wu_target_matches_resolved_winner"]
        valid_match = raw_match.isna() | _truth_series(raw_match)
        data = data[valid_match]
    data["_remaining_increment"] = (data["_target"] - data["_current_max"]).clip(lower=0.0)
    data["_rise_again"] = (data["_remaining_increment"] > 1e-9).astype(int)
    if "decision_mode" not in data:
        data["decision_mode"] = "event_driven"
    if "observation_key" not in data:
        data["observation_key"] = data.index.astype(str)
    data = data.sort_values("received_wall_ns").drop_duplicates(
        ["station_id", "local_date", "observation_key", "decision_mode"],
        keep="first",
    )
    return data, used_files


def _add_canonical_features(frame: Any, station: StationConfig) -> Any:

    import pandas as pd

    result = frame.copy()

    def coalesce(names: tuple[str, ...]) -> Any:
        series = pd.Series(float("nan"), index=result.index, dtype=float)
        for name in names:
            if name in result:
                series = series.fillna(pd.to_numeric(result[name], errors="coerce"))
        return series

    unit_scale = 1.8 if station.market_unit == "F" else 1.0
    unit_offset = 32.0 if station.market_unit == "F" else 0.0
    consensus_today_c = coalesce(
        ("model_consensus_today_max_c", "historical_model_consensus_day1_max_c")
    )
    consensus_remaining_c = coalesce(
        (
            "model_consensus_remaining_max_c",
            "historical_model_consensus_remaining_max_c",
        )
    )
    result["canonical_forecast_today_max_market_unit"] = (
        consensus_today_c * unit_scale + unit_offset
    )
    result["canonical_forecast_remaining_max_market_unit"] = (
        consensus_remaining_c * unit_scale + unit_offset
    )
    result["canonical_forecast_spread_market_unit"] = coalesce(
        (
            "model_spread_remaining_max_c",
            "historical_model_spread_remaining_max_c",
            "model_spread_today_max_c",
            "historical_model_spread_day1_max_c",
        )
    ) * unit_scale
    live_primary_change_c = coalesce(("slope_60m",)) * 60.0
    if station.market_unit == "F":
        live_primary_change_c = live_primary_change_c / 1.8
    result["canonical_primary_temperature_change_1h_c"] = coalesce(
        ("temperature_change_1h",)
    ).fillna(live_primary_change_c)
    result["canonical_peer_temperature_change_1h_c"] = coalesce(
        ("peer_temperature_change_1h_mean", "peer_mean_slope_60m_c_per_hour")
    )
    result["canonical_peer_temperature_mean_c"] = coalesce(
        ("peer_temperature_mean_c",)
    )
    return result


def _load_calibration_history(
    cfg: AppConfig,
) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    paths = sorted(
        cfg.reports_dir.glob("model_validation/master_remaining_max_validation*.csv"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    excluded_dates = set(cfg.paper_trading.excluded_calibration_dates)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                station_id = str(row.get("station_id") or "")
                test_date = str(row.get("test_date") or "")
                if (
                    station_id not in cfg.selected_station_ids
                    or not test_date
                    or test_date in excluded_dates
                ):
                    continue
                raw_probability = _safe_float(row.get("raw_probability_rise_again"))
                positive_point = _safe_float(row.get("conditional_increment_prediction"))
                actual_increment = _safe_float(row.get("actual_remaining_increment"))
                current_max = _safe_float(row.get("current_running_max"))
                final_max = _safe_float(row.get("final_wu_max"))
                if None in (
                    raw_probability,
                    positive_point,
                    actual_increment,
                    current_max,
                    final_max,
                ):
                    continue
                key = (
                    station_id,
                    test_date,
                    str(row.get("decision_mode") or ""),
                    str(row.get("received_wall_ns") or ""),
                )
                records[key] = {
                    "test_date": test_date,
                    "decision_mode": str(row.get("decision_mode") or ""),
                    "local_hour": _safe_float(row.get("local_hour")),
                    "rise_residual": float(actual_increment > 1e-9) - raw_probability,
                    "increment_residual": (
                        actual_increment - positive_point
                        if actual_increment > 1e-9
                        else None
                    ),
                    "final_residual": final_max
                    - (current_max + raw_probability * positive_point),
                    "raw_point_final": current_max
                    + raw_probability * positive_point,
                    "forecast_point_final": _safe_float(
                        row.get("forecast_consensus_prediction")
                    ),
                    "current_max": current_max,
                    "final_max": final_max,
                }
    by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in records.items():
        by_station[key[0]].append(row)
    for values in by_station.values():
        values.sort(
            key=lambda row: (
                str(row["test_date"]),
                str(row["decision_mode"]),
                _safe_float(row.get("local_hour")) or -1.0,
            )
        )
    return dict(by_station), paths


def _station_execution_buffers(cfg: AppConfig) -> dict[str, float]:
    path = cfg.reports_dir / "execution_costs" / "live_execution_cost_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        str(row["station_id"]): max(
            cfg.paper_trading.default_execution_buffer,
            float(row["vwap_cost_from_mid"]["p90"]),
        )
        for row in payload.get("by_station", [])
        if row.get("station_id")
        and _safe_float(row.get("vwap_cost_from_mid", {}).get("p90")) is not None
    }


def _final_wu_maxima(cfg: AppConfig) -> dict[tuple[str, str], float]:
    maxima: dict[tuple[str, str], tuple[int, float]] = {}
    events = load_events_across_runs(cfg.data_dir, {"wunderground_browser"})
    for event in events:
        if (
            event.get("event_type") != "resolution_snapshot"
            or event.get("payload", {}).get("resolution_final") is not True
        ):
            continue
        station_id = str(event.get("station_id") or "")
        local_date = str(event.get("payload", {}).get("local_date") or "")
        value = _safe_float(event.get("payload", {}).get("daily_max_from_network"))
        if station_id and local_date and value is not None:
            maxima[(station_id, local_date)] = (
                int(event.get("received_wall_ns") or 0),
                value,
            )
    return {key: value for key, (_, value) in maxima.items()}


def _settle_rows(
    audit_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    final_maxima: dict[tuple[str, str], float],
    cfg: AppConfig,
) -> None:
    stations = {station.station_id: station for station in cfg.stations.values()}
    for row in audit_rows:
        key = (str(row.get("station_id")), str(row.get("local_date")))
        final_max = final_maxima.get(key)
        station = stations.get(key[0])
        weather = _json_floats(row.get("weather_distribution_json"))
        market = _json_floats(row.get("market_distribution_json"))
        labels = sorted(set(weather) | set(market))
        if final_max is None or station is None or not labels:
            continue
        winner = _label_for_value(final_max, labels, station.market_unit)
        if not winner:
            continue
        row["winning_bucket"] = winner
        row["weather_brier"] = _brier(weather, winner) if weather else ""
        row["market_brier"] = _brier(market, winner) if market else ""
        if weather and market:
            row["brier_improvement"] = float(row["market_brier"]) - float(
                row["weather_brier"]
            )
    for trade in trade_rows:
        key = (str(trade.get("station_id")), str(trade.get("local_date")))
        final_max = final_maxima.get(key)
        station = stations.get(key[0])
        distribution = _json_floats(trade.get("weather_distribution_json"))
        if final_max is None or station is None or not distribution:
            continue
        winner = _label_for_value(final_max, sorted(distribution), station.market_unit)
        if not winner:
            continue
        won = str(trade.get("bucket")) == winner
        cost = _safe_float(trade.get("conservative_cost_per_share"))
        shares = _safe_float(trade.get("requested_shares"))
        if cost is None or shares is None:
            continue
        pnl = 1.0 - cost if won else -cost
        trade["winning_bucket"] = winner
        trade["won"] = won
        trade["realized_pnl_per_share"] = pnl
        trade["realized_pnl_total"] = pnl * shares
        trade["status"] = "paper_settled"


def _paper_summary(
    audit_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    bundle: PaperModelBundle,
    rules: PaperTradingConfig,
    station_buffers: dict[str, float],
) -> dict[str, Any]:
    station_ids = sorted(bundle.stations)
    by_station = [
        _station_paper_summary(station_id, audit_rows, trade_rows, rules)
        for station_id in station_ids
    ]
    supported = [
        row["station_id"] for row in by_station if row["alpha_gate_passed"] is True
    ]
    settled_trades = [
        row for row in trade_rows if str(row.get("status")) == "paper_settled"
    ]
    rejection_counts: Counter[str] = Counter()
    for row in audit_rows:
        for reason in str(row.get("rejection_reasons") or "").split("|"):
            if reason:
                rejection_counts[reason] += 1
    return {
        "status": "supported_candidate" if supported else "prospective_evidence_pending",
        "alpha_present": bool(supported),
        "supported_stations": supported,
        "model_deployed_at_ns": bundle.deployed_at_ns,
        "model_deployed_at_utc": bundle.deployed_at_utc,
        "model_name": bundle.model_name,
        "calibration_strategy": bundle.calibration_strategy,
        "hypothesis_only": bool(bundle.parameters.get("hypothesis_only", False)),
        "strategy_rules": bundle.parameters.get(
            "persistent_disagreement_rules"
        ),
        "live_orders_enabled": False,
        "audit_rows": len(audit_rows),
        "base_eligible_rows": sum(_truth(row.get("base_signal_eligible")) for row in audit_rows),
        "persistent_rows": sum(_truth(row.get("persistent")) for row in audit_rows),
        "paper_trades": len(trade_rows),
        "settled_paper_trades": len(settled_trades),
        "open_paper_trades": len(trade_rows) - len(settled_trades),
        "settled_total_pnl": sum(
            _safe_float(row.get("realized_pnl_total")) or 0.0 for row in settled_trades
        ),
        "rejection_counts": dict(rejection_counts.most_common()),
        "station_p90_cost_from_mid": station_buffers,
        "alpha_gate": {
            "minimum_settled_trade_station_days_per_station": rules.minimum_alpha_station_days,
            "requires_positive_pnl_ci95_lower_bound": True,
            "requires_positive_brier_improvement_ci95_lower_bound": True,
            "unit_of_independence": "station_id + local_date",
        },
        "by_station": by_station,
        "warnings": [
            "No real order is sent by this module.",
            "A signal must remain eligible for the configured persistence window.",
            "At most one paper entry is permitted per station-day.",
            "Alpha is not declared from point-temperature MAE or correlated intraday rows.",
        ],
    }


def _station_paper_summary(
    station_id: str,
    audit_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    rules: PaperTradingConfig,
) -> dict[str, Any]:
    station_audit = [
        row
        for row in audit_rows
        if str(row.get("station_id")) == station_id
        and _safe_float(row.get("brier_improvement")) is not None
    ]
    brier_by_day: dict[str, list[float]] = defaultdict(list)
    for row in station_audit:
        brier_by_day[str(row["local_date"])].append(
            float(row["brier_improvement"])
        )
    daily_brier = [
        sum(values) / len(values) for values in brier_by_day.values() if values
    ]
    station_trades = [
        row
        for row in trade_rows
        if str(row.get("station_id")) == station_id
        and str(row.get("status")) == "paper_settled"
    ]
    pnl_by_day: dict[str, float] = defaultdict(float)
    for row in station_trades:
        pnl_by_day[str(row["local_date"])] += _safe_float(
            row.get("realized_pnl_per_share")
        ) or 0.0
    pnl_values = list(pnl_by_day.values())
    pnl_ci = _bootstrap_mean_ci(pnl_values)
    brier_ci = _bootstrap_mean_ci(daily_brier)
    enough = len(pnl_values) >= rules.minimum_alpha_station_days
    passed = bool(
        enough
        and pnl_ci
        and brier_ci
        and _safe_float(pnl_ci.get("low")) is not None
        and _safe_float(brier_ci.get("low")) is not None
        and float(pnl_ci["low"]) > 0
        and float(brier_ci["low"]) > 0
    )
    return {
        "station_id": station_id,
        "settled_trade_station_days": len(pnl_values),
        "settled_brier_station_days": len(daily_brier),
        "total_pnl_per_share": sum(pnl_values),
        "mean_pnl_per_station_day": (
            sum(pnl_values) / len(pnl_values) if pnl_values else None
        ),
        "mean_pnl_per_station_day_ci95": pnl_ci,
        "mean_brier_improvement": (
            sum(daily_brier) / len(daily_brier) if daily_brier else None
        ),
        "mean_brier_improvement_ci95": brier_ci,
        "minimum_station_days_reached": enough,
        "alpha_gate_passed": passed,
    }


def _one_increment_per_day(frame: Any) -> list[float]:
    if frame.empty:
        return []
    return [
        float(day["_remaining_increment"].median())
        for _, day in frame.groupby("local_date", sort=True)
    ]


def _resolve_feature_path(cfg: AppConfig, run_id: str | None) -> Path:
    if run_id:
        path = cfg.reports_dir / run_id / "predictive_features.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(cfg.reports_dir.glob("*/predictive_features.csv"))
    if not candidates:
        raise FileNotFoundError("No built predictive feature file was found")
    return candidates[-1]


def _load_bundle(path: Path) -> PaperModelBundle:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, PaperModelBundle):
        raise RuntimeError("Invalid local paper model bundle")
    return bundle


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _decision_id(row: dict[str, Any]) -> str:
    raw = "|".join(
        (
            str(row.get("run_id") or ""),
            str(row.get("station_id") or ""),
            str(row.get("local_date") or ""),
            str(int(float(row.get("received_wall_ns") or 0))),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _json_floats(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): parsed_value
        for key, raw in parsed.items()
        if (parsed_value := _safe_float(raw)) is not None
    }


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _truth_series(series: Any) -> Any:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = sorted({str(key) for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
