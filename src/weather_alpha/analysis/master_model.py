from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..parsing.buckets import c_to_f
from .model import (
    NON_FEATURE_COLUMNS,
    _best_edge,
    _bootstrap_mean_ci,
    _brier,
    _daily_means,
    _fee_per_share,
    _label_for_value,
    _mean,
    _normalize,
    _read_market_quotes,
    _safe_float,
)

RECOMMENDED_RESEARCH_DAYS = 60
ROBUST_VALIDATION_DAYS = 100
FORECAST_BLEND_PRIOR_STRENGTH = 20.0
FORECAST_BLEND_MIN_RMSE_IMPROVEMENT = 0.02


def remaining_max_samples(
    current_max: float,
    positive_increment_samples: list[float],
) -> list[float]:

    return [current_max + max(0.0, float(increment)) for increment in positive_increment_samples]


def remaining_max_distribution(
    current_max: float,
    probability_rise_again: float,
    positive_increment_samples: list[float],
    labels: list[str],
    unit: str,
) -> dict[str, float]:

    if not labels:
        return {}
    probability_rise_again = min(1.0, max(0.0, float(probability_rise_again)))
    counts = {label: 0.0 for label in labels}
    no_rise_label = _label_for_value(current_max, labels, unit)
    if no_rise_label is not None:
        counts[no_rise_label] += 1.0 - probability_rise_again
    samples = remaining_max_samples(current_max, positive_increment_samples)
    if not samples:
        samples = [current_max]
    weight = probability_rise_again / len(samples)
    for sample in samples:
        label = _label_for_value(sample, labels, unit)
        if label is not None:
            counts[label] += weight

    return _normalize(counts)


def conformal_final_distribution(
    current_max: float,
    raw_point_final: float,
    prior_out_of_sample_residuals: list[float],
    labels: list[str],
    unit: str,
    *,
    pseudocount_per_bucket: float = 0.25,
) -> dict[str, float]:

    if not labels or not prior_out_of_sample_residuals:
        return {}
    counts = {label: max(0.0, pseudocount_per_bucket) for label in labels}
    for residual in prior_out_of_sample_residuals:
        sample = max(current_max, raw_point_final + float(residual))
        label = _label_for_value(sample, labels, unit)
        if label is not None:
            counts[label] += 1.0
    return _normalize(counts)


def forecast_anchored_conformal_distribution(
    current_max: float,
    raw_point_final: float,
    forecast_point_final: float | None,
    calibration_history: list[dict[str, Any]],
    labels: list[str],
    unit: str,
    *,
    local_hour: float | None,
    decision_mode: str,
    prior_strength: float = FORECAST_BLEND_PRIOR_STRENGTH,
    minimum_relative_rmse_improvement: float = FORECAST_BLEND_MIN_RMSE_IMPROVEMENT,
) -> tuple[dict[str, float], dict[str, Any]]:

    rows = _calibration_rows(calibration_history, local_hour, decision_mode)
    usable = [
        row
        for row in rows
        if all(
            _safe_float(row.get(field)) is not None
            for field in (
                "raw_point_final",
                "forecast_point_final",
                "current_max",
                "final_max",
            )
        )
    ]
    diagnostics: dict[str, Any] = {
        "calibration_strategy": "forecast_anchored_conformal_v2",
        "calibration_days": len(usable),
        "raw_model_weight": None,
        "model_weight": None,
        "raw_point_final": raw_point_final,
        "forecast_point_final": forecast_point_final,
        "blended_point_final": None,
        "raw_rmse": None,
        "blended_rmse": None,
        "relative_rmse_improvement": None,
        "challenger_accepted": False,
    }
    if (
        not labels
        or forecast_point_final is None
        or not math.isfinite(float(forecast_point_final))
        or not usable
    ):
        return {}, diagnostics

    raw_values = [float(row["raw_point_final"]) for row in usable]
    forecast_values = [float(row["forecast_point_final"]) for row in usable]
    current_values = [float(row["current_max"]) for row in usable]
    final_values = [float(row["final_max"]) for row in usable]
    deltas = [raw - forecast for raw, forecast in zip(raw_values, forecast_values)]
    denominator = sum(delta * delta for delta in deltas)
    raw_weight = (
        sum(
            delta * (final - forecast)
            for delta, final, forecast in zip(deltas, final_values, forecast_values)
        )
        / denominator
        if denominator > 1e-12
        else 0.5
    )
    raw_weight = min(1.0, max(0.0, raw_weight))
    strength = max(0.0, float(prior_strength))
    model_weight = (
        (len(usable) * raw_weight + strength * 0.5) / (len(usable) + strength)
        if len(usable) + strength > 0
        else 0.5
    )
    historical_centers = [
        max(current, model_weight * raw + (1.0 - model_weight) * forecast)
        for current, raw, forecast in zip(
            current_values, raw_values, forecast_values
        )
    ]
    raw_rmse = math.sqrt(
        sum((final - raw) ** 2 for final, raw in zip(final_values, raw_values))
        / len(usable)
    )
    blended_rmse = math.sqrt(
        sum(
            (final - center) ** 2
            for final, center in zip(final_values, historical_centers)
        )
        / len(usable)
    )
    relative_improvement = (
        (raw_rmse - blended_rmse) / raw_rmse if raw_rmse > 1e-12 else 0.0
    )
    blended_point = max(
        float(current_max),
        model_weight * float(raw_point_final)
        + (1.0 - model_weight) * float(forecast_point_final),
    )
    residuals = [
        final - center for final, center in zip(final_values, historical_centers)
    ]
    accepted = relative_improvement >= float(minimum_relative_rmse_improvement)
    diagnostics.update(
        {
            "raw_model_weight": raw_weight,
            "model_weight": model_weight,
            "blended_point_final": blended_point,
            "raw_rmse": raw_rmse,
            "blended_rmse": blended_rmse,
            "relative_rmse_improvement": relative_improvement,
            "challenger_accepted": accepted,
        }
    )
    return (
        conformal_final_distribution(
            current_max,
            blended_point,
            residuals,
            labels,
            unit,
        ),
        diagnostics,
    )


def evaluate_master_model(
    cfg: AppConfig,
    *,
    min_train_days: int = 20,
    min_calibration_days: int = 10,
    edge_threshold: float = 0.04,
    trees: int = 300,
    max_trades_per_day: int = 1,
    include_ncei_pretraining: bool = True,
    ncei_source_weight: float = 0.5,
    recency_half_life_years: float = 6.0,
    evaluation_start_date: str | None = None,
    calibration_seed_path: Path | None = None,
    calibration_exclude_dates: set[str] | None = None,
    output_tag: str | None = None,
) -> dict[str, Path]:

    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies: pip install -e '.[analysis]'") from exc

    if ncei_source_weight <= 0 or ncei_source_weight > 1:
        raise ValueError("ncei_source_weight must be in (0, 1]")
    if recency_half_life_years <= 0:
        raise ValueError("recency_half_life_years must be positive")
    if evaluation_start_date is not None:
        try:
            evaluation_start_date = date.fromisoformat(evaluation_start_date).isoformat()
        except ValueError as exc:
            raise ValueError("evaluation_start_date must use YYYY-MM-DD") from exc
    if calibration_seed_path is not None and evaluation_start_date is None:
        raise ValueError("calibration_seed_path requires evaluation_start_date")
    calibration_exclude_dates = calibration_exclude_dates or set()
    try:
        calibration_exclude_dates = {
            date.fromisoformat(value).isoformat() for value in calibration_exclude_dates
        }
    except ValueError as exc:
        raise ValueError("calibration_exclude_dates must use YYYY-MM-DD") from exc
    if output_tag and not output_tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("output_tag may contain only letters, digits, hyphens and underscores")
    feature_files = sorted(cfg.reports_dir.glob("*/predictive_features.csv"))
    pretraining_path = cfg.reports_dir / "ncei_history" / "pretraining_features.csv.gz"
    if include_ncei_pretraining and pretraining_path.exists():
        feature_files.append(pretraining_path)
    frames = []
    for path in feature_files:
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["feature_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No predictive_features.csv files found. Run collect and build-dataset first."
        )


    data = pd.concat(frames, ignore_index=True).copy()
    input_rows = len(data)
    if "decision_mode" not in data.columns:
        data["decision_mode"] = "event_driven"
    data["decision_mode"] = data["decision_mode"].fillna("event_driven").astype(str)
    if "final_temperature_target" not in data.columns:
        data["final_temperature_target"] = data.get("final_wu_max")
    elif "final_wu_max" in data.columns:
        data["final_temperature_target"] = data["final_temperature_target"].fillna(
            data["final_wu_max"]
        )
    data["final_temperature_target"] = pd.to_numeric(
        data["final_temperature_target"], errors="coerce"
    )
    if "target_source" not in data.columns:
        data["target_source"] = "wunderground"
    else:
        data["target_source"] = data["target_source"].fillna("wunderground").astype(str)
    if "evaluation_eligible" not in data.columns:
        data["_evaluation_eligible"] = True
    else:
        raw_evaluation = data["evaluation_eligible"]
        data["_evaluation_eligible"] = raw_evaluation.isna() | raw_evaluation.astype(
            str
        ).str.lower().isin({"true", "1", "1.0"})
    data = data.dropna(
        subset=[
            "station_id",
            "local_date",
            "final_temperature_target",
            "running_max_market_unit",
        ]
    )


    if "resolution_running_max_market_unit" not in data.columns:
        data["resolution_running_max_market_unit"] = data["running_max_market_unit"]
    else:


        data["resolution_running_max_market_unit"] = pd.to_numeric(
            data["resolution_running_max_market_unit"], errors="coerce"
        ).fillna(pd.to_numeric(data["running_max_market_unit"], errors="coerce"))
        data = data.dropna(subset=["resolution_running_max_market_unit"])
    if "day_history_complete" in data.columns:
        complete = data["day_history_complete"].astype(str).str.lower().isin({"true", "1", "1.0"})
        data = data[complete]
    if "continuity_eligible" in data.columns:
        continuous = data["continuity_eligible"].astype(str).str.lower().isin(
            {"true", "1", "1.0"}
        )
        data = data[continuous]
    data = data.sort_values("received_wall_ns").drop_duplicates(
        subset=["station_id", "local_date", "observation_key", "decision_mode"], keep="first"
    )
    data["remaining_increment_target"] = (
        data["final_temperature_target"].astype(float)
        - data["resolution_running_max_market_unit"].astype(float)
    ).clip(lower=0.0)
    data["rise_again_target"] = (data["remaining_increment_target"] > 1e-9).astype(int)

    calibration_seed_by_station: dict[str, list[dict[str, Any]]] = {}
    if calibration_seed_path is not None:
        if not calibration_seed_path.exists():
            raise FileNotFoundError(calibration_seed_path)
        seed_frame = pd.read_csv(calibration_seed_path)
        required_seed_columns = {
            "station_id",
            "test_date",
            "decision_mode",
            "local_hour",
            "actual_remaining_increment",
            "raw_probability_rise_again",
            "conditional_increment_prediction",
            "current_running_max",
            "final_wu_max",
            "probability_calibrated",
            "weather_model_distribution_json",
            "market_distribution_json",
            "winning_bucket",
        }
        missing_seed_columns = required_seed_columns - set(seed_frame.columns)
        if missing_seed_columns:
            raise ValueError(
                "Calibration seed is missing columns: "
                + ", ".join(sorted(missing_seed_columns))
            )
        seed_frame["test_date"] = seed_frame["test_date"].astype(str)
        seed_frame = seed_frame[seed_frame["test_date"] < str(evaluation_start_date)]
        seed_frame = seed_frame[
            ~seed_frame["test_date"].isin(calibration_exclude_dates)
        ]
        for seed in seed_frame.to_dict(orient="records"):
            actual_increment = float(seed["actual_remaining_increment"])
            raw_p_rise = float(seed["raw_probability_rise_again"])
            positive_point = float(seed["conditional_increment_prediction"])
            current_max = float(seed["current_running_max"])
            final_max = float(seed["final_wu_max"])
            calibration_seed_by_station.setdefault(str(seed["station_id"]), []).append(
                {
                    "test_date": str(seed["test_date"]),
                    "decision_mode": str(seed["decision_mode"]),
                    "local_hour": _safe_float(seed.get("local_hour")),
                    "rise_residual": float(actual_increment > 1e-9) - raw_p_rise,
                    "increment_residual": (
                        actual_increment - positive_point
                        if actual_increment > 1e-9
                        else None
                    ),
                    "final_residual": final_max
                    - (current_max + raw_p_rise * positive_point),
                    "raw_point_final": current_max + raw_p_rise * positive_point,
                    "forecast_point_final": _safe_float(
                        seed.get("forecast_consensus_prediction")
                    ),
                    "current_max": current_max,
                    "final_max": final_max,
                    "probability_calibrated": _as_bool(
                        seed.get("probability_calibrated")
                    ),
                    "weather_model_probs": _json_probability_dict(
                        seed.get("weather_model_distribution_json")
                    ),
                    "market_probs": _json_probability_dict(
                        seed.get("market_distribution_json")
                    ),
                    "winner": str(seed.get("winning_bucket") or ""),
                }
            )

    readiness_rows: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        station_data = data[data["station_id"] == station.station_id]
        dates = sorted(str(value) for value in station_data["local_date"].dropna().unique())
        evaluation_dates = sorted(
            str(value)
            for value in station_data.loc[
                station_data["_evaluation_eligible"], "local_date"
            ]
            .dropna()
            .unique()
        )
        pretraining_dates = sorted(
            str(value)
            for value in station_data.loc[
                ~station_data["_evaluation_eligible"], "local_date"
            ]
            .dropna()
            .unique()
        )
        months = sorted({value[:7] for value in dates})
        readiness_rows.append(
            {
                "station_id": station.station_id,
                "city": station.city,
                "eligible_rows": len(station_data),


                "finalized_station_days": len(evaluation_dates),
                "training_station_days": len(dates),
                "pretraining_station_days": len(pretraining_dates),
                "evaluation_station_days": len(evaluation_dates),
                "first_local_date": dates[0] if dates else None,
                "last_local_date": dates[-1] if dates else None,
                "calendar_months": len(months),
                "event_driven_rows": int(
                    (station_data["decision_mode"] == "event_driven").sum()
                ),
                "scheduled_rows": int((station_data["decision_mode"] == "scheduled").sum()),
                "forecast_update_rows": int(
                    (station_data["decision_mode"] == "forecast_update").sum()
                ),
                "chronological_validation_ready": any(
                    len([value for value in dates if value < evaluation_date])
                    >= min_train_days
                    for evaluation_date in evaluation_dates
                ),
                "calibration_horizon_ready": len(evaluation_dates) >= min_calibration_days,
                "recommended_research_days": RECOMMENDED_RESEARCH_DAYS,
                "recommended_research_ready": (
                    len(evaluation_dates) >= RECOMMENDED_RESEARCH_DAYS
                ),
                "robust_validation_days": ROBUST_VALIDATION_DAYS,
                "robust_validation_ready": len(evaluation_dates) >= ROBUST_VALIDATION_DAYS,
            }
        )

    results: list[dict[str, Any]] = []
    station_summaries: list[dict[str, Any]] = []
    importance_totals: dict[tuple[str, str, str], tuple[float, int]] = {}
    for station_id, station_frame in data.groupby("station_id", sort=True):
        station = next(
            (item for item in cfg.stations.values() if item.station_id == station_id), None
        )
        if station is None:
            continue
        station_frame = station_frame.copy()
        dates = sorted(station_frame["local_date"].dropna().unique())
        evaluation_dates = sorted(
            station_frame.loc[station_frame["_evaluation_eligible"], "local_date"]
            .dropna()
            .unique()
        )
        if evaluation_start_date is not None:
            evaluation_dates = [
                value for value in evaluation_dates if str(value) >= evaluation_start_date
            ]
        excluded = NON_FEATURE_COLUMNS | {
            "feature_file",
            "_evaluation_eligible",
            "remaining_increment_target",
            "rise_again_target",
        }
        feature_columns: list[str] = []
        for column in station_frame.columns:
            if column in excluded:
                continue
            converted = pd.to_numeric(station_frame[column], errors="coerce")
            if converted.notna().any():
                station_frame[column] = converted
                feature_columns.append(column)
        if len(dates) <= min_train_days or not evaluation_dates or not feature_columns:
            station_summaries.append(
                {
                    "station_id": station_id,
                    "status": "insufficient_days",
                    "days": len(dates),
                    "required_prior_days": min_train_days,
                    "feature_count": len(feature_columns),
                }
            )
            continue

        station_start = len(results)
        calibration_history = list(calibration_seed_by_station.get(str(station_id), []))
        trades_by_mode_day: dict[tuple[str, str], int] = {}
        for test_date in evaluation_dates:
            prior_dates = [date for date in dates if date < test_date]
            if len(prior_dates) < min_train_days:
                continue
            train = station_frame[station_frame["local_date"].isin(prior_dates)]
            test = station_frame[
                (station_frame["local_date"] == test_date)
                & station_frame["_evaluation_eligible"]
            ].sort_values("received_wall_ns")
            if train.empty or test.empty:
                continue
            fold_features = [
                feature for feature in feature_columns if train[feature].notna().any()
            ]
            if not fold_features:
                continue
            imputer = SimpleImputer(strategy="median", keep_empty_features=False)
            x_train = imputer.fit_transform(train[fold_features])
            x_test = imputer.transform(test[fold_features])
            retained = list(imputer.get_feature_names_out(fold_features))
            if x_train.shape[1] == 0:
                continue
            rows_per_day = train.groupby("local_date")["local_date"].transform("count")
            day_weights = (1.0 / rows_per_day.astype(float)).to_numpy()
            train_dates = pd.to_datetime(train["local_date"], errors="coerce")
            age_days = (pd.Timestamp(str(test_date)) - train_dates).dt.days.clip(lower=0)
            recency_weights = 0.5 ** (
                age_days.to_numpy(dtype=float) / (365.25 * recency_half_life_years)
            )
            source_weights = (
                train["target_source"]
                .astype(str)
                .str.lower()
                .eq("ncei_global_hourly")
                .map({True: ncei_source_weight, False: 1.0})
                .to_numpy(dtype=float)
            )
            day_weights = day_weights * recency_weights * source_weights
            rise_train = train["rise_again_target"].astype(int).to_numpy()
            increment_train = train["remaining_increment_target"].astype(float).to_numpy()
            rise_importances = None
            increment_importances = None

            if len(np.unique(rise_train)) == 1:
                raw_rise_probability = np.full(len(test), float(rise_train[0]))
            else:
                classifier = ExtraTreesClassifier(
                    n_estimators=trees,
                    min_samples_leaf=4,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                )
                classifier.fit(x_train, rise_train, sample_weight=day_weights)
                class_index = list(classifier.classes_).index(1)
                raw_rise_probability = classifier.predict_proba(x_test)[:, class_index]
                rise_importances = classifier.feature_importances_

            positive_mask = increment_train > 1e-9
            positive_days = train.loc[positive_mask, "local_date"].nunique()
            if positive_mask.sum() >= 4 and positive_days >= 2:
                regressor = ExtraTreesRegressor(
                    n_estimators=trees,
                    min_samples_leaf=2,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=43,
                )
                regressor.fit(
                    x_train[positive_mask],
                    increment_train[positive_mask],
                    sample_weight=day_weights[positive_mask],
                )
                positive_point = np.maximum(0.0, regressor.predict(x_test))
                positive_tree_samples = np.vstack(
                    [tree.predict(x_test) for tree in regressor.estimators_]
                ).T
                increment_importances = regressor.feature_importances_
            else:
                empirical = _one_value_per_day(train[positive_mask])
                fallback = empirical or [0.0]
                positive_point = np.full(len(test), float(np.mean(fallback)))
                positive_tree_samples = np.tile(np.asarray(fallback, dtype=float), (len(test), 1))

            for stage, importances in (
                ("rise_again", rise_importances),
                ("positive_increment", increment_importances),
            ):
                if importances is None:
                    continue
                for feature, importance in zip(retained, importances, strict=True):
                    key = (station_id, stage, feature)
                    total, folds = importance_totals.get(key, (0.0, 0))
                    importance_totals[key] = (total + float(importance), folds + 1)

            new_calibration_rows: list[dict[str, Any]] = []
            for row_index, (_, row) in enumerate(test.iterrows()):
                mode = str(row.get("decision_mode") or "event_driven")
                local_hour = _safe_float(row.get("local_hour"))
                (
                    rise_residuals,
                    increment_residuals,
                    final_residuals,
                    calibration_days,
                ) = _calibration_residuals(calibration_history, local_hour, mode)
                probability_calibrated = (
                    calibration_days >= min_calibration_days
                    and len(final_residuals) >= min_calibration_days
                )
                raw_p_rise = float(raw_rise_probability[row_index])
                p_rise = min(
                    1.0,
                    max(
                        0.0,
                        raw_p_rise + (_mean(rise_residuals) or 0.0)
                        if probability_calibrated
                        else raw_p_rise,
                    ),
                )
                if probability_calibrated:
                    increment_samples = [
                        max(0.0, float(positive_point[row_index]) + residual)
                        for residual in increment_residuals
                    ]
                else:
                    increment_samples = [
                        max(0.0, float(value)) for value in positive_tree_samples[row_index]
                    ]
                (
                    labels,
                    market_mid,
                    market_ask,
                    fee_rates,
                    fee_exponents,
                    fees_enabled,
                ) = _read_market_quotes(row)
                quote_coverage = _safe_float(row.get("market_quote_coverage")) or 0.0
                execution_metadata_complete = bool(labels) and all(
                    label in fees_enabled for label in labels
                )
                current_max = float(row["resolution_running_max_market_unit"])
                raw_point_final = current_max + raw_p_rise * float(
                    positive_point[row_index]
                )
                consensus_c = _safe_float(row.get("model_consensus_today_max_c"))
                if consensus_c is None:
                    consensus_c = _safe_float(
                        row.get("historical_model_consensus_day1_max_c")
                    )
                consensus_market = (
                    c_to_f(consensus_c)
                    if consensus_c is not None and station.market_unit == "F"
                    else consensus_c
                )
                forecast_baseline = (
                    max(current_max, consensus_market)
                    if consensus_market is not None
                    else None
                )
                weather_model_probs = (
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
                        p_rise,
                        increment_samples,
                        labels,
                        station.market_unit,
                    )
                )
                winner = str(row.get("winning_bucket") or "")
                if not winner:
                    winner = (
                        _label_for_value(
                            float(row["final_wu_max"]), labels, station.market_unit
                        )
                        or ""
                    )
                actual_increment = float(row["remaining_increment_target"])
                actual_rise = float(row["rise_again_target"])
                market_probs = _normalize(market_mid) if quote_coverage >= 0.999 else {}
                blend_weight, blend_days = _market_blend_weight(
                    calibration_history, local_hour, mode
                )
                market_blend_calibrated = bool(
                    probability_calibrated
                    and market_probs
                    and blend_days >= min_calibration_days
                )
                model_probs = (
                    _blend_distributions(
                        market_probs,
                        weather_model_probs,
                        blend_weight if market_blend_calibrated else 0.0,
                    )
                    if market_probs
                    else weather_model_probs
                )
                new_calibration_rows.append(
                    {
                        "test_date": test_date,
                        "decision_mode": mode,
                        "local_hour": local_hour,
                        "rise_residual": actual_rise - raw_p_rise,
                        "increment_residual": (
                            actual_increment - float(positive_point[row_index])
                            if actual_rise > 0
                            else None
                        ),
                        "final_residual": float(row["final_wu_max"]) - raw_point_final,
                        "raw_point_final": raw_point_final,
                        "forecast_point_final": forecast_baseline,
                        "current_max": current_max,
                        "final_max": float(row["final_wu_max"]),
                        "probability_calibrated": probability_calibrated,
                        "weather_model_probs": weather_model_probs,
                        "market_probs": market_probs,
                        "winner": winner,
                    }
                )
                if not labels or not winner or not model_probs:
                    continue
                chosen_label, chosen_edge = (
                    _best_edge(
                        model_probs,
                        market_ask,
                        fee_rates,
                        fee_exponents,
                        fees_enabled,
                    )
                    if execution_metadata_complete
                    else (None, None)
                )
                ask = market_ask.get(chosen_label) if chosen_label else None
                fee = (
                    _fee_per_share(
                        ask,
                        fee_rates.get(chosen_label),
                        fee_exponents.get(chosen_label),
                        fees_enabled.get(chosen_label, 0.0) > 0,
                    )
                    if chosen_label and ask is not None
                    else None
                )
                trade_key = (str(test_date), mode)
                traded = bool(
                    chosen_label
                    and ask is not None
                    and 0 < ask < 1
                    and chosen_edge is not None
                    and chosen_edge >= edge_threshold
                    and probability_calibrated
                    and market_blend_calibrated
                    and execution_metadata_complete
                    and quote_coverage >= 0.999
                    and trades_by_mode_day.get(trade_key, 0) < max_trades_per_day
                )
                realized_pnl = None
                if traded and ask is not None:
                    trades_by_mode_day[trade_key] = trades_by_mode_day.get(trade_key, 0) + 1
                    realized_pnl = (
                        1.0 - ask - (fee or 0.0)
                        if chosen_label == winner
                        else -ask - (fee or 0.0)
                    )
                point_final = current_max + p_rise * float(positive_point[row_index])
                results.append(
                    {
                        "station_id": station_id,
                        "test_date": test_date,
                        "decision_mode": mode,
                        "received_wall_ns": int(row["received_wall_ns"]),
                        "local_hour": local_hour,
                        "current_running_max": current_max,
                        "final_wu_max": float(row["final_wu_max"]),
                        "actual_remaining_increment": actual_increment,
                        "raw_probability_rise_again": raw_p_rise,
                        "probability_rise_again": p_rise,
                        "conditional_increment_prediction": float(
                            positive_point[row_index]
                        ),
                        "point_prediction": point_final,
                        "absolute_error": abs(point_final - float(row["final_wu_max"])),
                        "persistence_prediction": current_max,
                        "persistence_absolute_error": abs(
                            current_max - float(row["final_wu_max"])
                        ),
                        "forecast_consensus_prediction": forecast_baseline,
                        "forecast_consensus_absolute_error": (
                            abs(forecast_baseline - float(row["final_wu_max"]))
                            if forecast_baseline is not None
                            else None
                        ),
                        "winning_bucket": winner,
                        "model_probability_winner": model_probs.get(winner, 0.0),
                        "market_probability_winner": market_probs.get(winner, 0.0),
                        "model_brier": _brier(model_probs, winner),
                        "weather_model_brier": _brier(weather_model_probs, winner),
                        "market_brier": _brier(market_probs, winner) if market_probs else None,
                        "model_log_loss": -math.log(
                            max(model_probs.get(winner, 0.0), 1e-12)
                        ),
                        "market_log_loss": (
                            -math.log(max(market_probs.get(winner, 0.0), 1e-12))
                            if market_probs
                            else None
                        ),
                        "probability_calibrated": probability_calibrated,
                        "calibration_days": calibration_days,
                        "market_blend_calibrated": market_blend_calibrated,
                        "market_blend_calibration_days": blend_days,
                        "weather_model_weight": blend_weight
                        if market_blend_calibrated
                        else 0.0,
                        "chosen_bucket": chosen_label,
                        "chosen_ask": ask,
                        "chosen_fee_per_share": fee,
                        "estimated_edge": chosen_edge,
                        "traded": traded,
                        "realized_pnl_per_share": realized_pnl,
                        "train_days": len(prior_dates),
                        "train_rows": len(train),
                        "feature_count": len(retained),
                        "model_distribution_json": json.dumps(model_probs, sort_keys=True),
                        "weather_model_distribution_json": json.dumps(
                            weather_model_probs, sort_keys=True
                        ),
                        "market_distribution_json": json.dumps(market_probs, sort_keys=True),
                        "historical_market_price_json": json.dumps(
                            market_mid, sort_keys=True
                        ),
                        "market_price_history_only": str(
                            row.get("market_price_history_only", "")
                        ).strip().lower()
                        in {"true", "1", "1.0"},
                        "market_price_max_age_seconds": _safe_float(
                            row.get("market_price_max_age_seconds")
                        ),
                        "market_quote_coverage": quote_coverage,
                        "historical_market_fee_rate_json": json.dumps(
                            fee_rates, sort_keys=True
                        ),
                        "historical_market_fee_exponent_json": json.dumps(
                            fee_exponents, sort_keys=True
                        ),
                        "historical_market_fees_enabled_json": json.dumps(
                            fees_enabled, sort_keys=True
                        ),
                    }
                )
            calibration_history.extend(new_calibration_rows)

        station_rows = results[station_start:]
        if station_rows:
            for mode in sorted({str(row["decision_mode"]) for row in station_rows}):
                mode_rows = [row for row in station_rows if row["decision_mode"] == mode]
                station_summaries.append(
                    _summarize(station_id, mode, mode_rows, len(evaluation_dates))
                )

    output_dir = cfg.reports_dir / "model_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    results_path = output_dir / f"master_remaining_max_validation{suffix}.csv"
    summary_path = output_dir / f"master_remaining_max_summary{suffix}.json"
    importance_path = output_dir / f"master_feature_importance{suffix}.csv"
    readiness_path = output_dir / f"master_model_readiness{suffix}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    pd.DataFrame(readiness_rows).to_csv(readiness_path, index=False)
    importance_rows = [
        {
            "station_id": station_id,
            "stage": stage,
            "feature": feature,
            "mean_importance": total / folds,
            "evaluated_folds": folds,
        }
        for (station_id, stage, feature), (total, folds) in sorted(importance_totals.items())
        if folds > 0
    ]
    pd.DataFrame(
        importance_rows,
        columns=["station_id", "stage", "feature", "mean_importance", "evaluated_folds"],
    ).to_csv(importance_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "model": "two_stage_remaining_maximum_hurdle_with_rolling_conformal_distribution",
                "min_train_days": min_train_days,
                "min_calibration_days": min_calibration_days,
                "edge_threshold": edge_threshold,
                "trees": trees,
            "max_trades_per_day_per_mode": max_trades_per_day,
            "include_ncei_pretraining": include_ncei_pretraining,
            "ncei_pretraining_file": (
                str(pretraining_path)
                if include_ncei_pretraining and pretraining_path.exists()
                else None
            ),
                "ncei_source_weight": ncei_source_weight,
                "recency_half_life_years": recency_half_life_years,
                "evaluation_start_date": evaluation_start_date,
                "calibration_seed_file": (
                    str(calibration_seed_path) if calibration_seed_path else None
                ),
                "calibration_seed_rows": sum(
                    len(rows) for rows in calibration_seed_by_station.values()
                ),
                "calibration_exclude_dates": sorted(calibration_exclude_dates),
                "output_tag": output_tag,
                "input_rows": input_rows,
                "eligible_rows": len(data),
                "feature_files": [str(path) for path in feature_files],
                "station_modes": station_summaries,
                "feature_importance_file": str(importance_path),
                "model_readiness_file": str(readiness_path),
                "model_readiness": readiness_rows,
                "warnings": [
                    "All fits, calibration residuals and feature joins are restricted to earlier information.",
                    "Calibrated bucket probabilities use only final-value residuals from earlier out-of-sample station-days plus symmetric finite-sample smoothing.",
                    "The final-maximum support uses the point-in-time WU maximum when available; METAR/KNMI maxima remain predictors only.",
                    "Scheduled and event-driven simulations have separate daily trade limits.",
                    "PnL remains a quote-based research estimate; it is not proof of executable live alpha.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "importance": importance_path,
        "readiness": readiness_path,
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _json_probability_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw_probability in parsed.items():
        probability = _safe_float(raw_probability)
        if probability is not None:
            result[str(key)] = probability
    return result


def _one_value_per_day(frame: Any) -> list[float]:
    if frame.empty:
        return []
    values: list[float] = []
    for _, day in frame.groupby("local_date", sort=True):
        values.append(float(day["remaining_increment_target"].median()))
    return values


def _calibration_residuals(
    history: list[dict[str, Any]],
    local_hour: float | None,
    decision_mode: str,
) -> tuple[list[float], list[float], list[float], int]:
    selected_rows = _calibration_rows(history, local_hour, decision_mode)
    rise: list[float] = []
    increment: list[float] = []
    final: list[float] = []
    for chosen in selected_rows:
        rise_value = _safe_float(chosen.get("rise_residual"))
        increment_value = _safe_float(chosen.get("increment_residual"))
        final_value = _safe_float(chosen.get("final_residual"))
        if rise_value is not None:
            rise.append(rise_value)
        if increment_value is not None:
            increment.append(increment_value)
        if final_value is not None:
            final.append(final_value)
    return rise, increment, final, len(final)


def _calibration_rows(
    history: list[dict[str, Any]],
    local_hour: float | None,
    decision_mode: str,
) -> list[dict[str, Any]]:

    by_day: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        by_day.setdefault(str(item["test_date"]), []).append(item)
    selected: list[dict[str, Any]] = []
    for day in sorted(by_day):
        candidates = [
            item for item in by_day[day] if str(item.get("decision_mode")) == decision_mode
        ] or by_day[day]
        chosen = (
            candidates[-1]
            if local_hour is None
            else min(
                candidates,
                key=lambda item: abs(
                    (_safe_float(item.get("local_hour")) or 12.0) - local_hour
                ),
            )
        )
        selected.append(chosen)
    return selected


def _market_blend_weight(
    history: list[dict[str, Any]],
    local_hour: float | None,
    decision_mode: str,
) -> tuple[float, int]:

    by_day: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        if (
            item.get("probability_calibrated") is True
            and item.get("weather_model_probs")
            and item.get("market_probs")
            and item.get("winner")
        ):
            by_day.setdefault(str(item["test_date"]), []).append(item)
    selected: list[dict[str, Any]] = []
    for day in sorted(by_day):
        candidates = [
            item for item in by_day[day] if str(item.get("decision_mode")) == decision_mode
        ] or by_day[day]
        chosen = (
            candidates[-1]
            if local_hour is None
            else min(
                candidates,
                key=lambda item: abs(
                    (_safe_float(item.get("local_hour")) or 12.0) - local_hour
                ),
            )
        )
        selected.append(chosen)
    if not selected:
        return 0.0, 0
    best_weight = 0.0
    best_score = math.inf
    market_score = math.inf
    for step in range(21):
        weight = step / 20.0
        scores = [
            _brier(
                _blend_distributions(
                    item["market_probs"], item["weather_model_probs"], weight
                ),
                str(item["winner"]),
            )
            for item in selected
        ]
        score = sum(scores) / len(scores)
        if step == 0:
            market_score = score
        if score < best_score - 1e-12:
            best_score = score
            best_weight = weight


    if len(selected) < 20 or market_score - best_score < 0.01:
        best_weight = 0.0
    return best_weight, len(selected)


def _blend_distributions(
    market: dict[str, float], weather: dict[str, float], weather_weight: float
) -> dict[str, float]:
    weather_weight = min(1.0, max(0.0, float(weather_weight)))
    labels = set(market) | set(weather)
    return _normalize(
        {
            label: (1.0 - weather_weight) * market.get(label, 0.0)
            + weather_weight * weather.get(label, 0.0)
            for label in labels
        }
    )


def _summarize(
    station_id: str,
    decision_mode: str,
    rows: list[dict[str, Any]],
    total_days: int,
) -> dict[str, Any]:
    calibrated = [row for row in rows if row["probability_calibrated"]]
    model_brier = _daily_means(calibrated, "model_brier")
    weather_model_brier = _daily_means(calibrated, "weather_model_brier")
    market_brier = _daily_means(calibrated, "market_brier")
    common_days = sorted(set(model_brier) & set(market_brier))
    improvements = [market_brier[day] - model_brier[day] for day in common_days]
    traded = [row for row in rows if row["traded"]]
    pnl_by_day: dict[str, float] = {}
    for row in traded:
        pnl_by_day[str(row["test_date"])] = pnl_by_day.get(str(row["test_date"]), 0.0) + float(
            row["realized_pnl_per_share"]
        )
    return {
        "station_id": station_id,
        "decision_mode": decision_mode,
        "status": "evaluated",
        "days": total_days,
        "evaluated_days": len({row["test_date"] for row in rows}),
        "calibrated_days": len({row["test_date"] for row in calibrated}),
        "rows": len(rows),
        "mae_station_day_weighted": _mean(list(_daily_means(rows, "absolute_error").values())),
        "persistence_mae_station_day_weighted": _mean(
            list(_daily_means(rows, "persistence_absolute_error").values())
        ),
        "forecast_consensus_mae_station_day_weighted": _mean(
            list(_daily_means(rows, "forecast_consensus_absolute_error").values())
        ),
        "mean_model_brier_station_day_weighted": _mean(list(model_brier.values())),
        "mean_weather_model_brier_station_day_weighted": _mean(
            list(weather_model_brier.values())
        ),
        "mean_market_brier_station_day_weighted": _mean(list(market_brier.values())),
        "mean_brier_improvement": _mean(improvements),
        "brier_improvement_ci95": _bootstrap_mean_ci(improvements),
        "market_blend_calibrated_days": len(
            {
                row["test_date"]
                for row in rows
                if row.get("market_blend_calibrated") is True
            }
        ),
        "mean_weather_model_weight": _mean(
            [
                float(row["weather_model_weight"])
                for row in rows
                if row.get("market_blend_calibrated") is True
            ]
        ),
        "trades": len(traded),
        "trade_days": len(pnl_by_day),
        "pnl_per_share_sum": sum(pnl_by_day.values()),
        "mean_pnl_per_station_day": _mean(list(pnl_by_day.values())),
        "mean_pnl_per_station_day_ci95": _bootstrap_mean_ci(list(pnl_by_day.values())),
    }
