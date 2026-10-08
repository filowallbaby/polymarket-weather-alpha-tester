from __future__ import annotations

import json
import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from ..parsing.buckets import c_to_f, parse_temperature_bucket
from .model import _bootstrap_mean_ci
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import _load_validation_rows, _probabilities, _timestamp_ns
from .station_adaptive_alpha import (
    STATION_ADAPTIVE_LANES,
    _recent_dates_by_station,
)
from .station_specific_alpha import (
    _load_csv,
    _load_historical_snapshots,
    _read_json,
    _settle_trades,
    _truth,
    evaluate_station_policy_snapshot,
)
from .station_tail_diagnostics import (
    EDGE_BANDS,
    _build_candidates,
    _load_feature_rows,
    _upper_tail_probability,
)

MODEL_SCHEMA_VERSION = 1
MODEL_REGULARIZATION_C = 0.10
MINIMUM_ROLLING_TRAINING_CANDIDATES = 60
PRIMARY_MINIMUM_EDGE = 0.15
PRIMARY_MAXIMUM_EDGE = 0.20
ALLOWED_DECISION_MODES = {"event_driven", "forecast_update", "scheduled"}


HARD_SOURCE_REJECTIONS = {
    "incomplete_day_history",
    "continuity_warmup_or_gap",
    "market_websocket_unhealthy",
    "stale_primary_observation",
    "insufficient_forecast_providers",
    "stale_forecast_consensus",
    "insufficient_peer_station_coverage",
    "missing_resolution_provider_running_max",
    "missing_bucket_vocabulary",
    "missing_weather_probability_distribution",
    "incomplete_market_probability_state",
    "insufficient_prior_oos_calibration_days",
}

NUMERIC_FEATURES = (
    "market_logit",
    "weather_logit",
    "lane_logit",
    "market_minus_weather_upper_tail",
    "forecast_minus_candidate_high",
    "forecast_spread_remaining_max_c",
    "candidate_high_minus_current_max",
    "raw_probability_rise_again",
    "hours_since_running_max",
    "peer_temperature_change_1h_mean",
)
CATEGORICAL_FEATURES = ("station_id", "policy_id", "local_hour_category")


@dataclass(slots=True)
class StationTailV3Bundle:
    schema_version: int
    deployed_at_ns: int
    trained_through_local_date: str
    training_candidates: int
    regularization_c: float
    candidate_model: Any
    upper_conditional_model: Any
    upper_conditional_fallback: float
    lane_probabilities: dict[str, float]
    lane_candidate_counts: dict[str, int]


def evaluate_station_tail_v3(cfg: AppConfig) -> dict[str, Path]:

    candidates = _historical_candidates(cfg)
    predictions = _rolling_predictions(candidates)
    sensitivity = _historical_sensitivity(predictions)
    primary_trades = _replay_prediction_band(
        predictions,
        PRIMARY_MINIMUM_EDGE,
        PRIMARY_MAXIMUM_EDGE,
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "strategy": "station_tail_snapshot_calibrator_v3",
        "training_rule": (
            "For every test date, fit only candidates from strictly earlier "
            "resolved dates; start after 60 prior eligible candidates."
        ),
        "model": {
            "candidate_target": "candidate bucket wins",
            "upper_target": "winner is above candidate, conditional on a miss",
            "regularization_c": MODEL_REGULARIZATION_C,
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "station_specificity": (
                "Shared regularized slopes with airport, lane and local-hour "
                "intercepts. The underlying remaining-maximum models remain "
                "airport-specific. Airport-by-feature interactions were rejected "
                "because rolling Brier worsened."
            ),
        },
        "rolling_predictions": len(predictions),
        "probability_scores": _probability_scores(predictions),
        "primary_15_20_replay": _summarize_prediction_trades(primary_trades),
        "selection_warning": (
            "C=0.10 was selected by rolling probability score, not by P&L. "
            "The same history is still diagnostic; only post-deployment rows "
            "are prospective."
        ),
        "economic_warning": (
            "Historical prices are one-minute probability proxies plus fixed "
            "station cost. This is not executable bid/ask/fill P&L."
        ),
        "live_orders_enabled": False,
    }

    output_dir = cfg.reports_dir / "station_tail_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "rolling_predictions.csv"
    trades_path = output_dir / "rolling_primary_15_20_trades.csv"
    sensitivity_path = output_dir / "rolling_edge_sensitivity.csv"
    summary_path = output_dir / "summary.json"
    _write_csv_atomic(predictions_path, predictions)
    _write_csv_atomic(trades_path, primary_trades)
    _write_csv_atomic(sensitivity_path, sensitivity)
    _write_json_atomic(summary_path, summary)
    return {
        "summary": summary_path,
        "predictions": predictions_path,
        "trades": trades_path,
        "sensitivity": sensitivity_path,
    }


def deploy_station_tail_v3(cfg: AppConfig) -> dict[str, Path]:

    diagnostics = evaluate_station_tail_v3(cfg)
    candidates = _historical_candidates(cfg)
    deployed_at_ns = time.time_ns()
    bundle = _fit_bundle(candidates, deployed_at_ns)

    output_dir = cfg.reports_dir / "paper_trading_station_tail_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_bundle.pkl"
    temporary_model_path = model_path.with_suffix(".pkl.tmp")
    temporary_model_path.write_bytes(
        pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    )
    temporary_model_path.replace(model_path)

    deployment_path = output_dir / "deployment.json"
    state_path = output_dir / "state.json"
    _write_json_atomic(
        deployment_path,
        {
            "strategy": "station_tail_snapshot_calibrator_v3",
            "schema_version": MODEL_SCHEMA_VERSION,
            "deployed_at_ns": deployed_at_ns,
            "deployed_at_utc": datetime.fromtimestamp(
                deployed_at_ns / 1e9, tz=UTC
            ).isoformat(),
            "trained_through_local_date": bundle.trained_through_local_date,
            "training_candidates": bundle.training_candidates,
            "regularization_c": bundle.regularization_c,
            "minimum_edge_inclusive": PRIMARY_MINIMUM_EDGE,
            "maximum_edge_exclusive": PRIMARY_MAXIMUM_EDGE,
            "accepted_decision_modes": sorted(ALLOWED_DECISION_MODES),
            "maximum_entries_per_station_day": 1,
            "hard_source_rejections": sorted(HARD_SOURCE_REJECTIONS),
            "forecast_spread_handling": (
                "continuous model feature and audit warning, not a tuned hard veto"
            ),
            "source_diagnostics": str(diagnostics["summary"]),
            "processing_warning": (
                "Decision timestamps are preserved, but the scheduled paper cycle "
                "can process them later. Processing lag is recorded per row."
            ),
            "synthetic_cost_only": True,
            "live_orders_enabled": False,
        },
    )
    _write_json_atomic(
        state_path,
        {
            "deployed_at_ns": deployed_at_ns,
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    _write_csv_atomic(output_dir / "paper_audit.csv", [])
    _write_csv_atomic(output_dir / "paper_trades.csv", [])
    return {
        "deployment": deployment_path,
        "model": model_path,
        "state": state_path,
        "diagnostics": diagnostics["summary"],
    }


def run_station_tail_v3_paper_cycle(cfg: AppConfig) -> dict[str, Path]:

    output_dir = cfg.reports_dir / "paper_trading_station_tail_v3"
    deployment_path = output_dir / "deployment.json"
    model_path = output_dir / "model_bundle.pkl"
    state_path = output_dir / "state.json"
    source_path = (
        cfg.reports_dir
        / "paper_trading_challenger"
        / "paper_signal_audit.csv"
    )
    if not deployment_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Missing V3 deployment; run deploy-station-tail-v3"
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}; run paper-challenger-cycle"
        )

    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    bundle = pickle.loads(model_path.read_bytes())
    if not isinstance(bundle, StationTailV3Bundle):
        raise TypeError("Invalid V3 model bundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise RuntimeError("V3 model schema is incompatible; redeploy it")

    state = _read_json(
        state_path,
        {
            "deployed_at_ns": deployment["deployed_at_ns"],
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    audit_path = output_dir / "paper_audit.csv"
    trades_path = output_dir / "paper_trades.csv"
    audit_rows = _load_csv(audit_path)
    trade_rows = _load_csv(trades_path)
    processed = set(state.get("processed_decision_ids") or [])
    traded_days = set(state.get("traded_station_days") or [])
    deployed_at_ns = int(deployment["deployed_at_ns"])
    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    feature_rows = _load_feature_rows(cfg)
    cycle_started_ns = time.time_ns()

    lanes_by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for station_id, lanes in STATION_ADAPTIVE_LANES.items():
        lanes_by_station[station_id].extend(lanes)

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
        matching_lanes = [
            lane
            for lane in lanes_by_station[station_id]
            if local_dt.hour == int(lane["target_local_hour"])
        ]
        evaluations: list[dict[str, Any]] = []

        for lane in matching_lanes:
            evaluation = evaluate_station_policy_snapshot(
                lane, weather, market
            )
            reasons = list(evaluation["rejection_reasons"])
            if local_dt.minute > int(lane["latest_local_minute"]):
                reasons.append("after_station_entry_window")
            source_rejections = {
                item
                for item in str(
                    source_row.get("rejection_reasons") or ""
                ).split("|")
                if item
            }
            reasons.extend(sorted(source_rejections & HARD_SOURCE_REJECTIONS))
            candidate = None
            prediction = None
            if evaluation["candidate_bucket"]:
                candidate = _live_candidate(
                    source_row,
                    feature_row,
                    station,
                    lane,
                    evaluation,
                    weather,
                    market,
                    bundle,
                    timestamp_ns,
                    local_date,
                )
                prediction = _predict_candidate(bundle, candidate, station)
            if candidate is None or prediction is None:
                reasons.append("v3_candidate_features_unavailable")
                edge = None
            else:
                edge = (
                    float(prediction["candidate_probability"])
                    - float(candidate["synthetic_cost_per_share"])
                )
                if edge < PRIMARY_MINIMUM_EDGE:
                    reasons.append("edge_below_v3_primary_band")
                if edge >= PRIMARY_MAXIMUM_EDGE:
                    reasons.append("edge_above_v3_primary_band")
                if day_key in traded_days:
                    reasons.append("station_day_already_traded")
            evaluations.append(
                {
                    "lane": lane,
                    "evaluation": evaluation,
                    "candidate": candidate,
                    "prediction": prediction,
                    "edge": edge,
                    "reasons": sorted(set(reasons)),
                }
            )

        eligible = [
            item for item in evaluations if not item["reasons"]
        ]
        chosen = (
            max(eligible, key=lambda item: float(item["edge"]))
            if eligible
            else None
        )
        processing_lag_seconds = max(
            0.0, (cycle_started_ns - timestamp_ns) / 1e9
        )

        if not evaluations:
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
                    "processing_lag_seconds": processing_lag_seconds,
                    "paper_trade_opened": False,
                    "rejection_reasons": "outside_v3_station_entry_hour",
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
                reasons.append("lower_edge_than_selected_lane")
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
                    "processing_lag_seconds": processing_lag_seconds,
                    "policy_id": item["lane"]["policy_id"],
                    "candidate_bucket": item["evaluation"].get(
                        "candidate_bucket"
                    ),
                    "candidate_price_proxy": candidate.get(
                        "candidate_price_proxy"
                    ),
                    "synthetic_cost_per_share": candidate.get(
                        "synthetic_cost_per_share"
                    ),
                    "raw_snapshot_probability": candidate.get(
                        "raw_snapshot_probability"
                    ),
                    "lane_probability": candidate.get(
                        "chronological_lane_probability"
                    ),
                    "v3_candidate_probability": prediction.get(
                        "candidate_probability"
                    ),
                    "v3_upper_miss_probability": prediction.get(
                        "upper_probability"
                    ),
                    "v3_lower_miss_probability": prediction.get(
                        "lower_probability"
                    ),
                    "v3_distribution_json": prediction.get(
                        "distribution_json"
                    ),
                    "v3_estimated_edge": item["edge"],
                    "forecast_spread_remaining_max_c": candidate.get(
                        "forecast_spread_remaining_max_c"
                    ),
                    "forecast_spread_above_configured_limit": (
                        candidate.get(
                            "forecast_spread_above_configured_limit"
                        )
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
                "processing_lag_seconds": processing_lag_seconds,
                "bucket": chosen["evaluation"]["candidate_bucket"],
                "price_proxy": candidate["candidate_price_proxy"],
                "fixed_cost_per_share": candidate["fixed_cost_per_share"],
                "synthetic_cost_per_share": candidate[
                    "synthetic_cost_per_share"
                ],
                "v3_candidate_probability": prediction[
                    "candidate_probability"
                ],
                "v3_upper_miss_probability": prediction[
                    "upper_probability"
                ],
                "v3_lower_miss_probability": prediction[
                    "lower_probability"
                ],
                "estimated_synthetic_edge": chosen["edge"],
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
    summary_path = output_dir / "paper_summary.json"
    settled = [row for row in trade_rows if _truth(row.get("settled"))]
    _write_json_atomic(
        summary_path,
        {
            "strategy": deployment["strategy"],
            "deployed_at_ns": deployed_at_ns,
            "trained_through_local_date": (
                bundle.trained_through_local_date
            ),
            "accepted_decision_modes": sorted(ALLOWED_DECISION_MODES),
            "minimum_edge_inclusive": PRIMARY_MINIMUM_EDGE,
            "maximum_edge_exclusive": PRIMARY_MAXIMUM_EDGE,
            "processed_decisions": len(processed),
            "audit_rows": len(audit_rows),
            "paper_trades": len(trade_rows),
            "settled_trades": len(settled),
            "wins": sum(_truth(row.get("won")) for row in settled),
            "synthetic_value_per_share_total": sum(
                float(row["synthetic_value_per_share"])
                for row in settled
            ),
            "mean_processing_lag_seconds": _mean(
                [
                    _safe_float(row.get("processing_lag_seconds"))
                    for row in audit_rows
                ]
            ),
            "synthetic_cost_only": True,
            "live_orders_enabled": False,
        },
    )
    return {
        "audit": audit_path,
        "trades": trades_path,
        "summary": summary_path,
        "state": state_path,
    }


def _historical_candidates(cfg: AppConfig) -> list[dict[str, Any]]:
    snapshots = _load_historical_snapshots(cfg)
    validation_rows, _ = _load_validation_rows(cfg)
    return _build_candidates(
        cfg,
        snapshots,
        _recent_dates_by_station(snapshots),
        validation_rows,
        _load_feature_rows(cfg),
    )


def _rolling_predictions(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    dates = sorted({str(row["local_date"]) for row in candidates})
    for test_date in dates:
        training = [
            row for row in candidates if str(row["local_date"]) < test_date
        ]
        testing = [
            row for row in candidates if str(row["local_date"]) == test_date
        ]
        if len(training) < MINIMUM_ROLLING_TRAINING_CANDIDATES:
            continue
        bundle = _fit_bundle(training, deployed_at_ns=0)
        for row in testing:
            station = _station_stub(row)
            prediction = _predict_candidate(bundle, row, station)
            probability = float(prediction["candidate_probability"])
            result = dict(row)
            result.update(
                {
                    "rolling_training_candidates": len(training),
                    "v3_candidate_probability": probability,
                    "v3_upper_miss_probability": prediction[
                        "upper_probability"
                    ],
                    "v3_lower_miss_probability": prediction[
                        "lower_probability"
                    ],
                    "v3_estimated_edge": (
                        probability
                        - float(row["synthetic_cost_per_share"])
                    ),
                    "synthetic_value_per_share": (
                        (1.0 if _truth(row["won"]) else 0.0)
                        - float(row["synthetic_cost_per_share"])
                    ),
                }
            )
            predictions.append(result)
    return predictions


def _fit_bundle(
    candidates: list[dict[str, Any]], deployed_at_ns: int
) -> StationTailV3Bundle:
    if len(candidates) < MINIMUM_ROLLING_TRAINING_CANDIDATES:
        raise RuntimeError("Insufficient historical candidates for V3")
    candidate_model = _fit_binary_model(
        candidates, [_truth(row["won"]) for row in candidates]
    )
    misses = [row for row in candidates if not _truth(row["won"])]
    upper_targets = [
        _outcome_class(row) == "upper" for row in misses
    ]
    if len(misses) >= 20 and len(set(upper_targets)) == 2:
        upper_model = _fit_binary_model(misses, upper_targets)
    else:
        upper_model = None
    upper_fallback = (
        sum(upper_targets) / len(upper_targets) if upper_targets else 0.5
    )

    lane_probabilities: dict[str, float] = {}
    lane_counts: dict[str, int] = {}
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_lane[str(row["policy_id"])].append(row)
    for lane_id, rows in by_lane.items():
        wins = sum(_truth(row["won"]) for row in rows)
        raw_center = sum(
            float(row["raw_snapshot_probability"]) for row in rows
        ) / len(rows)
        lane_probabilities[lane_id] = (
            wins + 2.0 * raw_center
        ) / (len(rows) + 2.0)
        lane_counts[lane_id] = len(rows)

    return StationTailV3Bundle(
        schema_version=MODEL_SCHEMA_VERSION,
        deployed_at_ns=deployed_at_ns,
        trained_through_local_date=max(
            str(row["local_date"]) for row in candidates
        ),
        training_candidates=len(candidates),
        regularization_c=MODEL_REGULARIZATION_C,
        candidate_model=candidate_model,
        upper_conditional_model=upper_model,
        upper_conditional_fallback=upper_fallback,
        lane_probabilities=lane_probabilities,
        lane_candidate_counts=lane_counts,
    )


def _fit_binary_model(rows: list[dict[str, Any]], targets: list[bool]):
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Install analysis dependencies: pip install -e '.[analysis]'"
        ) from exc

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "logistic",
                LogisticRegression(
                    C=MODEL_REGULARIZATION_C,
                    max_iter=3000,
                    random_state=20260729,
                ),
            ),
        ]
    )
    frame = pd.DataFrame([_model_record(row) for row in rows])
    model.fit(frame, [int(value) for value in targets])
    return model


def _predict_candidate(
    bundle: StationTailV3Bundle,
    candidate: dict[str, Any],
    station: StationConfig | Any,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies") from exc

    frame = pd.DataFrame([_model_record(candidate)])
    candidate_probability = _positive_probability(
        bundle.candidate_model, frame
    )
    if bundle.upper_conditional_model is not None:
        conditional_upper = _positive_probability(
            bundle.upper_conditional_model, frame
        )
    else:
        conditional_upper = bundle.upper_conditional_fallback
    miss_probability = max(0.0, 1.0 - candidate_probability)
    upper_probability = miss_probability * conditional_upper
    lower_probability = miss_probability - upper_probability
    distribution = _recalibrated_distribution(
        candidate,
        candidate_probability,
        upper_probability,
        lower_probability,
        station.market_unit,
    )
    return {
        "candidate_probability": candidate_probability,
        "upper_probability": upper_probability,
        "lower_probability": lower_probability,
        "conditional_upper_given_miss": conditional_upper,
        "distribution_json": json.dumps(
            distribution, sort_keys=True, ensure_ascii=False
        ),
    }


def _positive_probability(model: Any, frame: Any) -> float:
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.named_steps["logistic"].classes_)
    positive_index = classes.index(1)
    return float(probabilities[positive_index])


def _model_record(row: dict[str, Any]) -> dict[str, Any]:
    lane_probability = _safe_float(
        row.get("chronological_lane_probability")
    )
    if lane_probability is None:
        lane_probability = _safe_float(row.get("fixed_lane_probability"))
    return {
        "market_logit": _logit(row.get("candidate_price_proxy")),
        "weather_logit": _logit(row.get("raw_snapshot_probability")),
        "lane_logit": _logit(lane_probability),
        "market_minus_weather_upper_tail": row.get(
            "market_minus_weather_upper_tail"
        ),
        "forecast_minus_candidate_high": row.get(
            "forecast_minus_candidate_high"
        ),
        "forecast_spread_remaining_max_c": row.get(
            "forecast_spread_remaining_max_c"
        ),
        "candidate_high_minus_current_max": row.get(
            "candidate_high_minus_current_max"
        ),
        "raw_probability_rise_again": row.get(
            "raw_probability_rise_again"
        ),
        "hours_since_running_max": row.get("hours_since_running_max"),
        "peer_temperature_change_1h_mean": row.get(
            "peer_temperature_change_1h_mean"
        ),
        "station_id": str(row.get("station_id") or ""),
        "policy_id": str(row.get("policy_id") or ""),
        "local_hour_category": str(
            int(float(row.get("local_hour") or 0))
        ),
    }


def _live_candidate(
    source_row: dict[str, Any],
    feature_row: dict[str, Any],
    station: StationConfig,
    lane: dict[str, Any],
    evaluation: dict[str, Any],
    weather: dict[str, float],
    market: dict[str, float],
    bundle: StationTailV3Bundle,
    timestamp_ns: int,
    local_date: str,
) -> dict[str, Any] | None:
    candidate_bucket = str(evaluation.get("candidate_bucket") or "")
    try:
        bucket = parse_temperature_bucket(
            candidate_bucket, station.market_unit
        )
    except ValueError:
        return None
    if bucket.high is None:
        return None
    lane_id = str(lane["policy_id"])
    lane_probability = bundle.lane_probabilities.get(lane_id)
    if lane_probability is None:
        return None

    current_max = _safe_float(
        source_row.get("current_resolution_running_max")
    )
    if current_max is None:
        current_max = _safe_float(
            feature_row.get("resolution_running_max_market_unit")
        )
    forecast_c = _safe_float(
        feature_row.get("model_consensus_today_max_c")
    )
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
    peer_change = _safe_float(
        feature_row.get("peer_mean_slope_60m_c_per_hour")
    )
    if peer_change is None:
        peer_change = _safe_float(
            feature_row.get("peer_temperature_change_1h_mean")
        )
    price = float(evaluation["candidate_price_proxy"])
    fixed_cost = float(lane["fixed_cost_per_share"])
    return {
        "station_id": station.station_id,
        "local_date": local_date,
        "local_hour": datetime.fromtimestamp(
            timestamp_ns / 1e9, tz=UTC
        ).astimezone(ZoneInfo(station.timezone)).hour,
        "policy_id": lane_id,
        "candidate_bucket": candidate_bucket,
        "candidate_price_proxy": price,
        "fixed_cost_per_share": fixed_cost,
        "synthetic_cost_per_share": price + fixed_cost,
        "raw_snapshot_probability": float(
            evaluation["candidate_weather_probability"]
        ),
        "chronological_lane_probability": lane_probability,
        "market_upper_tail_probability": _upper_tail_probability(
            market, candidate_bucket, station.market_unit
        ),
        "weather_upper_tail_probability": _upper_tail_probability(
            weather, candidate_bucket, station.market_unit
        ),
        "market_minus_weather_upper_tail": (
            _upper_tail_probability(
                market, candidate_bucket, station.market_unit
            )
            - _upper_tail_probability(
                weather, candidate_bucket, station.market_unit
            )
        ),
        "forecast_minus_candidate_high": (
            forecast_market - bucket.high
            if forecast_market is not None
            else None
        ),
        "forecast_spread_remaining_max_c": spread,
        "candidate_high_minus_current_max": (
            bucket.high - current_max if current_max is not None else None
        ),
        "raw_probability_rise_again": _safe_float(
            source_row.get("raw_probability_rise_again")
        ),
        "hours_since_running_max": _safe_float(
            feature_row.get("hours_since_running_max")
        ),
        "peer_temperature_change_1h_mean": peer_change,
        "weather_distribution": weather,
        "market_distribution": market,
        "forecast_spread_above_configured_limit": (
            spread is not None
            and spread > 3.0
        ),
    }


def _recalibrated_distribution(
    candidate: dict[str, Any],
    candidate_probability: float,
    upper_probability: float,
    lower_probability: float,
    unit: str,
) -> dict[str, float]:
    candidate_label = str(candidate["candidate_bucket"])
    weather = {
        str(label): float(value)
        for label, value in (
            candidate.get("weather_distribution") or {}
        ).items()
    }
    market = {
        str(label): float(value)
        for label, value in (
            candidate.get("market_distribution") or {}
        ).items()
    }
    labels = sorted(set(weather) | set(market))
    if candidate_label not in labels:
        labels.append(candidate_label)
    candidate_bucket = parse_temperature_bucket(candidate_label, unit)
    upper: list[str] = []
    lower: list[str] = []
    for label in labels:
        if label == candidate_label:
            continue
        try:
            bucket = parse_temperature_bucket(label, unit)
        except ValueError:
            continue
        if (
            candidate_bucket.high is not None
            and bucket.low is not None
            and bucket.low > candidate_bucket.high
        ):
            upper.append(label)
        elif (
            candidate_bucket.low is not None
            and bucket.high is not None
            and bucket.high < candidate_bucket.low
        ):
            lower.append(label)

    result = {label: 0.0 for label in labels}
    result[candidate_label] = candidate_probability
    _allocate_side(result, upper, upper_probability, weather, market)
    _allocate_side(result, lower, lower_probability, weather, market)
    unallocated = 1.0 - sum(result.values())
    result[candidate_label] += unallocated
    return {
        label: max(0.0, min(1.0, value))
        for label, value in result.items()
    }


def _allocate_side(
    result: dict[str, float],
    labels: list[str],
    probability: float,
    weather: dict[str, float],
    market: dict[str, float],
) -> None:
    if not labels:
        return
    weights = {
        label: math.sqrt(
            max(weather.get(label, 0.0), 1e-6)
            * max(market.get(label, 0.0), 1e-6)
        )
        for label in labels
    }
    total = sum(weights.values())
    for label in labels:
        result[label] = probability * weights[label] / total


def _outcome_class(row: dict[str, Any]) -> str:
    unit = "F" if "°F" in str(row["candidate_bucket"]) else "C"
    try:
        candidate = parse_temperature_bucket(
            str(row["candidate_bucket"]), unit
        )
        winner = parse_temperature_bucket(
            str(row["winning_bucket"]), unit
        )
    except ValueError:
        return "candidate" if _truth(row.get("won")) else "upper"
    if (
        candidate.high is not None
        and winner.low is not None
        and winner.low > candidate.high
    ):
        return "upper"
    if (
        candidate.low is not None
        and winner.high is not None
        and winner.high < candidate.low
    ):
        return "lower"
    return "candidate"


def _replay_prediction_band(
    predictions: list[dict[str, Any]],
    minimum_edge: float,
    maximum_edge: float,
) -> list[dict[str, Any]]:
    by_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_day[(str(row["station_id"]), str(row["local_date"]))].append(
            row
        )
    trades: list[dict[str, Any]] = []
    for rows in by_day.values():
        eligible = [
            row
            for row in rows
            if minimum_edge
            <= float(row["v3_estimated_edge"])
            < maximum_edge
        ]
        if not eligible:
            continue
        first_hour = min(int(row["local_hour"]) for row in eligible)
        chosen = max(
            (
                row
                for row in eligible
                if int(row["local_hour"]) == first_hour
            ),
            key=lambda row: float(row["v3_estimated_edge"]),
        )
        trades.append(dict(chosen))
    return sorted(
        trades,
        key=lambda row: (str(row["local_date"]), str(row["station_id"])),
    )


def _historical_sensitivity(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minimum_edge, maximum_edge in EDGE_BANDS:
        trades = _replay_prediction_band(
            predictions, minimum_edge, maximum_edge
        )
        summary = _trade_summary_block(trades)
        development = _trade_summary_block(
            [
                row
                for row in trades
                if row["split"] == "development_diagnostic"
            ]
        )
        recent = _trade_summary_block(
            [
                row
                for row in trades
                if row["split"] == "recent_diagnostic"
            ]
        )
        rows.append(
            {
                "minimum_edge_inclusive": minimum_edge,
                "maximum_edge_exclusive": maximum_edge,
                **summary,
                "development_trades": development["trades"],
                "development_value": development["synthetic_value_total"],
                "recent_trades": recent["trades"],
                "recent_value": recent["synthetic_value_total"],
            }
        )
    return rows


def _probability_scores(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = [1.0 if _truth(row["won"]) else 0.0 for row in predictions]

    def score(field: str) -> dict[str, float | int | None]:
        probabilities = [
            max(1e-9, min(1.0 - 1e-9, float(row[field])))
            for row in predictions
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
                    math.log(
                        probability
                        if target
                        else 1.0 - probability
                    )
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
        "raw_weather_candidate": score("raw_snapshot_probability"),
        "chronological_lane_probability": score(
            "chronological_lane_probability"
        ),
        "market_candidate_probability": score("candidate_price_proxy"),
        "v3_candidate_probability": score("v3_candidate_probability"),
    }


def _summarize_prediction_trades(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "all_available": _trade_summary_block(trades),
        "development_diagnostic": _trade_summary_block(
            [
                row
                for row in trades
                if row["split"] == "development_diagnostic"
            ]
        ),
        "recent_diagnostic": _trade_summary_block(
            [
                row
                for row in trades
                if row["split"] == "recent_diagnostic"
            ]
        ),
        "by_station": {
            station_id: _trade_summary_block(
                [
                    row
                    for row in trades
                    if row["station_id"] == station_id
                ]
            )
            for station_id in STATION_ADAPTIVE_LANES
        },
    }


def _trade_summary_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["synthetic_value_per_share"]) for row in rows
    ]
    stake = sum(
        float(row["synthetic_cost_per_share"]) for row in rows
    )
    total = sum(values)
    interval = _bootstrap_mean_ci(values) if values else None
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
        "mean_value_ci95_low": interval["low"] if interval else None,
        "mean_value_ci95_high": interval["high"] if interval else None,
    }


def _station_stub(row: dict[str, Any]) -> Any:
    class StationStub:
        market_unit = (
            "F" if "°F" in str(row.get("candidate_bucket")) else "C"
        )

    return StationStub()


def _logit(value: Any) -> float:
    probability = _safe_float(value)
    if probability is None:
        probability = 0.5
    probability = min(0.999, max(0.001, probability))
    return math.log(probability / (1.0 - probability))


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None
