from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .master_model import forecast_anchored_conformal_distribution
from .model import _bootstrap_mean_ci, _brier, _safe_float


def evaluate_calibration_challenger(cfg: AppConfig) -> dict[str, Path]:

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies: pip install -e '.[analysis]'") from exc

    validation_paths = sorted(
        cfg.reports_dir.glob(
            "model_validation/master_remaining_max_validation*.csv"
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in validation_paths:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            key = (
                str(row.get("station_id") or ""),
                str(row.get("test_date") or ""),
                str(row.get("decision_mode") or ""),
                str(row.get("received_wall_ns") or ""),
            )
            records[key] = row

    excluded_dates = set(cfg.paper_trading.excluded_calibration_dates)
    station_units = {
        station.station_id: station.market_unit for station in cfg.stations.values()
    }
    rows_by_station_date: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in records.values():
        station_id = str(row.get("station_id") or "")
        test_date = str(row.get("test_date") or "")
        if (
            station_id not in station_units
            or not test_date
            or test_date in excluded_dates
        ):
            continue
        raw_probability = _safe_float(row.get("raw_probability_rise_again"))
        positive_increment = _safe_float(
            row.get("conditional_increment_prediction")
        )
        current_max = _safe_float(row.get("current_running_max"))
        forecast_point = _safe_float(row.get("forecast_consensus_prediction"))
        final_max = _safe_float(row.get("final_wu_max"))
        if None in (
            raw_probability,
            positive_increment,
            current_max,
            forecast_point,
            final_max,
        ):
            continue
        row["_raw_point_final"] = float(current_max) + float(
            raw_probability
        ) * float(positive_increment)
        row["_forecast_point_final"] = max(
            float(current_max), float(forecast_point)
        )
        row["_current_max"] = float(current_max)
        row["_final_max"] = float(final_max)
        rows_by_station_date.setdefault(station_id, {}).setdefault(
            test_date, []
        ).append(row)

    comparison_rows: list[dict[str, Any]] = []
    for station_id, by_date in sorted(rows_by_station_date.items()):
        calibration_history: list[dict[str, Any]] = []
        for test_date in sorted(by_date):
            date_rows = sorted(
                by_date[test_date],
                key=lambda row: int(float(row.get("received_wall_ns") or 0)),
            )
            for row in date_rows:
                incumbent = _json_probabilities(
                    row.get("weather_model_distribution_json")
                )
                winner = str(row.get("winning_bucket") or "")
                if not incumbent or not winner:
                    continue
                challenger, diagnostics = forecast_anchored_conformal_distribution(
                    row["_current_max"],
                    row["_raw_point_final"],
                    row["_forecast_point_final"],
                    calibration_history,
                    list(incumbent),
                    station_units[station_id],
                    local_hour=_safe_float(row.get("local_hour")),
                    decision_mode=str(
                        row.get("decision_mode") or "event_driven"
                    ),
                )
                accepted = bool(
                    diagnostics.get("challenger_accepted") and challenger
                )
                selected = challenger if accepted else incumbent
                incumbent_probability = max(
                    incumbent.get(winner, 0.0), 1e-12
                )
                selected_probability = max(selected.get(winner, 0.0), 1e-12)
                comparison_rows.append(
                    {
                        "station_id": station_id,
                        "test_date": test_date,
                        "decision_mode": row.get("decision_mode"),
                        "received_wall_ns": row.get("received_wall_ns"),
                        "local_hour": row.get("local_hour"),
                        "winning_bucket": winner,
                        "calibration_days": diagnostics.get(
                            "calibration_days"
                        ),
                        "challenger_accepted": accepted,
                        "raw_model_weight": diagnostics.get(
                            "raw_model_weight"
                        ),
                        "model_weight": diagnostics.get("model_weight"),
                        "raw_point_final": row["_raw_point_final"],
                        "forecast_point_final": row[
                            "_forecast_point_final"
                        ],
                        "blended_point_final": diagnostics.get(
                            "blended_point_final"
                        ),
                        "raw_rmse": diagnostics.get("raw_rmse"),
                        "blended_rmse": diagnostics.get("blended_rmse"),
                        "relative_rmse_improvement": diagnostics.get(
                            "relative_rmse_improvement"
                        ),
                        "incumbent_brier": _brier(incumbent, winner),
                        "challenger_brier": _brier(selected, winner),
                        "brier_improvement": _brier(incumbent, winner)
                        - _brier(selected, winner),
                        "incumbent_log_loss": -math.log(
                            incumbent_probability
                        ),
                        "challenger_log_loss": -math.log(
                            selected_probability
                        ),
                        "log_loss_improvement": math.log(
                            selected_probability / incumbent_probability
                        ),
                        "incumbent_distribution_json": json.dumps(
                            incumbent, sort_keys=True
                        ),
                        "challenger_distribution_json": json.dumps(
                            selected, sort_keys=True
                        ),
                    }
                )
            calibration_history.extend(
                {
                    "test_date": test_date,
                    "decision_mode": str(
                        row.get("decision_mode") or "event_driven"
                    ),
                    "local_hour": _safe_float(row.get("local_hour")),
                    "raw_point_final": row["_raw_point_final"],
                    "forecast_point_final": row[
                        "_forecast_point_final"
                    ],
                    "current_max": row["_current_max"],
                    "final_max": row["_final_max"],
                }
                for row in date_rows
            )

    output_dir = cfg.reports_dir / "calibration_challenger"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "chronological_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    summary = _summary(comparison_rows)
    summary.update(
        {
            "strategy": "forecast_anchored_conformal_v2",
            "selection_rule": (
                "Use the challenger only when prior OOS blended RMSE improves "
                "on raw-model RMSE by at least 2%."
            ),
            "unit_of_independence": "station_id + test_date",
            "excluded_dates": sorted(excluded_dates),
            "validation_files": [str(path) for path in validation_paths],
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"comparison": comparison_path, "summary": summary_path}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["station_id"]), str(row["test_date"])), []
        ).append(row)
    daily = [
        {
            "station_id": station_id,
            "test_date": test_date,
            "incumbent_brier": _mean(
                [float(row["incumbent_brier"]) for row in values]
            ),
            "challenger_brier": _mean(
                [float(row["challenger_brier"]) for row in values]
            ),
            "incumbent_log_loss": _mean(
                [float(row["incumbent_log_loss"]) for row in values]
            ),
            "challenger_log_loss": _mean(
                [float(row["challenger_log_loss"]) for row in values]
            ),
            "acceptance_rate": _mean(
                [1.0 if row["challenger_accepted"] else 0.0 for row in values]
            ),
        }
        for (station_id, test_date), values in sorted(grouped.items())
    ]
    by_station = []
    for station_id in sorted({row["station_id"] for row in daily}):
        station_days = [
            row for row in daily if row["station_id"] == station_id
        ]
        improvements = [
            row["incumbent_brier"] - row["challenger_brier"]
            for row in station_days
        ]
        log_improvements = [
            row["incumbent_log_loss"] - row["challenger_log_loss"]
            for row in station_days
        ]
        by_station.append(
            {
                "station_id": station_id,
                "station_days": len(station_days),
                "mean_incumbent_brier": _mean(
                    [row["incumbent_brier"] for row in station_days]
                ),
                "mean_challenger_brier": _mean(
                    [row["challenger_brier"] for row in station_days]
                ),
                "mean_brier_improvement": _mean(improvements),
                "mean_brier_improvement_ci95": _bootstrap_mean_ci(
                    improvements
                ),
                "mean_log_loss_improvement": _mean(log_improvements),
                "acceptance_rate": _mean(
                    [row["acceptance_rate"] for row in station_days]
                ),
            }
        )
    all_improvements = [
        row["incumbent_brier"] - row["challenger_brier"] for row in daily
    ]
    return {
        "rows": len(rows),
        "independent_station_days": len(daily),
        "mean_incumbent_brier": _mean(
            [row["incumbent_brier"] for row in daily]
        ),
        "mean_challenger_brier": _mean(
            [row["challenger_brier"] for row in daily]
        ),
        "mean_brier_improvement": _mean(all_improvements),
        "mean_brier_improvement_ci95": _bootstrap_mean_ci(
            all_improvements
        ),
        "by_station": by_station,
    }


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
        if (probability := _safe_float(raw)) is not None
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
