from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .model import _bootstrap_mean_ci, _safe_float


def evaluate_historical_price_screen(
    cfg: AppConfig,
    *,
    validation_path: Path | None = None,
    edge_threshold: float = 0.06,
    execution_buffer: float = 0.02,
    minimum_model_probability: float = 0.05,
    minimum_market_price: float = 0.02,
    maximum_market_price: float = 0.95,
    maximum_price_age_seconds: float = 600.0,
    use_station_live_costs: bool = True,
    output_name: str = "historical_price_screen",
) -> dict[str, Path]:

    validation_path = validation_path or (
        cfg.reports_dir / "model_validation" / "master_remaining_max_validation.csv"
    )
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing {validation_path}; run evaluate-master-model after backfill-history"
        )
    with validation_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("decision_mode") == "historical_scheduled"
            and _truth(row.get("market_price_history_only"))
        ]
    rows.sort(key=lambda row: int(float(row["received_wall_ns"])))
    if not rows:
        raise RuntimeError("Master validation contains no calibrated historical-price rows")

    station_buffers: dict[str, float] = {}
    live_cost_path = cfg.reports_dir / "execution_costs" / "live_execution_cost_summary.json"
    if use_station_live_costs and live_cost_path.exists():
        try:
            live_costs = json.loads(live_cost_path.read_text(encoding="utf-8"))
            station_buffers = {
                str(item["station_id"]): max(
                    execution_buffer,
                    float(item["vwap_cost_from_mid"]["p90"]),
                )
                for item in live_costs.get("by_station", [])
                if item.get("station_id") and item.get("vwap_cost_from_mid", {}).get("p90")
                is not None
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            station_buffers = {}

    evaluated_days: set[tuple[str, str]] = set()
    calibrated_days: set[tuple[str, str]] = set()
    traded_days: set[tuple[str, str]] = set()
    trades: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["station_id"]), str(row["test_date"]))
        evaluated_days.add(key)
        if not _truth(row.get("probability_calibrated")):
            continue
        calibrated_days.add(key)
        if not _truth(row.get("market_blend_calibrated")):
            continue
        if key in traded_days:
            continue
        price_age = _safe_float(row.get("market_price_max_age_seconds"))
        coverage = _safe_float(row.get("market_quote_coverage")) or 0.0
        if price_age is None or price_age > maximum_price_age_seconds or coverage < 0.999:
            continue
        model = _json_floats(row.get("model_distribution_json"))
        prices = _json_floats(row.get("historical_market_price_json"))
        rates = _json_floats(row.get("historical_market_fee_rate_json"))
        exponents = _json_floats(row.get("historical_market_fee_exponent_json"))
        enabled = _json_truths(row.get("historical_market_fees_enabled_json"))
        candidates: list[tuple[str, float, float, float, float, float]] = []
        for label, probability in model.items():
            price = prices.get(label)
            if (
                price is None
                or probability < minimum_model_probability
                or price < minimum_market_price
                or price > maximum_market_price
            ):
                continue
            fee = _weather_fee(
                price,
                rates.get(label),
                exponents.get(label),
                enabled.get(label, False),
            )
            station_buffer = station_buffers.get(str(row["station_id"]), execution_buffer)
            conservative_cost = min(1.0, price + station_buffer + fee)
            candidates.append(
                (
                    label,
                    probability - conservative_cost,
                    price,
                    conservative_cost,
                    fee,
                    station_buffer,
                )
            )
        if not candidates:
            continue
        label, edge, price, cost, fee, station_buffer = max(
            candidates, key=lambda item: item[1]
        )
        if edge < edge_threshold:
            continue
        winner = str(row.get("winning_bucket") or "")
        pnl = (1.0 - cost) if label == winner else -cost
        trade = {
            "station_id": row["station_id"],
            "test_date": row["test_date"],
            "received_wall_ns": row["received_wall_ns"],
            "local_hour": row.get("local_hour"),
            "chosen_bucket": label,
            "winning_bucket": winner,
            "model_probability": model[label],
            "historical_price": price,
            "synthetic_execution_buffer": station_buffer,
            "fee_per_share": fee,
            "conservative_cost_per_share": cost,
            "estimated_edge_after_costs": edge,
            "won": label == winner,
            "pnl_per_share": pnl,
            "price_age_seconds": price_age,
            "execution_proven": False,
        }
        trades.append(trade)
        traded_days.add(key)

    all_calibrated_pnl = {
        key: 0.0 for key in calibrated_days
    }
    for trade in trades:
        key = (str(trade["station_id"]), str(trade["test_date"]))
        all_calibrated_pnl[key] += float(trade["pnl_per_share"])

    by_station: list[dict[str, Any]] = []
    for station_id in sorted({key[0] for key in evaluated_days}):
        station_values = [
            pnl for (candidate, _), pnl in all_calibrated_pnl.items() if candidate == station_id
        ]
        station_trades = [trade for trade in trades if trade["station_id"] == station_id]
        by_station.append(
            _summary_row(station_id, station_values, station_trades)
        )
    overall_values = list(all_calibrated_pnl.values())
    overall = _summary_row("ALL", overall_values, trades)
    interval = overall.get("mean_pnl_per_calibrated_station_day_ci95")
    enough_days = len(calibrated_days) >= 60
    positive_screen = bool(
        enough_days and interval and _safe_float(interval.get("low")) is not None
        and float(interval["low"]) > 0
    )
    if positive_screen:
        status = "candidate_predictive_edge_detected_execution_unproven"
    elif enough_days:
        status = "no_robust_edge_after_conservative_proxy_costs"
    else:
        status = "insufficient_independent_calibrated_station_days"

    output_dir = cfg.reports_dir / "historical_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / f"{output_name}_trades.csv"
    _write_csv(trades_path, trades)
    summary_path = output_dir / f"{output_name}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": status,
                "predictive_edge_detected": positive_screen,
                "execution_alpha_proven": False,
                "evaluated_station_days": len(evaluated_days),
                "calibrated_station_days": len(calibrated_days),
                "parameters": {
                    "edge_threshold": edge_threshold,
                    "execution_buffer": execution_buffer,
                    "station_execution_buffers": station_buffers,
                    "station_buffers_from_live_p90_depth_cost": bool(station_buffers),
                    "minimum_model_probability": minimum_model_probability,
                    "minimum_market_price": minimum_market_price,
                    "maximum_market_price": maximum_market_price,
                    "maximum_price_age_seconds": maximum_price_age_seconds,
                    "max_trades_per_station_day": 1,
                    "selection_rule": "first chronological signal crossing all fixed filters",
                },
                "overall": overall,
                "by_station": by_station,
                "warnings": [
                    "Polymarket price history is not a historical ask or L2 order book.",
                    "The execution buffer is deliberately conservative but synthetic.",
                    "Only live captured asks, depth and latency can prove executable alpha.",
                    "Prices below the configured floor are excluded; cheap tail contracts are not bought merely because they are cheap.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"trades": trades_path, "summary": summary_path}


def evaluate_historical_sensitivity(
    cfg: AppConfig,
    *,
    edge_thresholds: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10),
    execution_buffers: tuple[float, ...] = (0.01, 0.02, 0.03),
    validation_path: Path | None = None,
    output_prefix: str = "historical",
) -> dict[str, Path]:

    grid: list[dict[str, Any]] = []
    for edge_threshold in edge_thresholds:
        for execution_buffer in execution_buffers:
            suffix = f"sensitivity_e{edge_threshold:.3f}_c{execution_buffer:.3f}".replace(
                ".", "p"
            )
            paths = evaluate_historical_price_screen(
                cfg,
                validation_path=validation_path,
                edge_threshold=edge_threshold,
                execution_buffer=execution_buffer,
                output_name=f"{output_prefix}_{suffix}",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            overall = summary["overall"]
            grid.append(
                {
                    "edge_threshold": edge_threshold,
                    "minimum_execution_buffer": execution_buffer,
                    "status": summary["status"],
                    "predictive_edge_detected": summary["predictive_edge_detected"],
                    "trades": overall["trades"],
                    "wins": overall["wins"],
                    "total_pnl_per_share": overall["total_pnl_per_share"],
                    "mean_pnl_per_calibrated_station_day": overall[
                        "mean_pnl_per_calibrated_station_day"
                    ],
                    "ci95_low": overall[
                        "mean_pnl_per_calibrated_station_day_ci95"
                    ]["low"],
                    "ci95_high": overall[
                        "mean_pnl_per_calibrated_station_day_ci95"
                    ]["high"],
                }
            )
    output_dir = cfg.reports_dir / "historical_alpha"
    csv_path = output_dir / f"{output_prefix}_sensitivity_grid.csv"
    _write_csv(csv_path, grid)
    summary_path = output_dir / f"{output_prefix}_sensitivity_summary.json"
    robust_cells = [row for row in grid if row["predictive_edge_detected"]]
    summary_path.write_text(
        json.dumps(
            {
                "grid": grid,
                "robust_positive_cells": len(robust_cells),
                "sensitivity_robust": len(robust_cells) == len(grid),
                "selection_warning": (
                    "This grid is a robustness diagnostic. Its best cell must not be selected "
                    "after observing outcomes and reported as an out-of-sample strategy."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"grid": csv_path, "summary": summary_path}


def _summary_row(
    station_id: str, station_day_pnl: list[float], trades: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "calibrated_station_days": len(station_day_pnl),
        "trades": len(trades),
        "trade_days": len(
            {
                (str(trade["station_id"]), str(trade["test_date"]))
                for trade in trades
            }
        ),
        "wins": sum(bool(trade["won"]) for trade in trades),
        "hit_rate": (
            sum(bool(trade["won"]) for trade in trades) / len(trades) if trades else None
        ),
        "total_pnl_per_share": sum(station_day_pnl),
        "mean_pnl_per_calibrated_station_day": (
            sum(station_day_pnl) / len(station_day_pnl) if station_day_pnl else None
        ),
        "mean_pnl_per_calibrated_station_day_ci95": _bootstrap_mean_ci(
            station_day_pnl
        ),
    }


def _weather_fee(
    price: float, rate: float | None, exponent: float | None, enabled: bool
) -> float:
    if not enabled:
        return 0.0


    effective_rate = 0.05 if rate is None else max(0.0, rate)
    effective_exponent = 1.0 if exponent is None else max(0.0, exponent)
    return effective_rate * (max(0.0, price * (1.0 - price)) ** effective_exponent)


def _json_floats(value: Any) -> dict[str, float]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): number
        for key, raw in parsed.items()
        if (number := _safe_float(raw)) is not None and math.isfinite(number)
    }


def _json_truths(value: Any) -> dict[str, bool]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return (
        {str(key): _truth(raw) for key, raw in parsed.items()}
        if isinstance(parsed, dict)
        else {}
    )


def _truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "1.0", "yes"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
