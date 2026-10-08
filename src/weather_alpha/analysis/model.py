from __future__ import annotations

import json
import math
from typing import Any

from ..parsing.buckets import parse_temperature_bucket, round_half_up


NON_FEATURE_COLUMNS = {
    "run_id",
    "station_id",
    "city",
    "source",
    "decision_mode",
    "trigger_source",
    "received_wall_ns",
    "observation_time",
    "observation_key",
    "local_date",
    "day_history_complete",
    "continuity_segment",
    "continuity_eligible",
    "market_distribution_json",
    "market_labels_json",
    "market_bid_json",
    "market_ask_json",
    "market_bid_size_json",
    "market_ask_size_json",
    "market_fee_rate_json",
    "market_fee_exponent_json",
    "market_fees_enabled_json",
    "market_ws_healthy",
    "market_quote_coverage",
    "market_price_max_age_seconds",
    "market_price_history_only",
    "historical_market_price_json",
    "historical_market_fee_rate_json",
    "historical_market_fee_exponent_json",
    "historical_market_fees_enabled_json",
    "historical_point_in_time",
    "target_source",
    "evaluation_eligible",
    "final_temperature_target",
    "ncei_final_max_c",
    "ncei_running_max_c",
    "ncei_remaining_increment_c",
    "peer_features_source",
    "peer_history_point_in_time",
    "observation_publication_safety_lag_minutes",
    "event_id",
    "event_slug",
    "running_max_source",
    "wu_target_matches_resolved_winner",
    "winning_bucket",
    "final_wu_max",
}


def _read_market_quotes(
    row: Any,
) -> tuple[
    list[str],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    def decode(name: str) -> dict[str, float]:
        raw = row.get(name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return {}
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}
        return {
            str(key): float(value)
            for key, value in parsed.items()
            if _safe_float(value) is not None
        }

    mid = decode("market_distribution_json")
    ask = decode("market_ask_json")
    fee_rate = decode("market_fee_rate_json")
    fee_exponent = decode("market_fee_exponent_json")
    fees_enabled = decode("market_fees_enabled_json")
    labels = set(mid) | set(ask)
    raw_labels = row.get("market_labels_json")
    if raw_labels is not None and not (isinstance(raw_labels, float) and math.isnan(raw_labels)):
        try:
            parsed_labels = json.loads(str(raw_labels))
        except json.JSONDecodeError:
            parsed_labels = []
        if isinstance(parsed_labels, list):
            labels.update(str(label) for label in parsed_labels if str(label).strip())
    return sorted(labels), mid, ask, fee_rate, fee_exponent, fees_enabled


def _label_for_value(value: float, labels: list[str], unit: str) -> str | None:
    represented = round_half_up(value)
    for label in labels:
        try:
            if parse_temperature_bucket(label, unit).contains(represented):
                return label
        except ValueError:
            continue
    return None


def _best_edge(
    probabilities: dict[str, float],
    asks: dict[str, float],
    fee_rates: dict[str, float] | None = None,
    fee_exponents: dict[str, float] | None = None,
    fees_enabled: dict[str, float] | None = None,
) -> tuple[str | None, float | None]:
    fee_rates = fee_rates or {}
    fee_exponents = fee_exponents or {}
    fees_enabled = fees_enabled or {}
    candidates = [
        (
            label,
            probability
            - asks[label]
            - _fee_per_share(
                asks[label],
                fee_rates.get(label),
                fee_exponents.get(label),
                fees_enabled.get(label, 0.0) > 0,
            ),
        )
        for label, probability in probabilities.items()
        if label in asks
    ]
    return max(candidates, key=lambda item: item[1]) if candidates else (None, None)


def _fee_per_share(
    price: float,
    rate: float | None,
    exponent: float | None,
    enabled: bool,
) -> float:
    if not enabled:
        return 0.0
    if rate is None:

        return math.inf
    return rate * (max(0.0, price * (1.0 - price)) ** (exponent or 1.0))


def _normalize(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()} if total else {}


def _brier(probabilities: dict[str, float], winner: str) -> float:
    labels = set(probabilities) | {winner}
    return sum(
        (probabilities.get(label, 0.0) - (1.0 if label == winner else 0.0)) ** 2
        for label in labels
    )


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _daily_means(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _safe_float(row.get(field))
        if value is not None:
            grouped.setdefault(str(row["test_date"]), []).append(value)
    return {day: sum(values) / len(values) for day, values in grouped.items() if values}


def _bootstrap_mean_ci(values: list[float], samples: int = 2_000) -> dict[str, Any] | None:


    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return {"low": finite[0], "high": finite[0], "samples": samples}
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies: pip install -e '.[analysis]'") from exc
    array = np.asarray(finite, dtype=float)
    generator = np.random.default_rng(42)
    means = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "samples": samples,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
