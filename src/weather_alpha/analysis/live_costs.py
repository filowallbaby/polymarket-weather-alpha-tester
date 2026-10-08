from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import orjson

from ..config import AppConfig


def estimate_live_execution_costs(
    cfg: AppConfig,
    *,
    shares: float = 10.0,
    samples_per_file: int = 5_000,
    minimum_price: float = 0.02,
    maximum_price: float = 0.95,
) -> dict[str, Path]:

    if shares <= 0 or samples_per_file < 100:
        raise ValueError("shares must be positive and samples_per_file must be at least 100")
    files = sorted(cfg.data_dir.glob("runs/*/*__polymarket_book.jsonl"))
    if not files:
        raise FileNotFoundError("No uncompressed polymarket_book JSONL files are available")
    observations: list[dict[str, Any]] = []
    sampled_lines = 0
    parse_failures = 0
    for path in files:
        file_size = path.stat().st_size
        if file_size <= 0:
            continue
        with path.open("rb") as handle:
            for index in range(samples_per_file):
                offset = int(file_size * (index + 1) / (samples_per_file + 1))
                handle.seek(offset)
                if offset:
                    handle.readline()
                line = handle.readline()
                if not line:
                    continue
                sampled_lines += 1
                try:
                    event = orjson.loads(line)
                except orjson.JSONDecodeError:
                    parse_failures += 1
                    continue
                payload = event.get("payload", {})
                if (
                    event.get("event_type") != "book_state"
                    or payload.get("depth_valid") is not True
                    or payload.get("book_synchronized") is not True
                    or payload.get("book_crossed") is True
                ):
                    continue
                bid = _number(payload.get("best_bid"))
                ask = _number(payload.get("best_ask"))
                mid = _number(payload.get("mid"))
                if (
                    bid is None
                    or ask is None
                    or mid is None
                    or not (minimum_price <= mid <= maximum_price)
                    or bid > ask
                ):
                    continue
                vwap = _ask_vwap(payload.get("asks"), shares)
                if vwap is None:
                    continue
                observations.append(
                    {
                        "station_id": str(event.get("station_id") or ""),
                        "mid": mid,
                        "best_ask": ask,
                        "half_spread_from_mid": max(0.0, ask - mid),
                        "ask_vwap": vwap,
                        "vwap_cost_from_mid": max(0.0, vwap - mid),
                    }
                )
    if not observations:
        raise RuntimeError("Uniform book samples contained no synchronized depth for the request")

    by_station = []
    for station_id in sorted({row["station_id"] for row in observations}):
        station_rows = [row for row in observations if row["station_id"] == station_id]
        by_station.append(_summarize(station_id, station_rows))
    overall = _summarize("ALL", observations)


    recommended_buffer = max(0.001, float(overall["vwap_cost_from_mid"]["p90"]))
    output_dir = cfg.reports_dir / "execution_costs"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "live_execution_cost_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "sampled_plain_book_files": [str(path) for path in files],
                "samples_per_file": samples_per_file,
                "sampled_lines": sampled_lines,
                "parse_failures": parse_failures,
                "valid_depth_samples": len(observations),
                "requested_shares": shares,
                "price_filter": [minimum_price, maximum_price],
                "recommended_historical_execution_buffer": recommended_buffer,
                "recommendation_basis": "p90 live ask VWAP minus mid for requested shares",
                "overall": overall,
                "by_station": by_station,
                "warning": "A sampled live cost distribution is not a historical fill guarantee.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"summary": summary_path}


def _ask_vwap(levels: Any, shares: float) -> float | None:
    if not isinstance(levels, list):
        return None
    remaining = shares
    cost = 0.0
    parsed = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = _number(level.get("price"))
        size = _number(level.get("size"))
        if price is not None and size is not None and price > 0 and size > 0:
            parsed.append((price, size))
    for price, size in sorted(parsed):
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        if remaining <= 1e-12:


            return round(cost / shares, 12)
    return None


def _summarize(station_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "samples": len(rows),
        "half_spread_from_mid": _distribution(
            [float(row["half_spread_from_mid"]) for row in rows]
        ),
        "vwap_cost_from_mid": _distribution(
            [float(row["vwap_cost_from_mid"]) for row in rows]
        ),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
