from __future__ import annotations

import asyncio
import csv
import json
import os
import statistics
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

from ..config import AppConfig, StationConfig
from .historical import (
    _download_station_events,
    _normalize_station_events,
    _read_gzip_json,
    _write_csv,
    _write_gzip_json,
)
from .model import _bootstrap_mean_ci, _safe_float

DEFAULT_COMPARISON_PATH = Path("calibration_challenger/chronological_comparison.csv")
SIGNAL_PROBABILITY_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
SIGNAL_GAP_THRESHOLDS = (0.00, 0.04, 0.08, 0.12, 0.20)


async def backfill_resolution_ledger(
    cfg: AppConfig,
    *,
    refresh: bool = True,
    comparison_path: Path | None = None,
    http_concurrency: int = 4,
) -> dict[str, Path]:

    events_by_station, download_audit = await _fetch_complete_resolution_catalog(
        cfg,
        refresh=refresh,
        http_concurrency=http_concurrency,
    )
    if not events_by_station:
        raise RuntimeError("No resolved Polymarket weather events were found")

    comparison = (
        comparison_path.resolve()
        if comparison_path is not None
        else cfg.reports_dir / DEFAULT_COMPARISON_PATH
    )
    comparison_rows = _load_csv(comparison)
    market_rows, validation_paths = _load_validation_rows(cfg)
    rows, signal_sensitivity = _build_resolution_rows(
        cfg,
        events_by_station,
        comparison_rows=comparison_rows,
        market_rows=market_rows,
    )

    output_dir = cfg.reports_dir / "resolutions"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "resolution_ledger.csv"
    _write_csv(ledger_path, rows)
    sensitivity_path = output_dir / "nonoracle_signal_sensitivity.csv"
    _write_csv(sensitivity_path, signal_sensitivity)
    catalog_path = output_dir / "resolved_events_catalog.json"
    catalog_path.write_text(
        json.dumps(events_by_station, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = _resolution_summary(
        rows,
        download_audit=download_audit,
        comparison_path=comparison,
        validation_paths=validation_paths,
    )
    summary_path = output_dir / "resolution_ledger_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "ledger": ledger_path,
        "signal_sensitivity": sensitivity_path,
        "catalog": catalog_path,
        "summary": summary_path,
    }


async def _fetch_complete_resolution_catalog(
    cfg: AppConfig,
    *,
    refresh: bool,
    http_concurrency: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cache_dir = cfg.data_dir / "historical" / "gamma"
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": os.getenv(
            "HTTP_USER_AGENT",
            "polymarket-weather-alpha-tester/0.1.26 resolution-ledger",
        )
    }
    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(
        limit=max(2, http_concurrency),
        ttl_dns_cache=300,
    )
    semaphore = asyncio.Semaphore(max(1, http_concurrency))

    async with aiohttp.ClientSession(
        headers=headers,
        timeout=timeout,
        connector=connector,
    ) as session:

        async def fetch(station: StationConfig) -> tuple[str, list[Any], list[dict[str, Any]]]:
            if station.polymarket_series_id is None:
                return station.station_id, [], []
            destination = cache_dir / f"{station.station_id}.json.gz"
            if refresh or not destination.exists():
                raw_events = await _download_station_events(
                    session,
                    semaphore,
                    station,
                )
                _write_gzip_json(destination, raw_events)
            else:
                raw_events = _read_gzip_json(destination)
                if not isinstance(raw_events, list):
                    raise RuntimeError(
                        f"Invalid Gamma cache for {station.station_id}: expected a list"
                    )
            normalized = _normalize_station_events(raw_events)
            return station.station_id, raw_events, normalized

        results = await asyncio.gather(
            *(fetch(station) for station in cfg.stations.values())
        )

    catalog = {
        station_id: normalized
        for station_id, _raw, normalized in results
        if normalized
    }
    audit = {
        "pagination": "Gamma /events/keyset using opaque after_cursor",
        "refreshed": refresh,
        "raw_events_by_station": {
            station_id: len(raw)
            for station_id, raw, _normalized in results
        },
        "verified_resolved_events_by_station": {
            station_id: len(normalized)
            for station_id, _raw, normalized in results
        },
        "excluded_unverified_or_duplicate_events_by_station": {
            station_id: len(raw) - len(normalized)
            for station_id, raw, normalized in results
        },
    }
    return catalog, audit


def _build_resolution_rows(
    cfg: AppConfig,
    events_by_station: dict[str, list[dict[str, Any]]],
    *,
    comparison_rows: list[dict[str, str]],
    market_rows: dict[tuple[str, str, str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in comparison_rows:
        station_id = str(row.get("station_id") or "")
        test_date = str(row.get("test_date") or "")
        timestamp_ns = _timestamp_ns(row.get("received_wall_ns"))
        model_distribution = _probabilities(row.get("challenger_distribution_json"))
        if not station_id or not test_date or timestamp_ns is None or not model_distribution:
            continue
        validation = market_rows.get(_row_key(row), {})
        market_distribution = _probabilities(
            validation.get("historical_market_price_json")
        )
        predictions.setdefault((station_id, test_date), []).append(
            {
                "timestamp_ns": timestamp_ns,
                "timestamp_utc": datetime.fromtimestamp(
                    timestamp_ns / 1e9,
                    tz=UTC,
                ),
                "decision_mode": str(row.get("decision_mode") or ""),
                "local_hour": _safe_float(row.get("local_hour")),
                "source_winner": str(row.get("winning_bucket") or ""),
                "challenger_accepted": _truthy(row.get("challenger_accepted")),
                "model_distribution": model_distribution,
                "market_distribution": market_distribution,
                "market_price_history_only": _truthy(
                    validation.get("market_price_history_only")
                ),
                "market_price_max_age_seconds": _safe_float(
                    validation.get("market_price_max_age_seconds")
                ),
            }
        )

    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    rows: list[dict[str, Any]] = []
    signal_events: list[dict[str, Any]] = []
    for station_id, events in sorted(events_by_station.items()):
        station = stations[station_id]
        for event in events:
            event_date = str(event["event_date"])
            snapshots = _deduplicate_snapshots(
                predictions.get((station_id, event_date), [])
            )
            row = _base_resolution_row(station, event)
            row.update(_prediction_metrics(station, event, snapshots))
            rows.append(row)
            signal_events.extend(
                _signal_grid_events(station, event, snapshots)
            )
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["event_date"]),
            str(row["station_id"]),
        ),
    )
    return ordered_rows, _signal_sensitivity(signal_events)


def _base_resolution_row(
    station: StationConfig,
    event: dict[str, Any],
) -> dict[str, Any]:
    contracts = event.get("contracts") or []
    event_unit = _bucket_unit(event.get("winner"))
    return {
        "station_key": station.key,
        "station_id": station.station_id,
        "station_name": station.station_name,
        "city": station.city,
        "timezone": station.timezone,
        "configured_market_unit": station.market_unit,
        "event_market_unit": event_unit,
        "event_unit_matches_current_config": (
            event_unit == station.market_unit if event_unit else None
        ),
        "polymarket_series_id": station.polymarket_series_id,
        "event_date": event.get("event_date"),
        "event_id": event.get("event_id"),
        "event_slug": event.get("event_slug"),
        "polymarket_url": (
            f"https://polymarket.com/event/{event.get('event_slug')}"
            if event.get("event_slug")
            else ""
        ),
        "event_start_time_utc": event.get("start_time"),
        "market_end_time_utc": event.get("market_end_time"),
        "official_resolution_time_utc": event.get("end_time"),
        "resolution_source": event.get("resolution_source"),
        "official_winning_bucket": event.get("winner"),
        "winner_market_id": event.get("winner_market_id"),
        "winner_condition_id": event.get("winner_condition_id"),
        "winner_yes_token_id": event.get("winner_yes_token_id"),
        "winner_yes_resolution_price": event.get(
            "winner_yes_resolution_price"
        ),
        "winner_uma_resolution_status": event.get(
            "winner_uma_resolution_status"
        ),
        "contract_count": len(contracts),
        "contracts_json": json.dumps(
            contracts,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "official_winner_verified": True,
        "official_winner_method": (
            "single closed Gamma contract with YES resolution price >= 0.999"
        ),
        "economic_pnl_evaluable": False,
        "economic_pnl_exclusion_reason": (
            "resolution outcome alone has no executable historical ask or fill"
        ),
    }


def _prediction_metrics(
    station: StationConfig,
    event: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    winner = _canonical_bucket(event.get("winner"))
    local_day_end = datetime.combine(
        date.fromisoformat(str(event["event_date"])) + timedelta(days=1),
        time.min,
        tzinfo=ZoneInfo(station.timezone),
    ).astimezone(UTC)
    resolution_time = _parse_datetime(event.get("end_time"))

    for snapshot in snapshots:
        model_top, model_top_value = _top_probability(
            snapshot["model_distribution"]
        )
        market_top, market_top_value = _top_probability(
            snapshot["market_distribution"]
        )
        snapshot.update(
            {
                "model_top": model_top,
                "model_top_probability": model_top_value,
                "model_winner_probability": snapshot[
                    "model_distribution"
                ].get(winner),
                "model_correct": model_top == winner,
                "model_top_market_price_proxy": snapshot[
                    "market_distribution"
                ].get(model_top),
                "market_top": market_top,
                "market_top_price_proxy": market_top_value,
                "market_winner_price_proxy": snapshot[
                    "market_distribution"
                ].get(winner),
                "market_correct": bool(market_top) and market_top == winner,
            }
        )

    first = snapshots[0] if snapshots else None
    last = snapshots[-1] if snapshots else None
    first_model_correct = next(
        (snapshot for snapshot in snapshots if snapshot["model_correct"]),
        None,
    )
    first_model_correct_20 = next(
        (
            snapshot
            for snapshot in snapshots
            if snapshot["model_correct"]
            and (snapshot["model_top_probability"] or 0.0) >= 0.20
        ),
        None,
    )
    first_model_confirmed = _first_confirmed_correct(
        snapshots,
        label_key="model_top",
        winner=winner,
        minimum_confirmations=2,
    )
    market_snapshots = [
        snapshot for snapshot in snapshots if snapshot["market_distribution"]
    ]
    first_market_correct = next(
        (
            snapshot
            for snapshot in market_snapshots
            if snapshot["market_correct"]
        ),
        None,
    )
    first_nonoracle_signal = _select_nonoracle_signal(
        snapshots,
        minimum_probability=0.20,
        minimum_probability_gap=0.00,
    )
    signal_market_adoption = None
    if first_nonoracle_signal is not None:
        signal_label = first_nonoracle_signal["model_top"]
        signal_market_adoption = next(
            (
                snapshot
                for snapshot in snapshots
                if snapshot["timestamp_ns"]
                >= first_nonoracle_signal["timestamp_ns"]
                and snapshot["market_top"] == signal_label
            ),
            None,
        )
    model_time = (
        first_model_correct["timestamp_utc"]
        if first_model_correct is not None
        else None
    )
    market_time = (
        first_market_correct["timestamp_utc"]
        if first_market_correct is not None
        else None
    )
    lead_vs_market = (
        (market_time - model_time).total_seconds() / 3600
        if model_time is not None and market_time is not None
        else None
    )
    source_winners = {
        _canonical_bucket(snapshot["source_winner"])
        for snapshot in snapshots
        if snapshot["source_winner"]
    }
    first_model_correct_market_price = (
        first_model_correct.get("market_winner_price_proxy")
        if first_model_correct is not None
        else None
    )
    first_model_correct_probability = (
        first_model_correct.get("model_winner_probability")
        if first_model_correct is not None
        else None
    )
    first_model_correct_20_market_price = (
        first_model_correct_20.get("market_winner_price_proxy")
        if first_model_correct_20 is not None
        else None
    )
    first_model_correct_20_probability = (
        first_model_correct_20.get("model_winner_probability")
        if first_model_correct_20 is not None
        else None
    )

    return {
        "model_prediction_coverage": bool(snapshots),
        "model_prediction_rows": len(snapshots),
        "model_prediction_source": (
            "chronological selected forecast-anchored distribution"
            if snapshots
            else ""
        ),
        "model_source_winner_matches_official": (
            source_winners == {winner} if source_winners else None
        ),
        "first_prediction_time_utc": _iso(first),
        "first_prediction_local_hour": (
            first.get("local_hour") if first is not None else None
        ),
        "first_model_top_bucket": (
            first.get("model_top") if first is not None else ""
        ),
        "first_model_top_probability": (
            first.get("model_top_probability") if first is not None else None
        ),
        "first_model_winner_probability": (
            first.get("model_winner_probability")
            if first is not None
            else None
        ),
        "first_model_top_correct": (
            first.get("model_correct") if first is not None else None
        ),
        "last_prediction_time_utc": _iso(last),
        "last_model_top_bucket": (
            last.get("model_top") if last is not None else ""
        ),
        "last_model_top_probability": (
            last.get("model_top_probability") if last is not None else None
        ),
        "last_model_winner_probability": (
            last.get("model_winner_probability") if last is not None else None
        ),
        "last_model_top_correct": (
            last.get("model_correct") if last is not None else None
        ),
        "model_ever_top_correct": first_model_correct is not None,
        "first_model_top_correct_time_utc": _iso(first_model_correct),
        "first_model_top_correct_probability": (
            first_model_correct.get("model_top_probability")
            if first_model_correct is not None
            else None
        ),
        "first_model_top_correct_market_winner_price_proxy": (
            first_model_correct_market_price
        ),
        "first_model_top_correct_probability_minus_price_proxy": (
            first_model_correct_probability
            - first_model_correct_market_price
            if first_model_correct_probability is not None
            and first_model_correct_market_price is not None
            else None
        ),
        "first_model_top_correct_market_already_top_correct": (
            first_model_correct.get("market_correct")
            if first_model_correct is not None
            and first_model_correct["market_distribution"]
            else None
        ),
        "first_model_top_correct_lead_to_local_day_end_hours": _lead_hours(
            local_day_end,
            first_model_correct,
        ),
        "first_model_top_correct_lead_to_official_resolution_hours": _lead_hours(
            resolution_time,
            first_model_correct,
        ),
        "first_model_top_correct_ge_20_time_utc": _iso(
            first_model_correct_20
        ),
        "first_model_top_correct_ge_20_probability": (
            first_model_correct_20_probability
        ),
        "first_model_top_correct_ge_20_market_winner_price_proxy": (
            first_model_correct_20_market_price
        ),
        "first_model_top_correct_ge_20_probability_minus_price_proxy": (
            first_model_correct_20_probability
            - first_model_correct_20_market_price
            if first_model_correct_20_probability is not None
            and first_model_correct_20_market_price is not None
            else None
        ),
        "first_model_top_correct_ge_20_market_already_top_correct": (
            first_model_correct_20.get("market_correct")
            if first_model_correct_20 is not None
            and first_model_correct_20["market_distribution"]
            else None
        ),
        "first_model_top_correct_ge_20_lead_to_local_day_end_hours": (
            _lead_hours(local_day_end, first_model_correct_20)
        ),
        "first_model_two_confirmation_correct_time_utc": _iso(
            first_model_confirmed
        ),
        "first_model_two_confirmation_correct_lead_to_local_day_end_hours": (
            _lead_hours(local_day_end, first_model_confirmed)
        ),
        "market_price_proxy_coverage": bool(market_snapshots),
        "market_price_proxy_kind": (
            "CLOB one-minute price history; not bid/ask/L2"
            if market_snapshots
            else ""
        ),
        "first_market_top_correct_time_utc": _iso(first_market_correct),
        "first_market_top_correct_price_proxy": (
            first_market_correct.get("market_top_price_proxy")
            if first_market_correct is not None
            else None
        ),
        "model_lead_vs_market_first_correct_hours": lead_vs_market,
        "model_first_correct_before_market": (
            lead_vs_market > 0 if lead_vs_market is not None else None
        ),
        "model_correct_while_market_never_correct_in_observed_window": (
            first_model_correct is not None
            and bool(market_snapshots)
            and first_market_correct is None
        ),
        "first_nonoracle_disagreement_signal_time_utc": _iso(
            first_nonoracle_signal
        ),
        "first_nonoracle_disagreement_signal_bucket": (
            first_nonoracle_signal.get("model_top")
            if first_nonoracle_signal is not None
            else ""
        ),
        "first_nonoracle_disagreement_signal_model_probability": (
            first_nonoracle_signal.get("model_top_probability")
            if first_nonoracle_signal is not None
            else None
        ),
        "first_nonoracle_disagreement_signal_market_price_proxy": (
            first_nonoracle_signal.get("model_top_market_price_proxy")
            if first_nonoracle_signal is not None
            else None
        ),
        "first_nonoracle_disagreement_signal_probability_minus_price_proxy": (
            first_nonoracle_signal["model_top_probability"]
            - first_nonoracle_signal["model_top_market_price_proxy"]
            if first_nonoracle_signal is not None
            else None
        ),
        "first_nonoracle_disagreement_signal_correct": (
            first_nonoracle_signal.get("model_correct")
            if first_nonoracle_signal is not None
            else None
        ),
        "first_nonoracle_disagreement_signal_lead_to_local_day_end_hours": (
            _lead_hours(local_day_end, first_nonoracle_signal)
        ),
        "first_nonoracle_disagreement_signal_market_adoption_time_utc": _iso(
            signal_market_adoption
        ),
        "first_nonoracle_disagreement_signal_lead_to_market_adoption_hours": (
            (
                signal_market_adoption["timestamp_utc"]
                - first_nonoracle_signal["timestamp_utc"]
            ).total_seconds()
            / 3600
            if first_nonoracle_signal is not None
            and signal_market_adoption is not None
            else None
        ),
        "first_nonoracle_disagreement_signal_market_never_adopted_observed": (
            first_nonoracle_signal is not None
            and signal_market_adoption is None
        ),
        "maximum_market_price_proxy_age_seconds": max(
            (
                float(snapshot["market_price_max_age_seconds"])
                for snapshot in market_snapshots
                if snapshot["market_price_max_age_seconds"] is not None
            ),
            default=None,
        ),
        "all_market_rows_are_non_executable_history": (
            all(
                snapshot["market_price_history_only"]
                for snapshot in market_snapshots
            )
            if market_snapshots
            else None
        ),
    }


def _select_nonoracle_signal(
    snapshots: list[dict[str, Any]],
    *,
    minimum_probability: float,
    minimum_probability_gap: float,
) -> dict[str, Any] | None:
    for snapshot in snapshots:
        model_probability = snapshot.get("model_top_probability")
        market_price = snapshot.get("model_top_market_price_proxy")
        if (
            snapshot.get("market_distribution")
            and snapshot.get("model_top")
            and snapshot.get("market_top") != snapshot.get("model_top")
            and model_probability is not None
            and model_probability >= minimum_probability
            and market_price is not None
            and model_probability > market_price
            and model_probability - market_price
            >= minimum_probability_gap
        ):
            return snapshot
    return None


def _signal_grid_events(
    station: StationConfig,
    event: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    winner = _canonical_bucket(event.get("winner"))
    output: list[dict[str, Any]] = []
    for minimum_probability in SIGNAL_PROBABILITY_THRESHOLDS:
        for minimum_gap in SIGNAL_GAP_THRESHOLDS:
            signal = _select_nonoracle_signal(
                snapshots,
                minimum_probability=minimum_probability,
                minimum_probability_gap=minimum_gap,
            )
            if signal is None:
                continue
            model_probability = float(signal["model_top_probability"])
            market_price = float(signal["model_top_market_price_proxy"])
            correct = signal["model_top"] == winner
            output.append(
                {
                    "station_id": station.station_id,
                    "event_date": event["event_date"],
                    "minimum_model_probability": minimum_probability,
                    "minimum_probability_minus_price_proxy": minimum_gap,
                    "signal_time_utc": _iso(signal),
                    "selected_bucket": signal["model_top"],
                    "official_winning_bucket": event["winner"],
                    "resolved_correct": correct,
                    "outcome": 1.0 if correct else 0.0,
                    "model_probability": model_probability,
                    "market_price_proxy": market_price,
                    "probability_minus_price_proxy": (
                        model_probability - market_price
                    ),
                }
            )
    return output


def _signal_sensitivity(
    signal_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stations = sorted(
        {str(event["station_id"]) for event in signal_events}
    )
    output: list[dict[str, Any]] = []
    for station_id in ["ALL", *stations]:
        station_events = [
            event
            for event in signal_events
            if station_id == "ALL"
            or event["station_id"] == station_id
        ]
        for minimum_probability in SIGNAL_PROBABILITY_THRESHOLDS:
            for minimum_gap in SIGNAL_GAP_THRESHOLDS:
                selected = [
                    event
                    for event in station_events
                    if event["minimum_model_probability"]
                    == minimum_probability
                    and event[
                        "minimum_probability_minus_price_proxy"
                    ]
                    == minimum_gap
                ]
                outcomes = [
                    float(event["outcome"]) for event in selected
                ]
                model_probabilities = [
                    float(event["model_probability"])
                    for event in selected
                ]
                market_prices = [
                    float(event["market_price_proxy"])
                    for event in selected
                ]
                resolution_minus_price = [
                    outcome - price
                    for outcome, price in zip(
                        outcomes,
                        market_prices,
                        strict=True,
                    )
                ]
                model_brier = _mean(
                    [
                        (probability - outcome) ** 2
                        for probability, outcome in zip(
                            model_probabilities,
                            outcomes,
                            strict=True,
                        )
                    ]
                )
                market_brier = _mean(
                    [
                        (probability - outcome) ** 2
                        for probability, outcome in zip(
                            market_prices,
                            outcomes,
                            strict=True,
                        )
                    ]
                )
                output.append(
                    {
                        "station_id": station_id,
                        "minimum_model_probability": (
                            minimum_probability
                        ),
                        "minimum_probability_minus_price_proxy": (
                            minimum_gap
                        ),
                        "independent_station_days": len(selected),
                        "resolved_hits": int(sum(outcomes)),
                        "hit_rate": _mean(outcomes),
                        "hit_rate_ci95_json": (
                            json.dumps(
                                _bootstrap_mean_ci(outcomes),
                                sort_keys=True,
                            )
                            if outcomes
                            else ""
                        ),
                        "mean_model_probability": _mean(
                            model_probabilities
                        ),
                        "mean_market_price_proxy": _mean(
                            market_prices
                        ),
                        "resolved_frequency_minus_market_price_proxy": (
                            _mean(resolution_minus_price)
                        ),
                        "resolved_frequency_minus_market_price_proxy_ci95_json": (
                            json.dumps(
                                _bootstrap_mean_ci(
                                    resolution_minus_price
                                ),
                                sort_keys=True,
                            )
                            if resolution_minus_price
                            else ""
                        ),
                        "mean_probability_minus_price_proxy": _mean(
                            [
                                probability - price
                                for probability, price in zip(
                                    model_probabilities,
                                    market_prices,
                                    strict=True,
                                )
                            ]
                        ),
                        "model_brier": model_brier,
                        "market_price_proxy_brier": market_brier,
                        "model_brier_improvement_over_market_proxy": (
                            market_brier - model_brier
                            if model_brier is not None
                            and market_brier is not None
                            else None
                        ),
                        "selection_uses_outcome": False,
                        "economic_pnl_evaluable": False,
                    }
                )
    return output


def _first_confirmed_correct(
    snapshots: list[dict[str, Any]],
    *,
    label_key: str,
    winner: str,
    minimum_confirmations: int,
) -> dict[str, Any] | None:
    previous = ""
    confirmations = 0
    for snapshot in snapshots:
        label = str(snapshot.get(label_key) or "")
        if label and label == previous:
            confirmations += 1
        else:
            previous = label
            confirmations = 1 if label else 0
        if label == winner and confirmations >= minimum_confirmations:
            return snapshot
    return None


def _deduplicate_snapshots(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_timestamp: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        by_timestamp[int(snapshot["timestamp_ns"])] = snapshot
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def _load_validation_rows(
    cfg: AppConfig,
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, str]],
    list[Path],
]:
    paths = sorted(
        cfg.reports_dir.glob(
            "model_validation/master_remaining_max_validation*.csv"
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    records: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _load_csv(path):
            records[_row_key(row)] = row
    return records, paths


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    timestamp = _timestamp_ns(row.get("received_wall_ns"))
    return (
        str(row.get("station_id") or ""),
        str(row.get("test_date") or ""),
        str(row.get("decision_mode") or ""),
        str(timestamp or ""),
    )


def _timestamp_ns(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed > 0 else None


def _probabilities(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    if not isinstance(raw, dict):
        return {}
    output: dict[str, float] = {}
    for label, probability in raw.items():
        number = _safe_float(probability)
        canonical = _canonical_bucket(label)
        if canonical and number is not None and number >= 0:
            output[canonical] = output.get(canonical, 0.0) + number
    return output


def _canonical_bucket(value: Any) -> str:
    return " ".join(
        str(value or "")
        .replace("\u00c2\u00b0", "\u00b0")
        .replace("\u00a0", " ")
        .split()
    )


def _bucket_unit(value: Any) -> str:
    label = _canonical_bucket(value)
    if "\u00b0F" in label:
        return "F"
    if "\u00b0C" in label:
        return "C"
    return ""


def _top_probability(probabilities: dict[str, float]) -> tuple[str, float | None]:
    if not probabilities:
        return "", None
    label, probability = max(
        probabilities.items(),
        key=lambda item: item[1],
    )
    return label, probability


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(snapshot: dict[str, Any] | None) -> str:
    if snapshot is None:
        return ""
    value = snapshot.get("timestamp_utc")
    return value.isoformat() if isinstance(value, datetime) else ""


def _lead_hours(
    target: datetime | None,
    snapshot: dict[str, Any] | None,
) -> float | None:
    if target is None or snapshot is None:
        return None
    prediction_time = snapshot.get("timestamp_utc")
    if not isinstance(prediction_time, datetime):
        return None
    return (target - prediction_time).total_seconds() / 3600


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolution_summary(
    rows: list[dict[str, Any]],
    *,
    download_audit: dict[str, Any],
    comparison_path: Path,
    validation_paths: list[Path],
) -> dict[str, Any]:
    by_station = []
    for station_id in sorted({str(row["station_id"]) for row in rows}):
        station_rows = [
            row for row in rows if row["station_id"] == station_id
        ]
        by_station.append(
            _summary_group(station_rows, station_id=station_id)
        )
    total = _summary_group(rows, station_id=None)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **total,
        "by_station": by_station,
        "download_audit": download_audit,
        "model_comparison_path": str(comparison_path),
        "model_comparison_available": comparison_path.exists(),
        "market_proxy_validation_paths": [
            str(path) for path in validation_paths
        ],
        "methodology": {
            "official_winner": (
                "Gamma closed event with exactly one YES contract resolved "
                "at a price of at least 0.999"
            ),
            "model_distribution": (
                "chronological forecast-anchored distribution selected using "
                "prior station-days only"
            ),
            "first_correct": (
                "ex-post diagnostic: first point-in-time snapshot whose "
                "highest-probability bucket equals the later official winner"
            ),
            "two_confirmation_correct": (
                "second consecutive point-in-time snapshot with the same top "
                "bucket, later verified against the official winner"
            ),
            "model_lead_vs_market": (
                "market first-correct time minus model first-correct time at "
                "matched scheduled snapshots; positive means model was earlier"
            ),
            "nonoracle_disagreement_signal": (
                "first snapshot selected without knowing the outcome where "
                "the model top bucket has probability >= 0.20, differs from "
                "the market top bucket, has a price proxy, and model "
                "probability exceeds that proxy"
            ),
            "market_price": (
                "one-minute CLOB historical price proxy, not an executable "
                "bid/ask or L2 quote"
            ),
            "execution_claim": False,
            "pnl_claim": False,
        },
    }


def _summary_group(
    rows: list[dict[str, Any]],
    *,
    station_id: str | None,
) -> dict[str, Any]:
    predicted = [row for row in rows if row["model_prediction_coverage"]]
    market_covered = [
        row for row in rows if row["market_price_proxy_coverage"]
    ]
    paired = [
        row
        for row in rows
        if row["model_lead_vs_market_first_correct_hours"] is not None
    ]
    leads = [
        float(row["model_lead_vs_market_first_correct_hours"])
        for row in paired
    ]
    ge_20_before_market = [
        row
        for row in rows
        if row["first_model_top_correct_ge_20_time_utc"]
        and row["first_model_top_correct_ge_20_market_already_top_correct"]
        is False
    ]
    nonoracle_signals = [
        row
        for row in rows
        if row["first_nonoracle_disagreement_signal_time_utc"]
    ]
    signal_outcomes = [
        1.0
        if row["first_nonoracle_disagreement_signal_correct"]
        else 0.0
        for row in nonoracle_signals
    ]
    signal_model_probabilities = [
        float(
            row[
                "first_nonoracle_disagreement_signal_model_probability"
            ]
        )
        for row in nonoracle_signals
    ]
    signal_market_prices = [
        float(
            row[
                "first_nonoracle_disagreement_signal_market_price_proxy"
            ]
        )
        for row in nonoracle_signals
    ]
    signal_gaps = [
        float(
            row[
                "first_nonoracle_disagreement_signal_probability_minus_price_proxy"
            ]
        )
        for row in nonoracle_signals
    ]
    signal_resolution_minus_price = [
        outcome - price
        for outcome, price in zip(
            signal_outcomes,
            signal_market_prices,
            strict=True,
        )
    ]
    correct_leads = [
        float(row["first_model_top_correct_lead_to_local_day_end_hours"])
        for row in predicted
        if row["first_model_top_correct_lead_to_local_day_end_hours"]
        is not None
    ]
    output = {
        "resolved_events": len(rows),
        "first_event_date": min(
            (str(row["event_date"]) for row in rows),
            default=None,
        ),
        "last_event_date": max(
            (str(row["event_date"]) for row in rows),
            default=None,
        ),
        "events_with_model_predictions": len(predicted),
        "events_with_market_price_proxy": len(market_covered),
        "first_model_top_accuracy": _rate(
            predicted,
            "first_model_top_correct",
        ),
        "last_model_top_accuracy": _rate(
            predicted,
            "last_model_top_correct",
        ),
        "model_ever_top_correct_rate": _rate(
            predicted,
            "model_ever_top_correct",
        ),
        "model_two_confirmation_correct_rate": (
            sum(
                bool(row["first_model_two_confirmation_correct_time_utc"])
                for row in predicted
            )
            / len(predicted)
            if predicted
            else None
        ),
        "median_first_correct_lead_to_local_day_end_hours": (
            statistics.median(correct_leads) if correct_leads else None
        ),
        "paired_first_correct_model_market_events": len(paired),
        "model_first_correct_before_market_events": sum(
            lead > 0 for lead in leads
        ),
        "model_first_correct_tied_with_market_events": sum(
            lead == 0 for lead in leads
        ),
        "model_first_correct_after_market_events": sum(
            lead < 0 for lead in leads
        ),
        "model_ge_20_correct_while_market_not_top_events": len(
            ge_20_before_market
        ),
        "nonoracle_disagreement_signal_events": len(nonoracle_signals),
        "nonoracle_disagreement_signal_hit_rate": _mean(
            signal_outcomes
        ),
        "nonoracle_disagreement_signal_hit_rate_ci95": (
            _bootstrap_mean_ci(signal_outcomes)
            if signal_outcomes
            else None
        ),
        "nonoracle_disagreement_signal_mean_model_probability": _mean(
            signal_model_probabilities
        ),
        "nonoracle_disagreement_signal_mean_market_price_proxy": _mean(
            signal_market_prices
        ),
        "nonoracle_disagreement_signal_mean_probability_gap": _mean(
            signal_gaps
        ),
        "nonoracle_disagreement_signal_resolved_frequency_minus_market_price_proxy": (
            _mean(signal_resolution_minus_price)
        ),
        "nonoracle_disagreement_signal_resolved_frequency_minus_market_price_proxy_ci95": (
            _bootstrap_mean_ci(signal_resolution_minus_price)
            if signal_resolution_minus_price
            else None
        ),
        "nonoracle_disagreement_signal_model_brier": _mean(
            [
                (probability - outcome) ** 2
                for probability, outcome in zip(
                    signal_model_probabilities,
                    signal_outcomes,
                    strict=True,
                )
            ]
        ),
        "nonoracle_disagreement_signal_market_proxy_brier": _mean(
            [
                (probability - outcome) ** 2
                for probability, outcome in zip(
                    signal_market_prices,
                    signal_outcomes,
                    strict=True,
                )
            ]
        ),
        "nonoracle_disagreement_signal_market_adoption_rate": (
            sum(
                bool(
                    row[
                        "first_nonoracle_disagreement_signal_market_adoption_time_utc"
                    ]
                )
                for row in nonoracle_signals
            )
            / len(nonoracle_signals)
            if nonoracle_signals
            else None
        ),
        "median_model_lead_vs_market_hours": (
            statistics.median(leads) if leads else None
        ),
        "model_correct_market_never_correct_observed_events": sum(
            bool(
                row[
                    "model_correct_while_market_never_correct_in_observed_window"
                ]
            )
            for row in rows
        ),
        "economic_pnl_evaluable_events": 0,
    }
    if station_id is not None:
        output = {"station_id": station_id, **output}
    return output


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return (
        sum(bool(value) for value in values) / len(values)
        if values
        else None
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
