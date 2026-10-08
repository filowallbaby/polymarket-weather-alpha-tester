from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import orjson

from ..config import AppConfig, StationConfig
from ..sources.wunderground import WundergroundCollector
from .meteorology import station_time_features
from .model import _label_for_value, _safe_float

GAMMA_EVENTS_KEYSET_URL = "https://gamma-api.polymarket.com/events/keyset"
POLYMARKET_BATCH_HISTORY_URL = "https://clob.polymarket.com/batch-prices-history"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

DEFAULT_DECISION_HOURS = (9, 12, 15)
DEFAULT_OBSERVATION_LAG_MINUTES = 5
OPEN_METEO_MODELS = {
    "best_match": "best_match",
    "ecmwf": "ecmwf_ifs025",
    "gfs_hrrr": "gfs_seamless",
    "dwd_icon": "icon_seamless",
}
FORECAST_BASE_VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
)


async def backfill_historical_data(
    cfg: AppConfig,
    *,
    start_date: date,
    end_date: date,
    decision_hours: tuple[int, ...] = DEFAULT_DECISION_HOURS,
    observation_lag_minutes: int = DEFAULT_OBSERVATION_LAG_MINUTES,
    refresh: bool = False,
    http_concurrency: int = 6,
) -> dict[str, Path]:

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    if not decision_hours or any(hour < 0 or hour > 23 for hour in decision_hours):
        raise ValueError("decision_hours must contain values from 0 through 23")
    if observation_lag_minutes < 0:
        raise ValueError("observation_lag_minutes cannot be negative")

    cache_dir = cfg.data_dir / "historical"
    report_dir = cfg.reports_dir / "historical"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": os.getenv(
            "HTTP_USER_AGENT", "polymarket-weather-alpha-tester/0.1.26 historical-research"
        )
    }
    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(limit=max(2, http_concurrency), ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(max(1, http_concurrency))

    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        events_by_station = {
            station.station_id: await _fetch_station_events(
                session,
                semaphore,
                station,
                start_date,
                end_date,
                cache_dir,
                refresh=refresh,
            )
            for station in cfg.stations.values()
        }
        events_by_station = {
            station_id: events
            for station_id, events in events_by_station.items()
            if events
        }
        if not events_by_station:
            raise RuntimeError("No resolved Polymarket weather events were found in the interval")

        forecast_tasks: dict[tuple[str, str], asyncio.Task[dict[str, Any]]] = {}
        for station in cfg.stations.values():
            if station.station_id not in events_by_station:
                continue
            for provider in station.forecast_providers:
                model = OPEN_METEO_MODELS.get(provider)
                if model is None:
                    continue
                forecast_tasks[(station.station_id, provider)] = asyncio.create_task(
                    _fetch_previous_runs(
                        session,
                        semaphore,
                        station,
                        provider,
                        model,
                        start_date,
                        end_date,
                        cache_dir,
                        refresh=refresh,
                    )
                )

        price_tasks: dict[tuple[str, str], asyncio.Task[dict[str, Any]]] = {}
        for station_id, events in events_by_station.items():
            station = _station_by_id(cfg, station_id)
            for event in events:
                key = (station_id, str(event["event_id"]))
                price_tasks[key] = asyncio.create_task(
                    _fetch_event_prices(
                        session,
                        semaphore,
                        station,
                        event,
                        decision_hours,
                        cache_dir,
                        refresh=refresh,
                    )
                )

        forecasts = {key: await task for key, task in forecast_tasks.items()}
        prices = {key: await task for key, task in price_tasks.items()}

    target_dates = {
        station_id: sorted({date.fromisoformat(str(item["event_date"])) for item in events})
        for station_id, events in events_by_station.items()
    }
    wu_history = await _fetch_wunderground_history(
        cfg, target_dates, cache_dir, refresh=refresh
    )

    rows, audit = _build_historical_rows(
        cfg,
        events_by_station,
        forecasts,
        prices,
        wu_history,
        decision_hours=decision_hours,
        observation_lag_minutes=observation_lag_minutes,
    )
    if not rows:
        raise RuntimeError(
            "Historical sources were downloaded but no leakage-safe rows passed target validation"
        )

    feature_path = report_dir / "predictive_features.csv"
    _write_csv(feature_path, rows)
    events_path = report_dir / "resolved_events.json"
    events_path.write_text(
        json.dumps(events_by_station, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_path = report_dir / "historical_backfill_summary.json"
    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "included_first_local_date": min(
            (str(row["local_date"]) for row in rows), default=None
        ),
        "included_last_local_date": max(
            (str(row["local_date"]) for row in rows), default=None
        ),
        "decision_hours_local": list(decision_hours),
        "observation_publication_safety_lag_minutes": observation_lag_minutes,
        "resolved_events_by_station": {
            station_id: len(events) for station_id, events in events_by_station.items()
        },
        "feature_rows": len(rows),
        "station_days": len({(row["station_id"], row["local_date"]) for row in rows}),
        "rows_by_station": {
            station_id: sum(row["station_id"] == station_id for row in rows)
            for station_id in sorted(events_by_station)
        },
        "audit": audit,
        "methodology": {
            "weather_target": "final maximum captured from the WU resolution page",
            "winner_target": "resolved Gamma outcome",
            "current_max_floor": "WU observations timestamped before the simulated decision",
            "forecast_information": "Open-Meteo previous_day1/day2 only; no day0 hindsight",
            "market_information": "Polymarket one-minute historical price proxy; not bid/ask/L2",
            "execution_claim": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"features": feature_path, "events": events_path, "summary": summary_path}


async def _fetch_station_events(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    station: StationConfig,
    start_date: date,
    end_date: date,
    cache_dir: Path,
    *,
    refresh: bool,
) -> list[dict[str, Any]]:
    if station.polymarket_series_id is None:
        return []
    destination = cache_dir / "gamma" / f"{station.station_id}.json.gz"
    raw_events: list[dict[str, Any]]
    if destination.exists() and not refresh:
        cached_events = _read_gzip_json(destination)
        if _event_cache_covers(cached_events, start_date, end_date):
            raw_events = cached_events
        else:
            raw_events = await _download_station_events(session, semaphore, station)
            _write_gzip_json(destination, raw_events)
    else:
        raw_events = await _download_station_events(session, semaphore, station)
        _write_gzip_json(destination, raw_events)

    return _normalize_station_events(
        raw_events,
        start_date=start_date,
        end_date=end_date,
    )


def _normalize_station_events(
    raw_events: list[Any],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id") or "")
        event_date = _event_date(event)
        if not event_id or event_id in seen or event_date is None:
            continue
        seen.add(event_id)
        if (
            (start_date is not None and event_date < start_date)
            or (end_date is not None and event_date > end_date)
            or event.get("closed") is not True
        ):
            continue
        markets = _json_list(event.get("markets"))
        contracts: list[dict[str, Any]] = []
        winners: list[dict[str, Any]] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            token_ids = [str(value) for value in _json_list(market.get("clobTokenIds"))]
            outcomes = [str(value) for value in _json_list(market.get("outcomes"))]
            outcome_prices = [_safe_float(value) for value in _json_list(market.get("outcomePrices"))]
            yes_index = next(
                (index for index, value in enumerate(outcomes) if value.casefold() == "yes"), 0
            )
            if yes_index >= len(token_ids):
                continue
            label = str(
                market.get("groupItemTitle")
                or market.get("group_item_title")
                or market.get("question")
                or ""
            ).strip()
            if not label:
                continue
            yes_price = outcome_prices[yes_index] if yes_index < len(outcome_prices) else None
            fee_schedule = market.get("feeSchedule") or {}
            if not isinstance(fee_schedule, dict):
                fee_schedule = {}
            contract = {
                "label": label,
                "yes_token_id": token_ids[yes_index],
                "market_id": str(market.get("id") or ""),
                "condition_id": str(
                    market.get("conditionId") or market.get("condition_id") or ""
                ),
                "yes_resolution_price": yes_price,
                "uma_resolution_status": str(
                    market.get("umaResolutionStatus")
                    or market.get("uma_resolution_status")
                    or ""
                ),
                "fee_rate": _safe_float(fee_schedule.get("rate")),
                "fee_exponent": _safe_float(fee_schedule.get("exponent")),
                "fees_enabled": bool(market.get("feesEnabled") or market.get("fees_enabled")),
            }
            contracts.append(contract)
            if yes_price is not None and yes_price >= 0.999:
                winners.append(contract)
        if len(winners) != 1 or len(contracts) < 2:
            continue
        winner = winners[0]
        normalized.append(
            {
                "event_id": event_id,
                "event_slug": str(event.get("slug") or ""),
                "event_date": event_date.isoformat(),
                "start_time": event.get("startTime") or event.get("startDate"),
                "end_time": event.get("closedTime") or event.get("endDate"),
                "market_end_time": event.get("endDate"),
                "resolution_source": str(event.get("resolutionSource") or ""),
                "winner": winner["label"],
                "winner_market_id": winner["market_id"],
                "winner_condition_id": winner["condition_id"],
                "winner_yes_token_id": winner["yes_token_id"],
                "winner_yes_resolution_price": winner["yes_resolution_price"],
                "winner_uma_resolution_status": winner["uma_resolution_status"],
                "contracts": contracts,
            }
        )
    return _deduplicate_event_dates(normalized)


async def _download_station_events(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    station: StationConfig,
) -> list[dict[str, Any]]:

    raw_events: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _page_number in range(1_000):
        params: dict[str, Any] = {
            "series_id": station.polymarket_series_id,
            "closed": "true",
            "limit": 100,
        }
        if cursor is not None:
            params["after_cursor"] = cursor
        payload = await _request_json(
            session,
            semaphore,
            "GET",
            GAMMA_EVENTS_KEYSET_URL,
            params=params,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise RuntimeError(
                f"Gamma keyset returned an invalid page for {station.station_id}"
            )
        page = payload["events"]
        raw_events.extend(item for item in page if isinstance(item, dict))
        next_cursor = payload.get("next_cursor")
        if not page or not next_cursor:
            break
        cursor = str(next_cursor)
        if cursor in seen_cursors:
            raise RuntimeError(
                f"Gamma keyset repeated a cursor for {station.station_id}"
            )
        seen_cursors.add(cursor)
    else:
        raise RuntimeError(
            f"Gamma keyset exceeded 1,000 pages for {station.station_id}"
        )
    return _deduplicate_raw_events(raw_events)


def _deduplicate_raw_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for event in raw_events:
        event_id = str(event.get("id") or "")
        if event_id:
            selected[event_id] = event
    return list(selected.values())


def _event_cache_covers(
    raw_events: list[Any], start_date: date, end_date: date
) -> bool:
    cached_dates = sorted(
        event_date
        for item in raw_events
        if isinstance(item, dict) and (event_date := _event_date(item)) is not None
    )
    return bool(
        cached_dates
        and cached_dates[0] <= start_date
        and cached_dates[-1] >= end_date
    )


def _deduplicate_event_dates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:

    selected: dict[str, dict[str, Any]] = {}
    for event in events:
        event_date = str(event.get("event_date") or "")
        if not event_date:
            continue
        previous = selected.get(event_date)
        if previous is None or _event_preference(event) > _event_preference(previous):
            selected[event_date] = event
    return [selected[event_date] for event_date in sorted(selected)]


def _event_preference(event: dict[str, Any]) -> tuple[int, int, int]:
    slug = str(event.get("event_slug") or "").casefold()
    canonical = int(not slug.startswith("arch-"))
    contract_count = len(event.get("contracts") or [])
    try:
        event_id = int(event.get("event_id") or 0)
    except (TypeError, ValueError):
        event_id = 0
    return canonical, contract_count, event_id


async def _fetch_previous_runs(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    station: StationConfig,
    provider: str,
    model: str,
    start_date: date,
    end_date: date,
    cache_dir: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    destination = cache_dir / "forecasts" / station.station_id / f"{provider}.json.gz"
    cached: dict[str, Any] | None = None
    cached_start: date | None = None
    cached_end: date | None = None
    if destination.exists() and not refresh:
        candidate = _read_gzip_json(destination)
        try:
            cached_start = date.fromisoformat(str(candidate.get("start_date")))
            cached_end = date.fromisoformat(str(candidate.get("end_date")))
        except (TypeError, ValueError):
            cached_start = None
            cached_end = None
        if (
            cached_start is not None
            and cached_end is not None
            and cached_start <= start_date
            and cached_end >= end_date
        ):
            return candidate
        if (
            cached_start is not None
            and cached_end is not None
            and cached_start <= end_date + timedelta(days=1)
            and cached_end >= start_date - timedelta(days=1)
        ):
            cached = candidate

    fetch_ranges: list[tuple[date, date]] = []
    if cached is not None and cached_start is not None and cached_end is not None:
        if start_date < cached_start:
            fetch_ranges.append((start_date, min(end_date, cached_start - timedelta(days=1))))
        if end_date > cached_end:
            fetch_ranges.append((max(start_date, cached_end + timedelta(days=1)), end_date))
        result_start = min(start_date, cached_start)
        result_end = max(end_date, cached_end)
        merged = dict(cached.get("hourly") or {})
    else:
        fetch_ranges.append((start_date, end_date))
        result_start = start_date
        result_end = end_date
        merged = {}

    if cached is not None and not fetch_ranges:
            return cached

    variables = [
        f"{base}_previous_day{lead}"
        for base in FORECAST_BASE_VARIABLES
        for lead in (1, 2)
    ]
    for range_start, range_end in fetch_ranges:
        chunk_start = range_start
        while chunk_start <= range_end:
            chunk_end = min(range_end, chunk_start + timedelta(days=59))
            payload = await _request_json(
                session,
                semaphore,
                "GET",
                PREVIOUS_RUNS_URL,
                params={
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "hourly": ",".join(variables),
                    "models": model,
                    "timezone": station.timezone,
                },
            )
            hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
            if not isinstance(hourly, dict) or not hourly.get("time"):
                raise RuntimeError(
                    f"Previous Runs returned no hourly data for {station.station_id}/{provider}"
                )
            merged = _merge_hourly(merged, hourly)
            chunk_start = chunk_end + timedelta(days=1)
    result = {
        "provider": provider,
        "model": model,
        "start_date": result_start.isoformat(),
        "end_date": result_end.isoformat(),
        "hourly": merged,
    }
    _write_gzip_json(destination, result)
    return result


async def _fetch_event_prices(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    station: StationConfig,
    event: dict[str, Any],
    decision_hours: tuple[int, ...],
    cache_dir: Path,
    *,
    refresh: bool,
) -> dict[str, Any]:
    destination = cache_dir / "prices" / station.station_id / f"{event['event_id']}.json.gz"
    if destination.exists() and not refresh:
        return _read_gzip_json(destination)
    event_day = date.fromisoformat(str(event["event_date"]))
    timezone = ZoneInfo(station.timezone)
    decisions = [datetime.combine(event_day, time(hour), tzinfo=timezone) for hour in decision_hours]
    token_ids = [str(item["yes_token_id"]) for item in event["contracts"]]
    if len(token_ids) > 20:
        raise RuntimeError(f"Event {event['event_id']} exceeds Polymarket's 20-token batch limit")
    payload = await _request_json(
        session,
        semaphore,
        "POST",
        POLYMARKET_BATCH_HISTORY_URL,
        json_body={
            "markets": token_ids,
            "start_ts": int(min(decisions).astimezone(UTC).timestamp()) - 3600,
            "end_ts": int(max(decisions).astimezone(UTC).timestamp()) + 3600,
            "fidelity": 1,
        },
    )
    history = payload.get("history", {}) if isinstance(payload, dict) else {}
    result = {"event_id": event["event_id"], "history": history}
    _write_gzip_json(destination, result)
    return result


async def _fetch_wunderground_history(
    cfg: AppConfig,
    target_dates: dict[str, list[date]],
    cache_dir: Path,
    *,
    refresh: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for audited WU historical backfill") from exc

    results: dict[tuple[str, str], dict[str, Any]] = {}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="en-US", timezone_id="UTC", service_workers="block"
        )
        try:
            tasks = []
            for station_id, dates in target_dates.items():
                station = _station_by_id(cfg, station_id)
                tasks.append(
                    asyncio.create_task(
                        _fetch_wu_station(context, station, dates, cache_dir, refresh=refresh)
                    )
                )
            for station_rows in await asyncio.gather(*tasks):
                results.update(station_rows)
        finally:
            await context.close()
            await browser.close()
    return results


async def _fetch_wu_station(
    context: Any,
    station: StationConfig,
    dates: list[date],
    cache_dir: Path,
    *,
    refresh: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    page = await context.new_page()
    cdp_session = await context.new_cdp_session(page)
    await cdp_session.send("Network.enable")
    await cdp_session.send("Network.setCacheDisabled", {"cacheDisabled": True})
    output: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for target_date in dates:
            destination = (
                cache_dir / "wunderground" / station.station_id / f"{target_date.isoformat()}.json.gz"
            )
            if destination.exists() and not refresh:
                item = _read_gzip_json(destination)
                output[(station.station_id, target_date.isoformat())] = item
                continue
            date_token = target_date.strftime("%Y%m%d")
            target_url = WundergroundCollector._dated_url(station.resolution_url, target_date)
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    async with page.expect_response(
                        lambda response: (
                            "observations/historical" in response.url.lower()
                            and f"startDate={date_token}" in response.url
                            and f"endDate={date_token}" in response.url
                        ),
                        timeout=45_000,
                    ) as response_info:
                        await page.goto(target_url, wait_until="domcontentloaded", timeout=45_000)
                    response = await response_info.value
                    body = await response.body()
                    if response.status != 200 or len(body) > 5_000_000:
                        raise RuntimeError(
                            f"WU history response status={response.status}, bytes={len(body)}"
                        )
                    payload = orjson.loads(body)
                    observations = WundergroundCollector._extract_observations(
                        payload, response.url, station
                    )
                    converted = [
                        {
                            **observation,
                            "temperature_market_unit": WundergroundCollector._to_unit(
                                float(observation["temperature"]),
                                str(observation["unit"]),
                                station.market_unit,
                            ),
                        }
                        for observation in observations
                        if str(observation.get("station_id") or "").upper() == station.station_id
                    ]
                    converted = [
                        observation
                        for observation in converted
                        if datetime.fromtimestamp(
                            int(observation["timestamp_ns"]) / 1e9, tz=UTC
                        )
                        .astimezone(ZoneInfo(station.timezone))
                        .date()
                        == target_date
                    ]
                    if not converted:
                        raise RuntimeError("WU page returned no target-station observations")
                    raw_digest = hashlib.sha256(body).hexdigest()
                    raw_path = (
                        cache_dir
                        / "wunderground"
                        / "raw"
                        / station.station_id
                        / f"{target_date.isoformat()}-{raw_digest}.json.gz"
                    )
                    _write_gzip_bytes(raw_path, body)
                    item = {
                        "station_id": station.station_id,
                        "local_date": target_date.isoformat(),
                        "market_unit": station.market_unit,
                        "source_page": target_url,
                        "network_url": response.url,
                        "body_sha256": raw_digest,
                        "raw_path": str(raw_path),
                        "observations": converted,
                        "final_wu_max": max(
                            float(observation["temperature_market_unit"])
                            for observation in converted
                        ),
                    }
                    _write_gzip_json(destination, item)
                    output[(station.station_id, target_date.isoformat())] = item
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(1.0 + attempt)
            if last_error is not None:
                item = {
                    "station_id": station.station_id,
                    "local_date": target_date.isoformat(),
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "observations": [],
                    "final_wu_max": None,
                }
                _write_gzip_json(destination, item)
                output[(station.station_id, target_date.isoformat())] = item
            await asyncio.sleep(0.1)
    finally:
        await cdp_session.detach()
        await page.close()
    return output


def _build_historical_rows(
    cfg: AppConfig,
    events_by_station: dict[str, list[dict[str, Any]]],
    forecasts: dict[tuple[str, str], dict[str, Any]],
    prices: dict[tuple[str, str], dict[str, Any]],
    wu_history: dict[tuple[str, str], dict[str, Any]],
    *,
    decision_hours: tuple[int, ...],
    observation_lag_minutes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = {
        "missing_wu_days": 0,
        "winner_wu_mismatches": 0,
        "winner_wu_mismatch_details": [],
        "decisions_without_prior_observation": 0,
        "decisions_without_complete_market_prices": 0,
        "included_decisions": 0,
    }
    forecast_indexes = {
        key: _hourly_index(payload.get("hourly", {})) for key, payload in forecasts.items()
    }
    for station_id, events in sorted(events_by_station.items()):
        station = _station_by_id(cfg, station_id)
        timezone = ZoneInfo(station.timezone)
        provider_indexes = {
            provider: forecast_indexes.get((station_id, provider), {})
            for provider in station.forecast_providers
        }
        for event in events:
            local_date = str(event["event_date"])
            wu = wu_history.get((station_id, local_date), {})
            observations = sorted(
                wu.get("observations", []), key=lambda item: int(item["timestamp_ns"])
            )
            final_wu_max = _safe_float(wu.get("final_wu_max"))
            if final_wu_max is None or not observations:
                audit["missing_wu_days"] += 1
                continue
            labels = [str(item["label"]) for item in event["contracts"]]
            derived_winner = _label_for_value(final_wu_max, labels, station.market_unit)
            if derived_winner != str(event["winner"]):
                audit["winner_wu_mismatches"] += 1
                audit["winner_wu_mismatch_details"].append(
                    {
                        "station_id": station_id,
                        "local_date": local_date,
                        "final_wu_max": final_wu_max,
                        "derived_bucket": derived_winner,
                        "resolved_bucket": event["winner"],
                    }
                )
                continue
            price_history = prices.get((station_id, str(event["event_id"])), {}).get(
                "history", {}
            )
            for decision_hour in sorted(set(decision_hours)):
                decision = datetime.combine(
                    date.fromisoformat(local_date), time(decision_hour), tzinfo=timezone
                )
                cutoff_ns = int(
                    (decision - timedelta(minutes=observation_lag_minutes))
                    .astimezone(UTC)
                    .timestamp()
                    * 1e9
                )
                known = [
                    item for item in observations if int(item["timestamp_ns"]) <= cutoff_ns
                ]
                if not known:
                    audit["decisions_without_prior_observation"] += 1
                    continue
                current_temperature = float(known[-1]["temperature_market_unit"])
                running_max = max(float(item["temperature_market_unit"]) for item in known)
                market_prices: dict[str, float] = {}
                market_price_ages: list[float] = []
                fee_rate: dict[str, float] = {}
                fee_exponent: dict[str, float] = {}
                fees_enabled: dict[str, bool] = {}
                decision_ts = int(decision.astimezone(UTC).timestamp())
                for contract in event["contracts"]:
                    label = str(contract["label"])
                    point = _latest_price_at(
                        price_history.get(str(contract["yes_token_id"]), []), decision_ts
                    )
                    if point is not None:
                        market_prices[label] = float(point["p"])
                        market_price_ages.append(decision_ts - int(point["t"]))
                    fee_rate[label] = _safe_float(contract.get("fee_rate")) or 0.0
                    fee_exponent[label] = _safe_float(contract.get("fee_exponent")) or 1.0
                    fees_enabled[label] = bool(contract.get("fees_enabled"))
                coverage = len(market_prices) / len(labels) if labels else 0.0
                if coverage < 0.999:
                    audit["decisions_without_complete_market_prices"] += 1
                forecast_features = _forecast_features(
                    provider_indexes, local_date, decision_hour
                )
                observation_features = _historical_observation_features(
                    known, decision, running_max
                )
                row = {
                    "run_id": "historical",
                    "station_id": station_id,
                    "city": station.city,
                    "source": "wunderground_historical_page",
                    "decision_mode": "historical_scheduled",
                    "trigger_source": "historical_clock",
                    "received_wall_ns": decision_ts * 1_000_000_000,
                    "observation_time": known[-1]["timestamp"],
                    "observation_key": f"historical|{station_id}|{local_date}|{decision_hour:02d}",
                    "local_date": local_date,
                    "day_history_complete": True,
                    "continuity_segment": 0,
                    "continuity_eligible": True,
                    "local_hour": float(decision_hour),
                    "minutes_since_local_midnight": decision_hour * 60,
                    "minutes_since_last_observation": (
                        decision_ts - int(known[-1]["timestamp_ns"]) / 1e9
                    )
                    / 60.0,
                    "temperature_market_unit": current_temperature,
                    "temperature_c": (
                        (current_temperature - 32.0) * 5.0 / 9.0
                        if station.market_unit == "F"
                        else current_temperature
                    ),


                    "running_max_market_unit": running_max,
                    "resolution_running_max_market_unit": running_max,
                    "running_max_source": "wunderground_resolution_page",
                    "market_distribution_json": json.dumps(
                        market_prices, sort_keys=True, ensure_ascii=False
                    ),
                    "market_labels_json": json.dumps(
                        labels, sort_keys=True, ensure_ascii=False
                    ),
                    "market_bid_json": "{}",
                    "market_ask_json": "{}",
                    "market_bid_size_json": "{}",
                    "market_ask_size_json": "{}",
                    "market_fee_rate_json": json.dumps(
                        fee_rate, sort_keys=True, ensure_ascii=False
                    ),
                    "market_fee_exponent_json": json.dumps(
                        fee_exponent, sort_keys=True, ensure_ascii=False
                    ),
                    "market_fees_enabled_json": json.dumps(
                        fees_enabled, sort_keys=True, ensure_ascii=False
                    ),
                    "market_ws_healthy": False,
                    "market_quote_coverage": coverage,
                    "market_price_max_age_seconds": max(market_price_ages, default=None),
                    "market_price_history_only": True,
                    "historical_point_in_time": True,
                    "observation_publication_safety_lag_minutes": observation_lag_minutes,
                    "event_id": event["event_id"],
                    "event_slug": event["event_slug"],
                    "winning_bucket": event["winner"],
                    "final_wu_max": final_wu_max,
                    "wu_target_matches_resolved_winner": True,
                    **station_time_features(decision, station),
                    **observation_features,
                    **forecast_features,
                }
                rows.append(row)
                audit["included_decisions"] += 1
    return sorted(rows, key=lambda row: int(row["received_wall_ns"])), audit


def _historical_observation_features(
    known: list[dict[str, Any]], decision: datetime, running_max: float
) -> dict[str, float]:
    latest = float(known[-1]["temperature_market_unit"])
    maximum_observation = max(
        known, key=lambda item: float(item["temperature_market_unit"])
    )
    decision_ns = int(decision.astimezone(UTC).timestamp() * 1e9)

    def previous_delta(hours: float) -> float | None:
        threshold = decision_ns - int(hours * 3600 * 1e9)
        candidates = [item for item in known if int(item["timestamp_ns"]) <= threshold]
        return latest - float(candidates[-1]["temperature_market_unit"]) if candidates else None

    output: dict[str, float] = {
        "historical_observation_count": float(len(known)),
        "hours_since_running_max": max(
            0.0,
            (decision_ns - int(maximum_observation["timestamp_ns"])) / 3.6e12,
        ),
        "temperature_below_running_max": running_max - latest,
    }
    for hours in (1.0, 3.0, 6.0):
        value = previous_delta(hours)
        if value is not None:
            output[f"temperature_change_{int(hours)}h"] = value
    return output


def _forecast_features(
    provider_indexes: dict[str, dict[str, dict[str, Any]]],
    local_date: str,
    decision_hour: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    maxima: list[float] = []
    remaining_maxima: list[float] = []
    for provider, index in provider_indexes.items():
        day_rows = [
            values
            for timestamp, values in index.items()
            if timestamp.startswith(f"{local_date}T")
        ]
        if not day_rows:
            continue
        remaining_rows = [
            values
            for timestamp, values in index.items()
            if timestamp.startswith(f"{local_date}T")
            and int(timestamp[11:13]) >= decision_hour
        ]
        prefix = f"historical_forecast_{provider}"
        day1_temperatures = _finite(
            row.get("temperature_2m_previous_day1") for row in day_rows
        )
        day2_temperatures = _finite(
            row.get("temperature_2m_previous_day2") for row in day_rows
        )
        remaining_temperatures = _finite(
            row.get("temperature_2m_previous_day1") for row in remaining_rows
        )
        if day1_temperatures:
            maximum = max(day1_temperatures)
            maxima.append(maximum)
            output[f"{prefix}_day1_max_c"] = maximum
            output[f"{prefix}_day1_min_c"] = min(day1_temperatures)
            output[f"{prefix}_day1_mean_c"] = sum(day1_temperatures) / len(
                day1_temperatures
            )
        if remaining_temperatures:
            remaining_maximum = max(remaining_temperatures)
            remaining_maxima.append(remaining_maximum)
            output[f"{prefix}_day1_remaining_max_c"] = remaining_maximum
        if day2_temperatures:
            output[f"{prefix}_day2_max_c"] = max(day2_temperatures)
            if day1_temperatures:
                output[f"{prefix}_max_revision_day1_minus_day2_c"] = max(
                    day1_temperatures
                ) - max(day2_temperatures)
        decision_row = index.get(f"{local_date}T{decision_hour:02d}:00", {})
        for variable in ("temperature_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m"):
            value = _safe_float(decision_row.get(f"{variable}_previous_day1"))
            if value is not None:
                output[f"{prefix}_{variable}_at_decision_day1"] = value
        aggregates = (
            ("dew_point_2m", "mean", lambda values: sum(values) / len(values)),
            ("cloud_cover", "mean", lambda values: sum(values) / len(values)),
            ("shortwave_radiation", "max", max),
            ("precipitation", "sum", sum),
            ("wind_speed_10m", "max", max),
            ("surface_pressure", "mean", lambda values: sum(values) / len(values)),
        )
        for variable, name, reducer in aggregates:
            values = _finite(row.get(f"{variable}_previous_day1") for row in day_rows)
            if values:
                output[f"{prefix}_{variable}_day1_{name}"] = float(reducer(values))
    if maxima:
        output["historical_model_consensus_day1_max_c"] = sum(maxima) / len(maxima)
        output["historical_model_spread_day1_max_c"] = max(maxima) - min(maxima)
    if remaining_maxima:
        output["historical_model_consensus_remaining_max_c"] = sum(
            remaining_maxima
        ) / len(remaining_maxima)
        output["historical_model_spread_remaining_max_c"] = max(remaining_maxima) - min(
            remaining_maxima
        )
    return output


async def _request_json(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            retry_delay: float | None = None
            async with semaphore:
                async with session.request(
                    method, url, params=params, json=json_body
                ) as response:
                    body = await response.read()
                    if response.status == 429:
                        retry_after = _safe_float(response.headers.get("Retry-After")) or 2.0
                        retry_delay = min(30.0, retry_after + attempt)
                        last_error = RuntimeError(
                            f"HTTP 429 from {url}; retry_after={retry_delay:.1f}s"
                        )
                    elif response.status >= 400:
                        raise RuntimeError(
                            f"HTTP {response.status} from {url}: {body[:500].decode(errors='replace')}"
                        )
                    else:
                        return orjson.loads(body)
            if retry_delay is not None:
                if attempt < 4:
                    await asyncio.sleep(retry_delay)
                continue
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < 4:
                await asyncio.sleep(min(16.0, 2.0**attempt))
    raise RuntimeError(f"Request failed after retries: {url}: {last_error}")


def _merge_hourly(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, list[Any]]:

    rows_by_time: dict[str, dict[str, Any]] = {}
    keys: set[str] = set()
    for payload in (existing, incoming):
        times = payload.get("time")
        if not isinstance(times, list):
            continue
        payload_keys = {
            str(key)
            for key, values in payload.items()
            if key != "time" and isinstance(values, list)
        }
        keys.update(payload_keys)
        for index, raw_time in enumerate(times):
            timestamp = str(raw_time)
            row = rows_by_time.setdefault(timestamp, {})
            for key in payload_keys:
                values = payload.get(key)
                if isinstance(values, list) and index < len(values):
                    row[key] = values[index]
    ordered_times = sorted(rows_by_time)
    return {
        "time": ordered_times,
        **{
            key: [rows_by_time[timestamp].get(key) for timestamp in ordered_times]
            for key in sorted(keys)
        },
    }


def _event_date(event: dict[str, Any]) -> date | None:


    raw = event.get("eventDate") or event.get("event_date") or event.get("endDate")
    if raw:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            return None
    return None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _hourly_index(hourly: dict[str, Any]) -> dict[str, dict[str, Any]]:
    times = hourly.get("time", []) if isinstance(hourly, dict) else []
    output: dict[str, dict[str, Any]] = {}
    for index, timestamp in enumerate(times):
        row: dict[str, Any] = {}
        for key, values in hourly.items():
            if key == "time" or not isinstance(values, list) or index >= len(values):
                continue
            row[str(key)] = values[index]
        output[str(timestamp)] = row
    return output


def _latest_price_at(history: Any, timestamp: int) -> dict[str, Any] | None:
    if not isinstance(history, list):
        return None
    candidates = [
        point
        for point in history
        if isinstance(point, dict)
        and _safe_float(point.get("t")) is not None
        and _safe_float(point.get("p")) is not None
        and int(float(point["t"])) <= timestamp
    ]
    return max(candidates, key=lambda point: int(float(point["t"])), default=None)


def _finite(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _safe_float(value)) is not None]


def _station_by_id(cfg: AppConfig, station_id: str) -> StationConfig:
    station = next(
        (item for item in cfg.stations.values() if item.station_id == station_id), None
    )
    if station is None:
        raise KeyError(station_id)
    return station


def _write_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(orjson.dumps(payload))
    temporary.replace(path)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return orjson.loads(handle.read())


def _write_gzip_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(payload)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
