from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import orjson

from ..config import AppConfig, StationConfig
from ..domain import EventEnvelope
from ..http_client import TimedHttpClient
from ..parsing.buckets import c_to_f, round_half_up
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import sleep_aligned

logger = logging.getLogger(__name__)


PROVIDERS: dict[str, str] = {
    "best_match": "https://api.open-meteo.com/v1/forecast",
    "ecmwf": "https://api.open-meteo.com/v1/ecmwf",
    "dwd_icon": "https://api.open-meteo.com/v1/dwd-icon",
    "gfs_hrrr": "https://api.open-meteo.com/v1/gfs",
}

SURFACE_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
]
UPPER_AIR_VARIABLES = ["temperature_925hPa", "temperature_850hPa"]


class ForecastCollector:


    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink, http: TimedHttpClient) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.http = http
        self.stations = [s for s in cfg.stations.values() if s.sources.forecasts]
        self._seen_hash: dict[tuple[str, str], str] = {}
        self._last_summary: dict[tuple[str, str], dict[str, Any]] = {}
        self._request_slots = asyncio.Semaphore(cfg.collector.forecast_concurrency)

    async def run(self) -> None:
        if not self.stations:
            return
        while True:
            call_specs = [
                (station, provider, self.poll_provider(station, provider, url))
                for station in self.stations
                for provider, url in self._providers_for(station).items()
            ]
            results = await asyncio.gather(
                *(coroutine for _, _, coroutine in call_specs), return_exceptions=True
            )
            for (station, provider, _), result in zip(call_specs, results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, CriticalSinkError):
                    raise result
                if isinstance(result, Exception):
                    logger.warning(
                        "Forecast collector request failed for %s/%s: %r",
                        station.station_id,
                        provider,
                        result,
                    )
                    await self.sink.append(
                        EventEnvelope.build(
                            run_id=self.run_id,
                            source=f"forecast_{provider}",
                            event_type="collector_error",
                            payload={
                                "error_type": type(result).__name__,
                                "error": str(result)[:1000],
                            },
                            station_id=station.station_id,
                        )
                    )
            await sleep_aligned(self.cfg.collector.forecast_poll_seconds, phase_seconds=2.0)

    @staticmethod
    def _providers_for(station: StationConfig) -> dict[str, str]:
        requested = station.forecast_providers
        if not requested:
            requested = ["best_match", "ecmwf"]
            requested.append("gfs_hrrr" if station.station_id.startswith("K") else "dwd_icon")
        unknown = sorted(set(requested) - set(PROVIDERS))
        if unknown:
            raise ValueError(
                f"Unknown forecast providers for {station.station_id}: {unknown}; "
                f"available={sorted(PROVIDERS)}"
            )
        return {provider: PROVIDERS[provider] for provider in requested}

    async def poll_provider(self, station: StationConfig, provider: str, url: str) -> None:
        params = self._params(station, include_upper_air=True)
        maximum_attempts = self.cfg.collector.forecast_retry_attempts
        response = None
        for attempt in range(1, maximum_attempts + 1):
            try:
                response = await self._get(url, params)
                await self._record_http_response(
                    station, provider, response, attempt, maximum_attempts
                )
                if response.status == 400 and provider != "best_match":


                    params = self._params(station, include_upper_air=False)
                    response = await self._get(url, params)
                    await self._record_http_response(
                        station, provider, response, attempt, maximum_attempts
                    )
            except (asyncio.CancelledError, CriticalSinkError):
                raise
            except Exception as exc:
                if attempt >= maximum_attempts:
                    raise
                delay = self._retry_delay(attempt)
                await self._record_retry(
                    station,
                    provider,
                    attempt,
                    maximum_attempts,
                    delay,
                    error=repr(exc),
                )
                await asyncio.sleep(delay)
                continue

            if response.status == 200:
                break
            if (
                response.status in {408, 425, 429, 500, 502, 503, 504}
                and attempt < maximum_attempts
            ):
                delay = self._retry_delay(attempt, response.headers)
                await self._record_retry(
                    station,
                    provider,
                    attempt,
                    maximum_attempts,
                    delay,
                    status=response.status,
                )
                await asyncio.sleep(delay)
                continue
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source=f"forecast_{provider}",
                    event_type="collector_error",
                    payload={
                        "status": response.status,
                        "attempts": attempt,
                        "body_sample": response.body[:1000].decode("utf-8", errors="replace"),
                    },
                    station_id=station.station_id,
                    received_wall_ns=response.response_received_wall_ns,
                    received_mono_ns=response.response_received_mono_ns,
                )
            )
            return

        if response is None or response.status != 200:
            return

        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected forecast payload type: {type(payload).__name__}")
        digest = forecast_content_digest(payload)
        key = (station.station_id, provider)
        previous_digest = self._seen_hash.get(key)
        if previous_digest == digest:
            return
        summary = summarize_forecast(payload, station)
        previous_summary = self._last_summary.get(key)
        self._seen_hash[key] = digest
        self._last_summary[key] = summary
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source=f"forecast_{provider}",
                event_type="forecast_snapshot",
                payload={
                    "provider": provider,
                    "content_sha256": digest,
                    "previous_content_sha256": previous_digest,
                    "snapshot_kind": (
                        "initial"
                        if previous_summary is None
                        else (
                            "day_rollover"
                            if previous_summary.get("local_date") != summary.get("local_date")
                            else "content_change"
                        )
                    ),
                    "summary": summary,
                    "forecast": payload,
                },
                station_id=station.station_id,
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )
        revision = forecast_revision_payload(
            previous_summary,
            summary,
            station,
            provider=provider,
            previous_digest=previous_digest,
            current_digest=digest,
        )
        if revision is not None:
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source=f"forecast_{provider}",
                    event_type="forecast_revision",
                    payload=revision,
                    station_id=station.station_id,
                    request_started_wall_ns=response.request_started_wall_ns,
                    response_received_wall_ns=response.response_received_wall_ns,
                    received_wall_ns=response.response_received_wall_ns,
                    received_mono_ns=response.response_received_mono_ns,
                )
            )

    async def _get(self, url: str, params: dict[str, Any]):
        async with self._request_slots:
            return await self.http.get(url, params=params)

    async def _record_http_response(
        self,
        station: StationConfig,
        provider: str,
        response,
        attempt: int,
        maximum_attempts: int,
    ) -> None:
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source=f"forecast_{provider}",
                event_type="http_response",
                payload={
                    "url": response.url,
                    "status": response.status,
                    "headers": response.headers,
                    "body_bytes": len(response.body),
                    "attempt": attempt,
                    "maximum_attempts": maximum_attempts,
                },
                station_id=station.station_id,
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )

    async def _record_retry(
        self,
        station: StationConfig,
        provider: str,
        attempt: int,
        maximum_attempts: int,
        delay: float,
        *,
        status: int | None = None,
        error: str | None = None,
    ) -> None:
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source=f"forecast_{provider}",
                event_type="retry_scheduled",
                payload={
                    "attempt": attempt,
                    "maximum_attempts": maximum_attempts,
                    "delay_seconds": delay,
                    "status": status,
                    "error": error,
                },
                station_id=station.station_id,
            )
        )

    def _retry_delay(self, attempt: int, headers: dict[str, str] | None = None) -> float:
        configured = self.cfg.collector.forecast_retry_base_seconds * (2 ** (attempt - 1))
        retry_after = 0.0
        for key, value in (headers or {}).items():
            if key.lower() != "retry-after":
                continue
            try:
                retry_after = max(0.0, float(value))
            except ValueError:
                pass
            break
        return min(
            self.cfg.collector.forecast_retry_max_seconds,
            max(configured, retry_after),
        )

    @staticmethod
    def _params(station: StationConfig, *, include_upper_air: bool) -> dict[str, Any]:
        variables = list(SURFACE_VARIABLES)
        if include_upper_air:
            variables.extend(UPPER_AIR_VARIABLES)
        return {
            "latitude": station.latitude,
            "longitude": station.longitude,
            "elevation": station.elevation_m if station.elevation_m is not None else "nan",
            "hourly": ",".join(variables),
            "daily": "temperature_2m_max,temperature_2m_min,shortwave_radiation_sum,precipitation_sum",
            "timezone": station.timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kn",
            "forecast_days": 3,
            "past_hours": 6,
            "cell_selection": "nearest",
        }


def summarize_forecast(payload: dict[str, Any], station: StationConfig) -> dict[str, Any]:
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), dict) else {}
    times = hourly.get("time") if isinstance(hourly, dict) else []
    temps = hourly.get("temperature_2m") if isinstance(hourly, dict) else []
    now = datetime.now(ZoneInfo(station.timezone))
    local_date = now.date()
    rows: list[tuple[datetime, float]] = []
    for raw_time, raw_temp in zip(times or [], temps or [], strict=False):
        try:
            dt = datetime.fromisoformat(str(raw_time))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(station.timezone))
            temp = float(raw_temp)
        except (TypeError, ValueError):
            continue
        if dt.date() == local_date:
            rows.append((dt, temp))
    remaining = [
        (dt, temp) for dt, temp in rows if dt >= now.replace(minute=0, second=0, microsecond=0)
    ]
    daily_max = max((temp for _, temp in rows), default=None)
    remaining_max = max((temp for _, temp in remaining), default=None)
    max_time = None
    if rows:
        max_time = max(rows, key=lambda row: row[1])[0].isoformat()
    return {
        "local_date": local_date.isoformat(),
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "grid_elevation": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "generationtime_ms": payload.get("generationtime_ms"),
        "today_hourly_max_c": daily_max,
        "remaining_hourly_max_c": remaining_max,
        "today_hourly_max_time": max_time,
        "hourly_points_today": len(rows),
    }


def forecast_content_digest(payload: dict[str, Any]) -> str:

    stable_payload = dict(payload)
    stable_payload.pop("generationtime_ms", None)
    return hashlib.sha256(orjson.dumps(stable_payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def forecast_revision_payload(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    station: StationConfig,
    *,
    provider: str,
    previous_digest: str | None,
    current_digest: str,
) -> dict[str, Any] | None:

    if previous is None or previous.get("local_date") != current.get("local_date"):
        return None
    previous_today = _finite_float(previous.get("today_hourly_max_c"))
    current_today = _finite_float(current.get("today_hourly_max_c"))
    previous_remaining = _finite_float(previous.get("remaining_hourly_max_c"))
    current_remaining = _finite_float(current.get("remaining_hourly_max_c"))
    today_changed = _values_changed(previous_today, current_today)
    remaining_changed = _values_changed(previous_remaining, current_remaining)
    if not today_changed and not remaining_changed:
        return None

    previous_market = _to_market_unit(previous_today, station.market_unit)
    current_market = _to_market_unit(current_today, station.market_unit)
    previous_bucket = round_half_up(previous_market) if previous_market is not None else None
    current_bucket = round_half_up(current_market) if current_market is not None else None
    return {
        "provider": provider,
        "local_date": current.get("local_date"),
        "market_unit": station.market_unit,
        "previous_content_sha256": previous_digest,
        "content_sha256": current_digest,
        "previous_today_max_c": previous_today,
        "current_today_max_c": current_today,
        "today_max_delta_c": (
            current_today - previous_today
            if previous_today is not None and current_today is not None
            else None
        ),
        "previous_remaining_max_c": previous_remaining,
        "current_remaining_max_c": current_remaining,
        "remaining_max_delta_c": (
            current_remaining - previous_remaining
            if previous_remaining is not None and current_remaining is not None
            else None
        ),
        "previous_market_max": previous_market,
        "current_market_max": current_market,
        "previous_bucket_value": previous_bucket,
        "current_bucket_value": current_bucket,
        "bucket_changed": (
            previous_bucket is not None
            and current_bucket is not None
            and previous_bucket != current_bucket
        ),
        "today_max_changed": today_changed,
        "remaining_max_changed": remaining_changed,
        "previous_max_time": previous.get("today_hourly_max_time"),
        "current_max_time": current.get("today_hourly_max_time"),
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed and abs(parsed) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _values_changed(previous: float | None, current: float | None) -> bool:
    if previous is None or current is None:
        return previous != current
    return abs(current - previous) > 1e-9


def _to_market_unit(value_c: float | None, market_unit: str) -> float | None:
    if value_c is None:
        return None
    return c_to_f(value_c) if market_unit == "F" else value_c
