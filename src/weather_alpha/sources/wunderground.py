from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from ..domain import EventEnvelope
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import datetime_to_ns, now_pair, sleep_aligned

logger = logging.getLogger(__name__)

_TIME_KEYS = ("valid_time_gmt", "validTimeUtc", "obsTimeUtc", "epoch", "timestamp")
_TEMP_KEYS = ("temp", "temperature", "metric_temp", "imperial_temp")


class WundergroundCollector:


    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.stations = [s for s in cfg.stations.values() if s.sources.wunderground]

    async def run(self) -> None:
        if not self.stations:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="wunderground_browser",
                    event_type="collector_disabled",
                    payload={
                        "reason": "Playwright is not installed. Run pip install -e '.[wunderground]' and playwright install chromium"
                    },
                )
            )
            return

        delay = self.cfg.collector.reconnect_min_seconds
        while True:
            try:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(headless=True)
                    context = await browser.new_context(
                        locale="en-US",
                        timezone_id="UTC",
                        service_workers="block",
                    )
                    tasks = [
                        asyncio.create_task(self._station_loop(context, station))
                        for station in self.stations
                    ]
                    delay = self.cfg.collector.reconnect_min_seconds
                    try:
                        await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await context.close()
                        await browser.close()
            except asyncio.CancelledError:
                raise
            except CriticalSinkError:
                raise
            except Exception as exc:
                logger.exception("Weather Underground browser collector failure")
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="wunderground_browser",
                        event_type="collector_error",
                        payload={"error": repr(exc), "retry_seconds": delay},
                    )
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.cfg.collector.reconnect_max_seconds)

    async def _station_loop(self, context: Any, station: StationConfig) -> None:
        page = await context.new_page()


        cdp_session = await context.new_cdp_session(page)
        await cdp_session.send("Network.enable")
        await cdp_session.send("Network.setCacheDisabled", {"cacheDisabled": True})
        captured: list[dict[str, Any]] = []
        capture_lock = asyncio.Lock()
        pending_capture_tasks: set[asyncio.Task[None]] = set()
        history_response_received = asyncio.Event()
        first_observation_by_local_date: dict[date, int] = {}
        generation = 0

        async def capture_response(response: Any, response_generation: int) -> None:
            url = response.url
            content_type = (response.headers.get("content-type") or "").lower()
            if "json" not in content_type:
                return
            if "weather.com" not in url and "wunderground.com" not in url:
                return
            lowered_url = url.lower()
            if not any(
                marker in lowered_url for marker in ("observations", "historical", "history")
            ):
                return
            if any(marker in lowered_url for marker in ("forecast", "almanac", "pollen")):
                return
            headers_stamp = now_pair()
            entry: dict[str, Any] = {
                "url": url,
                "status": response.status,
                "headers_received_wall_ns": headers_stamp.wall_ns,
                "headers_received_mono_ns": headers_stamp.mono_ns,
                "generation": response_generation,
                "observations": [],
            }
            try:
                body = await response.body()
                if len(body) > 5_000_000:
                    raise ValueError(f"WU history response too large: {len(body)} bytes")
                import orjson

                payload = orjson.loads(body)
                extracted = self._extract_observations(payload, url, station)
                digest = hashlib.sha256(body).hexdigest()
                blob_path = await asyncio.to_thread(
                    self._store_blob, station.station_id, digest, body
                )
                stamp = now_pair()
                entry.update(
                    {
                        "received_wall_ns": stamp.wall_ns,
                        "received_mono_ns": stamp.mono_ns,
                        "body_bytes": len(body),
                        "body_sha256": digest,
                        "blob_path": str(blob_path),
                        "observations": extracted,
                    }
                )
            except Exception as exc:
                stamp = now_pair()
                entry.update(
                    {
                        "received_wall_ns": stamp.wall_ns,
                        "received_mono_ns": stamp.mono_ns,
                        "capture_error": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                )
            async with capture_lock:
                captured.append(entry)
            history_response_received.set()

        def schedule_capture(response: Any) -> None:
            task = asyncio.create_task(capture_response(response, generation))
            pending_capture_tasks.add(task)
            task.add_done_callback(pending_capture_tasks.discard)

        page.on("response", schedule_capture)
        try:
            cycle = 0
            while True:
                local_today = datetime.now(ZoneInfo(station.timezone)).date()


                target_date = local_today - timedelta(days=1) if cycle % 10 == 9 else local_today
                target_url = self._dated_url(station.resolution_url, target_date)
                cycle += 1
                generation += 1
                current_generation = generation
                async with capture_lock:
                    captured.clear()
                history_response_received.clear()
                start = now_pair()
                errors: list[str] = []
                try:
                    await page.goto(
                        target_url,
                        wait_until="domcontentloaded",
                        timeout=int(self.cfg.collector.request_timeout_seconds * 1000),
                    )
                    try:
                        await asyncio.wait_for(
                            history_response_received.wait(),
                            timeout=max(
                                5.0,
                                min(15.0, self.cfg.collector.request_timeout_seconds),
                            ),
                        )
                    except TimeoutError:
                        errors.append("wu_history_response_timeout")

                    await page.wait_for_timeout(500)


                    if pending_capture_tasks:
                        await asyncio.gather(*list(pending_capture_tasks), return_exceptions=True)
                    body_text = await page.locator("body").inner_text(timeout=5000)
                    title = await page.title()
                except Exception as exc:
                    body_text = ""
                    title = ""
                    errors.append(repr(exc))
                end = now_pair()
                async with capture_lock:
                    network = [
                        entry for entry in captured if entry.get("generation") == current_generation
                    ]
                station_tz = ZoneInfo(station.timezone)
                for entry in network:
                    for observation in entry.get("observations", []):
                        if str(observation.get("station_id") or "").upper() != station.station_id:
                            continue
                        observation_ns = int(observation["timestamp_ns"])
                        observation_date = (
                            datetime.fromtimestamp(observation_ns / 1e9, tz=UTC)
                            .astimezone(station_tz)
                            .date()
                        )
                        first_observation_by_local_date[observation_date] = min(
                            observation_ns,
                            first_observation_by_local_date.get(observation_date, observation_ns),
                        )
                summary = self._summarize(
                    station,
                    network,
                    body_text,
                    target_date,
                    first_observation_by_local_date=first_observation_by_local_date,
                )
                capture_errors = [
                    str(entry["capture_error"]) for entry in network if entry.get("capture_error")
                ]
                errors.extend(f"wu_capture_error: {error}" for error in capture_errors)
                if summary["observation_count"] == 0:
                    errors.append("wu_no_target_station_observations")
                if summary["station_mismatch_count"]:
                    errors.append("wu_station_mismatch_observations_rejected")
                if not summary["unit_verified"]:
                    errors.append("wu_temperature_unit_unverified")
                first_network = min(
                    network, key=lambda entry: entry["received_wall_ns"], default=None
                )
                first_network_wall_ns = (
                    first_network["received_wall_ns"] if first_network else end.wall_ns
                )
                first_network_mono_ns = (
                    first_network["received_mono_ns"] if first_network else end.mono_ns
                )
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="wunderground_browser",
                        event_type="resolution_snapshot",
                        payload={
                            "url": target_url,
                            "configured_base_url": station.resolution_url,
                            "target_date": target_date.isoformat(),
                            "title": title,
                            "body_text_sample": body_text[:4000],
                            "network_responses": network,
                            **summary,
                        },
                        station_id=station.station_id,
                        request_started_wall_ns=start.wall_ns,
                        response_received_wall_ns=end.wall_ns,
                        provider_timestamp_ns=summary.get("latest_observation_timestamp_ns"),
                        observation_timestamp_ns=summary.get("latest_observation_timestamp_ns"),
                        parse_errors=errors,
                        received_wall_ns=first_network_wall_ns,
                        received_mono_ns=first_network_mono_ns,
                    )
                )
                await sleep_aligned(self.cfg.collector.wunderground_poll_seconds, phase_seconds=1.0)
                if cycle >= self.cfg.collector.wunderground_browser_recycle_cycles:
                    return
        finally:
            for task in list(pending_capture_tasks):
                task.cancel()
            if pending_capture_tasks:
                await asyncio.gather(*list(pending_capture_tasks), return_exceptions=True)
            await cdp_session.detach()
            await page.close()

    def _store_blob(self, station_id: str, digest: str, body: bytes) -> Path:
        blob_dir = self.cfg.data_dir / "runs" / self.run_id / "blobs" / "wunderground" / station_id
        blob_dir.mkdir(parents=True, exist_ok=True)
        destination = blob_dir / f"{digest}.json.gz"
        if not destination.exists():
            temporary = Path(str(destination) + ".tmp")
            with gzip.open(temporary, "wb", compresslevel=6) as handle:
                handle.write(body)
            temporary.replace(destination)
        return destination

    @staticmethod
    def _dated_url(base_url: str, target_date: date) -> str:
        clean = re.sub(r"/date/\d{4}-\d{1,2}-\d{1,2}/?$", "", base_url.rstrip("/"))
        return f"{clean}/date/{target_date.year}-{target_date.month}-{target_date.day}"

    @staticmethod
    def _extract_observations(
        payload: Any, url: str, station: StationConfig
    ) -> list[dict[str, Any]]:

        lowered_url = url.lower()
        if not any(marker in lowered_url for marker in ("observations", "historical", "history")):
            return []
        if any(marker in lowered_url for marker in ("forecast", "almanac", "pollen")):
            return []
        query = parse_qs(urlparse(url).query)
        units_hint = str((query.get("units") or [""])[0]).lower()
        if units_hint in {"m", "metric"}:
            default_unit = "C"
        elif units_hint in {"e", "imperial"}:
            default_unit = "F"
        else:
            default_unit = None

        found: list[dict[str, Any]] = []

        def append_observation(
            raw_time: Any,
            raw_temp: Any,
            *,
            unit: str | None,
            node: dict[str, Any],
        ) -> None:
            timestamp_ns = WundergroundCollector._timestamp_ns(raw_time)
            temperature = WundergroundCollector._number(raw_temp)
            if (
                timestamp_ns is None
                or temperature is None
                or unit not in {"C", "F"}
                or not (-120 <= temperature <= 170)
            ):
                return
            observed_station_id = str(
                node.get("icaoCode")
                or node.get("stationID")
                or node.get("stationId")
                or node.get("obs_id")
                or node.get("key")
                or station.station_id
            ).upper()
            found.append(
                {
                    "timestamp_ns": timestamp_ns,
                    "timestamp": datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC).isoformat(),
                    "temperature": temperature,
                    "unit": unit,
                    "unit_verified": True,
                    "station_id": observed_station_id,
                    "source_keys": sorted(node.keys()),
                }
            )

        def normalized_unit(value: Any) -> str | None:
            text = str(value or "").strip().lower()
            if text in {"m", "metric", "c", "celsius", "°c"}:
                return "C"
            if text in {"e", "imperial", "f", "fahrenheit", "°f"}:
                return "F"
            return None

        def node_unit(node: dict[str, Any], inherited: str | None) -> str | None:
            for key in ("units", "unit", "temperatureUnit", "temperature_unit"):
                if unit := normalized_unit(node.get(key)):
                    return unit
            metadata = node.get("metadata")
            if isinstance(metadata, dict):
                for key in ("units", "unit", "temperatureUnit", "temperature_unit"):
                    if unit := normalized_unit(metadata.get(key)):
                        return unit
            return inherited

        def temperature_series(
            value: Any, fallback_unit: str | None
        ) -> tuple[list[Any], str | None]:
            if isinstance(value, dict):

                preferred_unit = fallback_unit or station.market_unit
                explicit_keys = (
                    (("metric", "C"), ("celsius", "C"), ("imperial", "F"), ("fahrenheit", "F"))
                    if preferred_unit == "C"
                    else (("imperial", "F"), ("fahrenheit", "F"), ("metric", "C"), ("celsius", "C"))
                )
                candidates = list(explicit_keys)
                if fallback_unit is not None:
                    candidates.append(("value", fallback_unit))
                for key, unit in candidates:
                    if key in value:
                        selected = value[key]
                        return (selected if isinstance(selected, list) else [selected]), unit
                return [], fallback_unit
            return (value if isinstance(value, list) else [value]), fallback_unit

        def walk(node: Any, inherited_unit: str | None = default_unit) -> None:
            if isinstance(node, dict):
                effective_unit = node_unit(node, inherited_unit)
                time_key = next((key for key in _TIME_KEYS if key in node), None)
                temp_key = next((key for key in _TEMP_KEYS if key in node), None)
                if time_key and temp_key:
                    times = node.get(time_key)
                    time_values = times if isinstance(times, list) else [times]
                    temp_values, explicit_unit = temperature_series(
                        node.get(temp_key), effective_unit
                    )
                    if len(time_values) == len(temp_values):
                        for raw_time, raw_temp in zip(time_values, temp_values, strict=True):
                            append_observation(
                                raw_time,
                                raw_temp,
                                unit=explicit_unit,
                                node=node,
                            )
                    elif len(time_values) == 1 and len(temp_values) == 1:
                        append_observation(
                            time_values[0], temp_values[0], unit=explicit_unit, node=node
                        )
                for value in node.values():
                    walk(value, effective_unit)
            elif isinstance(node, list):
                for value in node:
                    walk(value, inherited_unit)

        walk(payload)
        unique: dict[tuple[int, float, str, str], dict[str, Any]] = {}
        for obs in found:
            unique[
                (
                    obs["timestamp_ns"],
                    obs["temperature"],
                    obs["unit"],
                    str(obs["station_id"]),
                )
            ] = obs
        return sorted(unique.values(), key=lambda item: item["timestamp_ns"])

    @staticmethod
    def _summarize(
        station: StationConfig,
        network: list[dict[str, Any]],
        body_text: str,
        target_date: date,
        *,
        first_observation_by_local_date: dict[date, int] | None = None,
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        for response in network:
            observations.extend(response.get("observations", []))
        tz = ZoneInfo(station.timezone)
        observed_station_ids = sorted(
            {str(obs.get("station_id") or "").upper() for obs in observations}
        )
        target_observations = [
            obs
            for obs in observations
            if str(obs.get("station_id") or "").upper() == station.station_id
        ]
        today_obs: list[dict[str, Any]] = []
        next_day_obs: list[dict[str, Any]] = []
        for obs in target_observations:
            local_date = (
                datetime.fromtimestamp(obs["timestamp_ns"] / 1e9, tz=UTC).astimezone(tz).date()
            )
            if local_date == target_date:
                today_obs.append(obs)
            elif local_date == target_date + timedelta(days=1):
                next_day_obs.append(obs)
        latest = max(today_obs, key=lambda item: item["timestamp_ns"], default=None)
        first_next_day = min(next_day_obs, key=lambda item: item["timestamp_ns"], default=None)
        next_day_evidence_ns = first_next_day["timestamp_ns"] if first_next_day else None
        if first_observation_by_local_date:
            persisted_evidence = first_observation_by_local_date.get(
                target_date + timedelta(days=1)
            )
            if persisted_evidence is not None:
                next_day_evidence_ns = min(
                    next_day_evidence_ns or persisted_evidence,
                    persisted_evidence,
                )
        temps_market_unit = []
        for obs in today_obs:
            value = obs["temperature"]
            unit = obs["unit"]
            if unit != station.market_unit:
                value = value * 9 / 5 + 32 if unit == "C" else (value - 32) * 5 / 9
            temps_market_unit.append(value)


        visible_day_high = WundergroundCollector._visible_day_high(
            body_text, station.market_unit
        )
        if visible_day_high is not None:
            temps_market_unit.append(visible_day_high)

        visible_candidates = [
            float(value)
            for value in re.findall(r"(-?\d+(?:\.\d+)?)\s*°", body_text)
            if -100 <= float(value) <= 150
        ]
        return {
            "local_date": target_date.isoformat(),
            "market_unit": station.market_unit,
            "observation_count": len(today_obs),
            "observed_station_ids": observed_station_ids,
            "station_mismatch_count": len(observations) - len(target_observations),
            "unit_verified": (
                bool(target_observations)
                and all(obs.get("unit") in {"C", "F"} for obs in target_observations)
            )
            or visible_day_high is not None,
            "resolution_final": bool(today_obs and next_day_evidence_ns is not None),
            "finalization_observation_timestamp_ns": next_day_evidence_ns,
            "latest_temperature": (
                WundergroundCollector._to_unit(
                    latest["temperature"], latest["unit"], station.market_unit
                )
                if latest
                else None
            ),
            "latest_observation_timestamp_ns": latest["timestamp_ns"] if latest else None,
            "daily_max_from_network": max(temps_market_unit) if temps_market_unit else None,
            "visible_temperature_candidates": visible_candidates[:100],
        }

    @staticmethod
    def _visible_day_high(body_text: str, target_unit: str) -> float | None:

        match = re.search(
            r"Day\s+High\s*&\s*Low\s+HIGH\s+(-?\d+(?:\.\d+)?)\s*°\s*([CF])\b",
            body_text,
            flags=re.IGNORECASE,
        )
        if match is None or target_unit not in {"C", "F"}:
            return None
        value = float(match.group(1))
        source_unit = match.group(2).upper()
        if not (-100.0 <= value <= 170.0):
            return None
        return WundergroundCollector._to_unit(value, source_unit, target_unit)

    @staticmethod
    def _to_unit(value: float, source: str, target: str) -> float:
        if source == target:
            return value
        return value * 9 / 5 + 32 if source == "C" else (value - 32) * 5 / 9

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _timestamp_ns(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            seconds = float(value)
            if seconds > 10_000_000_000:
                seconds /= 1000
            return int(seconds * 1e9)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return datetime_to_ns(parsed)
