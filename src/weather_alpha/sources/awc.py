from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from dateutil import parser as date_parser

from ..config import AppConfig
from ..domain import EventEnvelope
from ..http_client import TimedHttpClient
from ..parsing.metar import parse_awc_metar
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import datetime_to_ns, sleep_aligned

logger = logging.getLogger(__name__)


class AviationWeatherCollector:


    BASE_URL = "https://aviationweather.gov/api/data/metar"

    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink, http: TimedHttpClient) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.http = http
        self._seen: set[tuple[str, str]] = set()
        self._targets = {
            station.station_id: station for station in cfg.stations.values() if station.sources.awc
        }
        self._peer_targets: dict[str, list[str]] = defaultdict(list)
        for target_id, station in self._targets.items():
            for peer_id in station.peer_station_ids:
                peer_id = peer_id.upper()
                if peer_id != target_id:
                    self._peer_targets[peer_id].append(target_id)
        self._station_ids = sorted(set(self._targets) | set(self._peer_targets))

    async def run(self) -> None:
        if not self._station_ids:
            return
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except CriticalSinkError:
                raise
            except Exception as exc:
                logger.exception("Aviation Weather collector failure")
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="awc_metar",
                        event_type="collector_error",
                        payload={"error": repr(exc)},
                    )
                )
            await sleep_aligned(self.cfg.collector.awc_poll_seconds)

    async def poll_once(self) -> None:
        params = {"ids": ",".join(self._station_ids), "format": "json", "hours": "2"}
        response = await self.http.get(self.BASE_URL, params=params)
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="awc_metar",
                event_type="http_response",
                payload={
                    "url": response.url,
                    "status": response.status,
                    "headers": response.headers,
                    "body_bytes": len(response.body),
                    "requested_station_ids": self._station_ids,
                },
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )

        if response.status == 204:
            return
        if response.status != 200:
            raise RuntimeError(f"AWC returned HTTP {response.status}: {response.body[:300]!r}")
        data = response.json()
        if isinstance(data, dict):

            data = data.get("data") or data.get("features") or data.get("metars") or []
        if not isinstance(data, list):
            raise TypeError(f"Unexpected AWC payload type: {type(data).__name__}")

        parsed_items: list[tuple[str, dict[str, Any], list[str], datetime | None]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "properties" in item and isinstance(item["properties"], dict):
                merged: dict[str, Any] = dict(item["properties"])
                geometry = item.get("geometry") or {}
                coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
                if isinstance(coords, list) and len(coords) >= 2:
                    merged.setdefault("lon", coords[0])
                    merged.setdefault("lat", coords[1])
                item = merged
            normalized, errors = parse_awc_metar(item)
            observed_station_id = str(normalized.get("station_id") or "").upper()
            if observed_station_id not in self._station_ids:
                continue
            obs_time = normalized.get("observation_time")
            obs_dt: datetime | None = date_parser.isoparse(obs_time) if obs_time else None
            parsed_items.append((observed_station_id, normalized, errors, obs_dt))

        latest_by_station: dict[str, datetime] = {}
        for observed_station_id, _, _, obs_dt in parsed_items:
            if obs_dt is not None and (
                observed_station_id not in latest_by_station
                or obs_dt > latest_by_station[observed_station_id]
            ):
                latest_by_station[observed_station_id] = obs_dt


        parsed_items.sort(key=lambda item: item[3] or datetime.min.replace(tzinfo=UTC))
        for observed_station_id, normalized, errors, obs_dt in parsed_items:
            dedupe_key = normalized.get("raw_observation") or normalized.get("observation_time")
            seen_key = (observed_station_id, str(dedupe_key))
            if not dedupe_key or seen_key in self._seen:
                continue
            self._seen.add(seen_key)
            backfill = bool(
                obs_dt is not None
                and observed_station_id in latest_by_station
                and obs_dt < latest_by_station[observed_station_id]
            )
            normalized = {
                **normalized,
                "backfill": backfill,
                "is_latest_in_response": not backfill,
            }

            if observed_station_id in self._targets:
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="awc_metar",
                        event_type="weather_observation",
                        payload=normalized,
                        station_id=observed_station_id,
                        request_started_wall_ns=response.request_started_wall_ns,
                        response_received_wall_ns=response.response_received_wall_ns,
                        observation_timestamp_ns=datetime_to_ns(obs_dt),
                        parse_errors=errors,
                        received_wall_ns=response.response_received_wall_ns,
                        received_mono_ns=response.response_received_mono_ns,
                    )
                )

            for target_id in self._peer_targets.get(observed_station_id, []):
                peer_payload = {
                    **normalized,
                    "peer_station_id": observed_station_id,
                    "target_station_id": target_id,
                }
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="awc_peer_metar",
                        event_type="peer_weather_observation",
                        payload=peer_payload,


                        station_id=target_id,
                        request_started_wall_ns=response.request_started_wall_ns,
                        response_received_wall_ns=response.response_received_wall_ns,
                        observation_timestamp_ns=datetime_to_ns(obs_dt),
                        parse_errors=errors,
                        received_wall_ns=response.response_received_wall_ns,
                        received_mono_ns=response.response_received_mono_ns,
                    )
                )
