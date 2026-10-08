from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta

from ..config import AppConfig, StationConfig
from ..domain import EventEnvelope
from ..http_client import TimedHttpClient
from ..parsing.metar import parse_awc_metar
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import datetime_to_ns, sleep_aligned

logger = logging.getLogger(__name__)

_RECEIPT_LINE = re.compile(r"^(?P<date>\d{4}/\d{2}/\d{2})\s+(?P<time>\d{2}:\d{2})\s*$")
_METAR_TIME = re.compile(r"(?:^|\s)(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})Z(?:\s|$)")


class NwsTgftpCollector:


    BASE_URL = "https://tgftp.nws.noaa.gov/data/observations/metar/stations"

    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink, http: TimedHttpClient) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.http = http
        self.stations = [s for s in cfg.stations.values() if s.sources.nws_tgftp]
        self._seen_raw: dict[str, str] = {}
        self._validators: dict[str, dict[str, str]] = {}

    async def run(self) -> None:
        if not self.stations:
            return
        while True:
            results = await asyncio.gather(
                *(self.poll_station(station) for station in self.stations),
                return_exceptions=True,
            )
            for station, result in zip(self.stations, results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, CriticalSinkError):
                    raise result
                if isinstance(result, Exception):
                    logger.warning(
                        "NWS TGFTP collector failed for %s: %r", station.station_id, result
                    )
                    await self.sink.append(
                        EventEnvelope.build(
                            run_id=self.run_id,
                            source="nws_tgftp_metar",
                            event_type="collector_error",
                            payload={"error": repr(result)},
                            station_id=station.station_id,
                        )
                    )
            await sleep_aligned(self.cfg.collector.nws_tgftp_poll_seconds, phase_seconds=0.05)

    async def poll_station(self, station: StationConfig) -> None:
        url = f"{self.BASE_URL}/{station.station_id}.TXT"
        headers = dict(self._validators.get(station.station_id, {}))
        response = await self.http.get(url, headers=headers or None)
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="nws_tgftp_metar",
                event_type="http_response",
                payload={
                    "url": response.url,
                    "status": response.status,
                    "headers": response.headers,
                    "body_bytes": len(response.body),
                },
                station_id=station.station_id,
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )
        if response.status == 304:
            return
        if response.status != 200:
            raise RuntimeError(f"NWS TGFTP returned HTTP {response.status}")

        validators: dict[str, str] = {}
        if etag := response.headers.get("ETag"):
            validators["If-None-Match"] = etag
        if modified := response.headers.get("Last-Modified"):
            validators["If-Modified-Since"] = modified
        if validators:
            self._validators[station.station_id] = validators

        text = response.body.decode("utf-8", errors="replace").strip()
        receipt_time, raw_metar, errors = parse_tgftp_station_text(text, station.station_id)
        if not raw_metar:
            raise ValueError(f"No METAR found in {station.station_id}.TXT")
        if self._seen_raw.get(station.station_id) == raw_metar:
            return
        self._seen_raw[station.station_id] = raw_metar

        observation_time = infer_metar_observation_time(raw_metar, receipt_time)
        normalized, parse_errors = parse_awc_metar(
            {
                "icaoId": station.station_id,
                "rawOb": raw_metar,
                "reportTime": observation_time.isoformat() if observation_time else None,
                "sourceReceiptTime": receipt_time.isoformat() if receipt_time else None,
            }
        )
        normalized["gateway_receipt_time"] = receipt_time.isoformat() if receipt_time else None
        normalized["gateway_url"] = url
        normalized["backfill"] = False
        normalized["is_latest_in_response"] = True
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="nws_tgftp_metar",
                event_type="weather_observation",
                payload=normalized,
                station_id=station.station_id,
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                provider_timestamp_ns=datetime_to_ns(receipt_time),
                observation_timestamp_ns=datetime_to_ns(observation_time),
                parse_errors=errors + parse_errors,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )


def parse_tgftp_station_text(
    text: str, expected_station_id: str
) -> tuple[datetime | None, str | None, list[str]]:
    errors: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, None, ["empty_gateway_file"]

    receipt_time: datetime | None = None
    match = _RECEIPT_LINE.match(lines[0])
    if match:
        try:
            receipt_time = datetime.strptime(
                f"{match.group('date')} {match.group('time')}", "%Y/%m/%d %H:%M"
            ).replace(tzinfo=UTC)
        except ValueError:
            errors.append("invalid_gateway_receipt_time")
    else:
        errors.append("missing_gateway_receipt_time")

    station_upper = expected_station_id.upper()
    raw = next(
        (line for line in lines[1:] if line.startswith(station_upper + " ")),
        None,
    )
    if raw is None and len(lines) >= 2:
        raw = lines[-1]
        errors.append("gateway_station_prefix_not_found")
    return receipt_time, raw, errors


def infer_metar_observation_time(raw_metar: str, reference: datetime | None) -> datetime | None:
    match = _METAR_TIME.search(raw_metar)
    if not match:
        return reference
    ref = reference or datetime.now(UTC)
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))

    candidates: list[datetime] = []
    for month_shift in (-1, 0, 1):
        year = ref.year
        month = ref.month + month_shift
        if month < 1:
            year -= 1
            month += 12
        elif month > 12:
            year += 1
            month -= 12
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=UTC))
        except ValueError:
            continue
    if not candidates:
        return reference

    plausible = [candidate for candidate in candidates if candidate <= ref + timedelta(minutes=10)]
    pool = plausible or candidates
    return min(pool, key=lambda candidate: abs((candidate - ref).total_seconds()))
