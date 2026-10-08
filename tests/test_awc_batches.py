from __future__ import annotations

import asyncio

import orjson

from weather_alpha.analysis.io import weather_observation_key
from weather_alpha.config import load_config
from weather_alpha.http_client import TimedResponse
from weather_alpha.sources.awc import AviationWeatherCollector


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


class _Http:
    def __init__(self, payload) -> None:
        self.payload = payload

    async def get(self, *_args, **_kwargs) -> TimedResponse:
        return TimedResponse(
            status=200,
            headers={},
            body=orjson.dumps(self.payload),
            request_started_wall_ns=1_784_500_000_000_000_000,
            request_started_mono_ns=100,
            response_received_wall_ns=1_784_500_001_000_000_000,
            response_received_mono_ns=1_000_000_100,
            url="https://example.test/metar",
        )


def test_awc_batch_marks_backfill_and_dedupes_across_polls() -> None:
    payload = [
        {
            "icaoId": "EHAM",
            "reportTime": "2026-07-20T00:25:00Z",
            "temp": 16,
            "rawOb": "METAR EHAM 200025Z 33006KT 9999 16/09 Q1022",
        },
        {
            "icaoId": "EHAM",
            "reportTime": "2026-07-19T23:55:00Z",
            "temp": 15,
            "rawOb": "METAR EHAM 192355Z 33006KT 9999 15/09 Q1022",
        },
    ]
    cfg = load_config(project_root=".")
    sink = _Sink()
    collector = AviationWeatherCollector(cfg, "run", sink, _Http(payload))

    asyncio.run(collector.poll_once())
    asyncio.run(collector.poll_once())

    observations = [event for event in sink.events if event.event_type == "weather_observation"]
    assert len(observations) == 2
    assert observations[0].payload["backfill"] is True
    assert observations[1].payload["backfill"] is False
    assert all(event.received_wall_ns == 1_784_500_001_000_000_000 for event in observations)


def test_weather_key_normalizes_metar_prefix_and_maintenance_suffix() -> None:
    base = {
        "station_id": "KLGA",
        "observation_timestamp_ns": 123,
        "source": "awc_metar",
        "payload": {"raw_observation": "METAR KLGA 201200Z 31010KT 10SM CLR 25/10 $"},
    }
    alternate = {
        **base,
        "source": "nws_tgftp_metar",
        "payload": {"raw_observation": "KLGA   201200Z 31010KT 10SM CLR 25/10"},
    }
    assert weather_observation_key(base) == weather_observation_key(alternate)
