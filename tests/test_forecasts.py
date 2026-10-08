import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import orjson
import pytest

from weather_alpha.config import StationConfig, load_config
from weather_alpha.http_client import TimedResponse
from weather_alpha.sources.forecasts import (
    PROVIDERS,
    ForecastCollector,
    forecast_content_digest,
    forecast_revision_payload,
    summarize_forecast,
)


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


class _SequencedHttp:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = iter(statuses)
        self.calls = 0

    async def get(self, url, **_kwargs) -> TimedResponse:
        self.calls += 1
        status = next(self.statuses)
        payload = (
            {"timezone": "Europe/Amsterdam", "hourly": {"time": [], "temperature_2m": []}}
            if status == 200
            else {"error": True}
        )
        return TimedResponse(
            status=status,
            headers={"Retry-After": "0"},
            body=orjson.dumps(payload),
            request_started_wall_ns=100 + self.calls,
            request_started_mono_ns=200 + self.calls,
            response_received_wall_ns=300 + self.calls,
            response_received_mono_ns=400 + self.calls,
            url=url,
        )


class _PayloadHttp:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = iter(payloads)
        self.calls = 0

    async def get(self, url, **_kwargs) -> TimedResponse:
        self.calls += 1
        return TimedResponse(
            status=200,
            headers={},
            body=orjson.dumps(next(self.payloads)),
            request_started_wall_ns=100 + self.calls,
            request_started_mono_ns=200 + self.calls,
            response_received_wall_ns=300 + self.calls,
            response_received_mono_ns=400 + self.calls,
            url=url,
        )


def test_forecast_summary_local_day() -> None:
    station = StationConfig(
        key="amsterdam",
        city="Amsterdam",
        city_slug="amsterdam",
        station_id="EHAM",
        station_name="Schiphol",
        timezone="Europe/Amsterdam",
        latitude=52.3,
        longitude=4.7,
        market_unit="C",
        resolution_url="https://www.wunderground.com/history/daily/nl/schiphol/EHAM",
    )
    today = datetime.now(ZoneInfo("Europe/Amsterdam")).date().isoformat()
    payload = {
        "latitude": 52.3,
        "longitude": 4.7,
        "elevation": -3,
        "timezone": "Europe/Amsterdam",
        "hourly": {
            "time": [f"{today}T10:00", f"{today}T11:00", f"{today}T12:00"],
            "temperature_2m": [20.0, 22.5, 21.0],
        },
    }
    summary = summarize_forecast(payload, station)
    assert summary["today_hourly_max_c"] == 22.5
    assert summary["today_hourly_max_time"].endswith("11:00:00+02:00") or summary[
        "today_hourly_max_time"
    ].endswith("11:00:00+01:00")


def test_forecast_retries_transient_responses_before_recording_error() -> None:
    cfg = load_config(project_root=".")
    cfg = cfg.model_copy(
        update={
            "collector": cfg.collector.model_copy(
                update={
                    "forecast_retry_attempts": 3,
                    "forecast_retry_base_seconds": 0.001,
                    "forecast_retry_max_seconds": 0.001,
                }
            )
        }
    )
    station = cfg.stations["amsterdam"]
    sink = _Sink()
    http = _SequencedHttp([429, 503, 200])
    collector = ForecastCollector(cfg, "run", sink, http)

    asyncio.run(collector.poll_provider(station, "best_match", PROVIDERS["best_match"]))

    assert http.calls == 3
    assert [event.event_type for event in sink.events].count("http_response") == 3
    assert [event.event_type for event in sink.events].count("retry_scheduled") == 2
    assert any(event.event_type == "forecast_snapshot" for event in sink.events)
    assert not any(event.event_type == "collector_error" for event in sink.events)


def test_forecast_records_error_only_after_retry_budget_is_exhausted() -> None:
    cfg = load_config(project_root=".")
    cfg = cfg.model_copy(
        update={
            "collector": cfg.collector.model_copy(
                update={
                    "forecast_retry_attempts": 2,
                    "forecast_retry_base_seconds": 0.001,
                    "forecast_retry_max_seconds": 0.001,
                }
            )
        }
    )
    station = cfg.stations["amsterdam"]
    sink = _Sink()
    http = _SequencedHttp([503, 503])
    collector = ForecastCollector(cfg, "run", sink, http)

    asyncio.run(collector.poll_provider(station, "best_match", PROVIDERS["best_match"]))

    errors = [event for event in sink.events if event.event_type == "collector_error"]
    assert http.calls == 2
    assert len(errors) == 1
    assert errors[0].payload["attempts"] == 2


def test_collector_emits_revision_only_after_changed_snapshot() -> None:
    cfg = load_config(project_root=".")
    station = cfg.stations["amsterdam"]
    today = datetime.now(ZoneInfo(station.timezone)).date().isoformat()
    base_payload = {
        "timezone": station.timezone,
        "hourly": {
            "time": [f"{today}T10:00", f"{today}T11:00"],
            "temperature_2m": [20.0, 22.0],
        },
    }
    sink = _Sink()
    http = _PayloadHttp(
        [
            base_payload,
            {
                **base_payload,
                "hourly": {**base_payload["hourly"], "temperature_2m": [20.0, 23.0]},
            },
        ]
    )
    collector = ForecastCollector(cfg, "run", sink, http)

    asyncio.run(collector.poll_provider(station, "dwd_icon", PROVIDERS["dwd_icon"]))
    asyncio.run(collector.poll_provider(station, "dwd_icon", PROVIDERS["dwd_icon"]))

    assert [event.event_type for event in sink.events].count("forecast_snapshot") == 2
    revisions = [event for event in sink.events if event.event_type == "forecast_revision"]
    assert len(revisions) == 1
    assert revisions[0].payload["previous_bucket_value"] == 22
    assert revisions[0].payload["current_bucket_value"] == 23
    assert revisions[0].payload["bucket_changed"] is True


def test_forecast_digest_ignores_provider_generation_runtime() -> None:
    first = {
        "generationtime_ms": 2.5,
        "hourly": {"time": ["2026-07-20T12:00"], "temperature_2m": [20.0]},
    }
    second = {**first, "generationtime_ms": 15.75}

    assert forecast_content_digest(first) == forecast_content_digest(second)


def test_forecast_revision_detects_market_bucket_change() -> None:
    cfg = load_config(project_root=".")
    station = cfg.stations["nyc"]
    previous = {
        "local_date": "2026-07-22",
        "today_hourly_max_c": 26.4,
        "remaining_hourly_max_c": 26.4,
        "today_hourly_max_time": "2026-07-22T14:00:00-04:00",
    }
    current = {
        "local_date": "2026-07-22",
        "today_hourly_max_c": 27.0,
        "remaining_hourly_max_c": 27.0,
        "today_hourly_max_time": "2026-07-22T15:00:00-04:00",
    }

    revision = forecast_revision_payload(
        previous,
        current,
        station,
        provider="gfs_hrrr",
        previous_digest="old",
        current_digest="new",
    )

    assert revision is not None
    assert revision["previous_bucket_value"] == 80
    assert revision["current_bucket_value"] == 81
    assert revision["bucket_changed"] is True
    assert revision["today_max_delta_c"] == pytest.approx(0.6)


def test_forecast_revision_ignores_same_max_and_day_rollover() -> None:
    station = load_config(project_root=".").stations["amsterdam"]
    previous = {
        "local_date": "2026-07-22",
        "today_hourly_max_c": 22.0,
        "remaining_hourly_max_c": 21.0,
    }

    assert (
        forecast_revision_payload(
            previous,
            dict(previous),
            station,
            provider="dwd_icon",
            previous_digest="old",
            current_digest="new",
        )
        is None
    )
    assert (
        forecast_revision_payload(
            previous,
            {**previous, "local_date": "2026-07-23", "today_hourly_max_c": 23.0},
            station,
            provider="dwd_icon",
            previous_digest="old",
            current_digest="new",
        )
        is None
    )
