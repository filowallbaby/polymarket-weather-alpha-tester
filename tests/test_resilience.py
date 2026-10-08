from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from email.utils import format_datetime

from weather_alpha.config import load_config
from weather_alpha.doctor import Check, _clock_offset_check, run_doctor
from weather_alpha.http_client import TimedResponse
from weather_alpha.sources.polymarket import PolymarketCollector


class _FailingHttp:
    async def get(self, *args, **kwargs):
        raise OSError("synthetic network failure")

    async def close(self) -> None:
        return None


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None


def _dated_response(local_offset_seconds: float) -> TimedResponse:
    server_time = datetime(2026, 7, 21, 0, 0, tzinfo=UTC)
    server_ns = int(server_time.timestamp() * 1e9)
    local_ns = server_ns + int(local_offset_seconds * 1e9)
    return TimedResponse(
        status=200,
        headers={"Date": format_datetime(server_time, usegmt=True)},
        body=b"",
        request_started_wall_ns=local_ns - 50_000_000,
        request_started_mono_ns=1,
        response_received_wall_ns=local_ns + 50_000_000,
        response_received_mono_ns=100_000_001,
        url="https://example.test",
    )


def test_clock_check_uses_multiple_endpoints_and_ignores_one_stale_date() -> None:
    check = _clock_offset_check(
        [
            ("one", _dated_response(0.04)),
            ("two", _dated_response(0.06)),
            ("stale", _dated_response(-5.0)),
        ]
    )

    assert check.ok is True
    assert "median=+0.040s" in check.detail


def test_doctor_reports_network_failures_without_raising(monkeypatch, tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    cfg.data_dir.mkdir(parents=True)
    cfg.reports_dir.mkdir(parents=True)

    monkeypatch.setattr("weather_alpha.doctor.TimedHttpClient", lambda *_: _FailingHttp())

    async def fake_playwright_check() -> Check:
        return Check("playwright", False, "synthetic")

    monkeypatch.setattr("weather_alpha.doctor._playwright_check", fake_playwright_check)
    async def fake_wis2_check() -> Check:
        return Check("wis2_mqtt", False, "synthetic")

    monkeypatch.setattr("weather_alpha.doctor._wis2_check", fake_wis2_check)
    monkeypatch.setattr(
        "weather_alpha.doctor._clock_service_check",
        lambda: Check("clock_sync_service", False, "synthetic"),
    )

    checks = asyncio.run(run_doctor(cfg))
    by_name = {check.name: check for check in checks}
    assert by_name["aviation_weather"].ok is False
    assert by_name["nws_tgftp"].ok is False
    assert by_name["polymarket_gamma"].ok is False
    assert by_name["forecast_provider"].ok is False
    assert "synthetic network failure" in by_name["aviation_weather"].detail


def test_polymarket_discovery_failure_is_recorded_and_retried(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    cfg = cfg.model_copy(
        update={"collector": cfg.collector.model_copy(update={"market_discovery_seconds": 0.01})}
    )
    sink = _Sink()
    collector = PolymarketCollector(cfg, "run", sink, _FailingHttp())
    calls = 0

    async def flaky_discover():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary gamma failure")
        return {}

    collector.discover = flaky_discover

    async def scenario() -> None:
        task = asyncio.create_task(collector.run())
        await asyncio.sleep(0.04)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert calls >= 2
    assert any(
        event.source == "polymarket_gamma" and event.event_type == "collector_error"
        for event in sink.events
    )


def test_polymarket_tracks_execution_metadata_and_skips_closed_book(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    sink = _Sink()
    collector = PolymarketCollector(cfg, "run", sink, _FailingHttp())
    event = {
        "id": "event",
        "slug": "highest-temperature-in-nyc-on-july-20-2026",
        "description": (
            "This market will resolve to the temperature range that contains the highest "
            "temperature recorded at the LaGuardia Airport Station in degrees Fahrenheit "
            "on 20 Jul '26. Resolution source: "
            "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA."
        ),
        "markets": [
            {
                "id": "open",
                "question": "Will the highest temperature in NYC be 84°F on July 20?",
                "groupItemTitle": "84°F",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["yes-token", "no-token"]',
                "orderPriceMinTickSize": 0.001,
                "feesEnabled": True,
                "feeSchedule": {"rate": 0.05, "exponent": 1, "rebateRate": 0.25},
            },
            {
                "id": "not-accepting",
                "question": "Will the highest temperature in NYC be 85°F on July 20?",
                "groupItemTitle": "85°F",
                "active": True,
                "closed": False,
                "acceptingOrders": False,
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["closed-yes", "closed-no"]',
            },
        ],
    }

    tokens = asyncio.run(collector._parse_event(cfg.stations["nyc"], event))
    assert len(tokens) == 1
    token = tokens[0]
    assert token.asset_id == "yes-token"
    assert token.paired_no_asset_id == "no-token"
    assert token.outcome == "Yes"
    assert token.tick_size == 0.001
    assert token.fees_enabled is True
    assert token.fee_rate == 0.05
    assert token.fee_exponent == 1
