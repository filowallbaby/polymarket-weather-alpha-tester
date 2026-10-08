from __future__ import annotations

import asyncio

from weather_alpha.config import MarketConfig
from weather_alpha.domain import EventEnvelope
from weather_alpha.storage.market_capture import MarketCaptureSink


class _Sink:
    def __init__(self) -> None:
        self.events = []
        self.closed = False

    async def append(self, event) -> None:
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


def _event(source: str, event_type: str, wall_ns: int, *, asset_id: str | None = None):
    payload = {"value": 1}
    if source == "polymarket_book":
        payload = {
            "asset_id": asset_id,
            "bids": [{"price": str(index / 100), "size": "1"} for index in range(25)],
            "asks": [{"price": str(index / 100), "size": "1"} for index in range(25)],
        }
    return EventEnvelope.build(
        run_id="run",
        source=source,
        event_type=event_type,
        payload=payload,
        asset_id=asset_id,
        received_wall_ns=wall_ns,
        received_mono_ns=wall_ns,
    )


def test_compact_mode_drops_raw_throttles_books_and_truncates_depth() -> None:
    async def scenario():
        base = 1_000_000_000_000
        target = _Sink()
        cfg = MarketConfig(
            compact_interval_seconds=1,
            compact_depth_levels=5,
            burst_before_seconds=2,
        )
        sink = MarketCaptureSink(target, cfg)
        await sink.append(_event("polymarket_ws_raw", "price_change", base, asset_id="a"))
        await sink.append(_event("polymarket_book", "book_state", base, asset_id="a"))
        await sink.append(
            _event("polymarket_book", "book_state", base + 500_000_000, asset_id="a")
        )
        await sink.append(_event("system", "heartbeat", base + 4_000_000_000))
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    books = [event for event in target.events if event.source == "polymarket_book"]
    assert not [event for event in target.events if event.source == "polymarket_ws_raw"]
    assert len(books) == 1
    assert len(books[0].payload["bids"]) == 5
    assert books[0].payload["_capture"]["mode"] == "compact"
    assert metrics["raw_events_dropped"] == 1
    assert metrics["raw_events_dropped_before_buffer"] == 1
    assert metrics["max_buffer_events"] == 2
    assert metrics["book_events_throttled"] == 1
    assert target.closed is True


def test_weather_observation_no_longer_triggers_full_market_burst() -> None:
    async def scenario():
        base = 2_000_000_000_000
        target = _Sink()
        cfg = MarketConfig(burst_before_seconds=30, burst_after_seconds=120)
        sink = MarketCaptureSink(target, cfg)
        await sink.append(_event("polymarket_ws_raw", "price_change", base, asset_id="a"))
        await sink.append(_event("polymarket_book", "book_state", base, asset_id="a"))
        await sink.append(
            _event("awc_metar", "weather_observation", base + 10_000_000_000)
        )
        await sink.append(
            _event(
                "polymarket_ws_raw",
                "price_change",
                base + 11_000_000_000,
                asset_id="a",
            )
        )
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    market = [
        event for event in target.events if event.source in {"polymarket_ws_raw", "polymarket_book"}
    ]
    assert [event.source for event in market] == ["polymarket_book"]
    assert market[0].payload["_capture"]["mode"] == "compact"
    assert metrics.get("weather_burst_triggers", 0) == 0


def test_backfill_does_not_trigger_burst() -> None:
    async def scenario():
        base = 3_000_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(target, MarketConfig(burst_before_seconds=30))
        await sink.append(_event("polymarket_ws_raw", "price_change", base, asset_id="a"))
        backfill = _event("awc_metar", "weather_observation", base + 1_000_000_000)
        backfill.payload["backfill"] = True
        await sink.append(backfill)
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    assert not [event for event in target.events if event.source == "polymarket_ws_raw"]
    assert metrics.get("burst_triggers", 0) == 0


def test_duplicate_equivalent_forecast_delivery_does_not_extend_burst() -> None:
    async def scenario():
        base = 4_000_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(
            target,
            MarketConfig(burst_before_seconds=30, burst_after_seconds=30),
        )
        for label in ("82-83°F", "84-85°F"):
            token = _event("polymarket_gamma", "tracked_token", base)
            token.station_id = "KLGA"
            token.payload = {"event_date": "2026-07-22", "bucket_label": label}
            await sink.append(token)
        first = _event("forecast_best_match", "forecast_revision", base)
        first.station_id = "KLGA"
        first.payload = {
            "local_date": "2026-07-22",
            "market_unit": "F",
            "previous_market_max": 82.4,
            "current_market_max": 84.1,
            "bucket_changed": True,
        }
        duplicate = _event("forecast_gfs_hrrr", "forecast_revision", base + 2_000_000_000)
        duplicate.station_id = "KLGA"
        duplicate.payload = dict(first.payload)
        await sink.append(first)
        first_burst_end = sink.metrics()["burst_active_until_ns"]
        await sink.append(duplicate)
        second_burst_end = sink.metrics()["burst_active_until_ns"]
        await sink.close()
        return sink.metrics(), first_burst_end, second_burst_end

    metrics, first_burst_end, second_burst_end = asyncio.run(scenario())
    assert metrics["burst_triggers"] == 1
    assert metrics["duplicate_forecast_revision_triggers"] == 1
    assert second_burst_end == first_burst_end


def test_forecast_bucket_revision_triggers_market_burst() -> None:
    async def scenario():
        base = 5_000_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(
            target,
            MarketConfig(
                burst_before_seconds=30,
                burst_after_seconds=30,
                capture_raw_ws_in_bursts=True,
            ),
        )
        await sink.append(_event("polymarket_ws_raw", "price_change", base, asset_id="a"))
        revision = _event(
            "forecast_dwd_icon", "forecast_revision", base + 10_000_000_000
        )
        revision.station_id = "EGLC"
        revision.payload = {
            "bucket_changed": True,
            "content_sha256": "new-forecast",
            "previous_bucket_value": 24,
            "current_bucket_value": 22,
        }
        await sink.append(revision)
        await sink.append(
            _event(
                "polymarket_ws_raw",
                "price_change",
                base + 11_000_000_000,
                asset_id="a",
            )
        )
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    market = [event for event in target.events if event.source == "polymarket_ws_raw"]
    assert [event.payload["_capture"]["mode"] for event in market] == [
        "burst_pre",
        "burst_post",
    ]
    assert metrics["forecast_revision_burst_triggers"] == 1


def test_forecast_burst_keeps_normalized_books_but_drops_raw_by_default() -> None:
    async def scenario():
        base = 5_500_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(
            target,
            MarketConfig(burst_before_seconds=30, burst_after_seconds=30),
        )
        await sink.append(_event("polymarket_ws_raw", "price_change", base, asset_id="a"))
        await sink.append(_event("polymarket_book", "book_state", base, asset_id="a"))
        revision = _event(
            "forecast_dwd_icon", "forecast_revision", base + 10_000_000_000
        )
        revision.station_id = "EGLC"
        revision.payload = {
            "bucket_changed": True,
            "previous_bucket_value": 24,
            "current_bucket_value": 22,
        }
        await sink.append(revision)
        await sink.append(
            _event(
                "polymarket_ws_raw",
                "price_change",
                base + 11_000_000_000,
                asset_id="a",
            )
        )
        await sink.append(
            _event(
                "polymarket_book",
                "book_state",
                base + 11_000_000_000,
                asset_id="a",
            )
        )
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    assert not [event for event in target.events if event.source == "polymarket_ws_raw"]
    books = [event for event in target.events if event.source == "polymarket_book"]
    assert [event.payload["_capture"]["mode"] for event in books] == [
        "burst_pre",
        "burst_post",
    ]
    assert metrics["raw_events_dropped"] == 1
    assert metrics["raw_burst_events_dropped"] == 1
    assert metrics["raw_events_dropped_before_buffer"] == 2
    assert metrics["capture_raw_ws_in_bursts"] is False


def test_forecast_degree_change_inside_same_market_bucket_does_not_trigger_burst() -> None:
    async def scenario():
        base = 6_000_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(target, MarketConfig(burst_before_seconds=30))
        token = _event("polymarket_gamma", "tracked_token", base)
        token.station_id = "KLGA"
        token.payload = {
            "event_date": "2026-07-22",
            "bucket_label": "88-89°F",
        }
        await sink.append(token)
        await sink.append(_event("polymarket_ws_raw", "price_change", base + 1, asset_id="a"))
        revision = _event("forecast_gfs_hrrr", "forecast_revision", base + 2)
        revision.station_id = "KLGA"
        revision.payload = {
            "local_date": "2026-07-22",
            "market_unit": "F",
            "previous_market_max": 88.1,
            "current_market_max": 89.1,
            "bucket_changed": True,
        }
        await sink.append(revision)
        await sink.close()
        return target, sink.metrics()

    target, metrics = asyncio.run(scenario())
    revisions = [event for event in target.events if event.event_type == "forecast_revision"]
    assert revisions[0].payload["economic_bucket_changed"] is False
    assert not [event for event in target.events if event.source == "polymarket_ws_raw"]
    assert metrics.get("forecast_revision_burst_triggers", 0) == 0


def test_market_pre_window_is_bounded_by_serialized_bytes() -> None:
    async def scenario():
        base = 7_000_000_000_000
        target = _Sink()
        sink = MarketCaptureSink(
            target,
            MarketConfig(
                burst_before_seconds=30,
                burst_max_buffer_events=100,
                burst_max_buffer_bytes=1_000_000,
            ),
        )
        first = _event("polymarket_book", "book_state", base, asset_id="a")
        first.payload["padding"] = "x" * 600_000
        second = _event(
            "polymarket_book", "book_state", base + 1_000_000_000, asset_id="b"
        )
        second.payload["padding"] = "y" * 600_000
        await sink.append(first)
        await sink.append(second)
        before_close = sink.metrics()
        await sink.close()
        return before_close, sink.metrics()

    before_close, after_close = asyncio.run(scenario())
    assert before_close["buffer_forced_drains"] == 1
    assert before_close["buffered_events"] == 1
    assert before_close["buffered_bytes"] <= 1_000_000
    assert after_close["buffered_events"] == 0
    assert after_close["buffered_bytes"] == 0
