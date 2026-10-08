from __future__ import annotations

import asyncio
import time

import pytest

from weather_alpha.analysis.io import load_events
from weather_alpha.domain import EventEnvelope
from weather_alpha.storage.jsonl import JsonlEventStore
from weather_alpha.storage.multi import CriticalSinkError, MultiSink


class _Sink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events = []
        self.closed = False

    async def append(self, event) -> None:
        if self.fail:
            raise OSError("synthetic disk failure")
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


def _event() -> EventEnvelope:
    return EventEnvelope.build(run_id="run", source="test", event_type="test", payload={})


def test_required_sink_failure_is_fatal() -> None:
    required = _Sink(fail=True)
    mirror = _Sink()
    sink = MultiSink([required], [mirror])

    with pytest.raises(CriticalSinkError):
        asyncio.run(sink.append(_event()))
    assert mirror.events == []


def test_best_effort_sink_failure_does_not_hide_raw_success() -> None:
    required = _Sink()
    mirror = _Sink(fail=True)
    sink = MultiSink([required], [mirror])

    asyncio.run(sink.append(_event()))
    assert len(required.events) == 1


def test_jsonl_disk_write_does_not_block_asyncio_loop(monkeypatch, tmp_path) -> None:
    async def scenario() -> None:
        store = JsonlEventStore(tmp_path, "run", fsync_every=1)
        original = store._write_batch

        def slow_write(batch):
            time.sleep(0.15)
            return original(batch)

        monkeypatch.setattr(store, "_write_batch", slow_write)
        started = time.perf_counter()
        await store.append(_event())
        await asyncio.sleep(0.01)
        assert time.perf_counter() - started < 0.10
        await store.close()

    asyncio.run(scenario())


def test_jsonl_batches_events_without_losing_order(tmp_path) -> None:
    async def scenario() -> dict:
        store = JsonlEventStore(
            tmp_path,
            "run",
            fsync_every=10,
            write_batch_events=100,
            write_flush_seconds=0.05,
        )
        for index in range(100):
            event = EventEnvelope.build(
                run_id="run",
                source="test",
                event_type="sample",
                payload={"index": index},
            )
            await store.append(event)
        await store.close()
        return store.metrics()

    metrics = asyncio.run(scenario())
    events = load_events(tmp_path / "runs" / "run")
    assert [event["payload"]["index"] for event in events] == list(range(100))
    assert metrics["written_events"] == 100
    assert metrics["largest_batch"] > 1
