from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from weather_alpha.analysis.io import load_events
from weather_alpha.domain import EventEnvelope
from weather_alpha.storage.jsonl import JsonlEventStore


def _event(day: int) -> EventEnvelope:
    timestamp = int(datetime(2026, 7, day, 12, tzinfo=UTC).timestamp() * 1e9)
    return EventEnvelope.build(
        run_id="run",
        source="test",
        event_type="sample",
        payload={"day": day},
        received_wall_ns=timestamp,
        received_mono_ns=timestamp,
    )


def test_daily_rotation_compresses_previous_date_and_stays_readable(tmp_path) -> None:
    async def scenario() -> JsonlEventStore:
        store = JsonlEventStore(tmp_path, "run", fsync_every=1)
        await store.append(_event(20))
        await store.append(_event(21))
        await store.close()
        return store

    store = asyncio.run(scenario())
    run_dir = tmp_path / "runs" / "run"
    assert not (run_dir / "2026-07-20__test.jsonl").exists()
    assert (run_dir / "2026-07-20__test.jsonl.gz").exists()
    assert (run_dir / "2026-07-21__test.jsonl").exists()
    assert [event["payload"]["day"] for event in load_events(run_dir)] == [20, 21]
    assert len(store.rotated_archives()) == 1


def test_daily_compression_does_not_block_new_date_writes(monkeypatch, tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_compress_paths(self, paths) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(JsonlEventStore, "_compress_paths", fake_compress_paths)

    async def scenario() -> None:
        store = JsonlEventStore(tmp_path, "run", fsync_every=1)
        await store.append(_event(20))
        await asyncio.wait_for(store.append(_event(21)), timeout=0.2)
        await asyncio.wait_for(store.flush(), timeout=0.5)
        await asyncio.wait_for(started.wait(), timeout=0.2)
        assert (tmp_path / "runs" / "run" / "2026-07-21__test.jsonl").exists()
        release.set()
        await store.close()

    asyncio.run(scenario())


def test_daily_compression_failure_fails_required_store(monkeypatch, tmp_path) -> None:
    async def failing_compress_paths(self, paths) -> None:
        raise OSError("synthetic compression failure")

    monkeypatch.setattr(JsonlEventStore, "_compress_paths", failing_compress_paths)

    async def scenario() -> None:
        store = JsonlEventStore(tmp_path, "run", fsync_every=1)
        await store.append(_event(20))
        await store.append(_event(21))
        with pytest.raises(OSError, match="synthetic compression failure"):
            await store.close()

    asyncio.run(scenario())


def test_late_previous_date_event_after_midnight_is_preserved(tmp_path) -> None:
    async def scenario() -> dict:
        store = JsonlEventStore(
            tmp_path,
            "run",
            fsync_every=1,
            compress_rotated_files=False,
        )
        await store.append(_event(20))
        await store.append(_event(21))

        await store.append(_event(20))
        await store.append(_event(21))
        await store.close()
        return store.metrics()

    metrics = asyncio.run(scenario())
    events = load_events(tmp_path / "runs" / "run")
    assert [event["payload"]["day"] for event in events] == [20, 20, 21, 21]
    assert metrics["written_events"] == 4
    assert metrics["late_date_events"] == 1
