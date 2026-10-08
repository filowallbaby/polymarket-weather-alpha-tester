from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import orjson

from ..domain import EventEnvelope
from .archive import compress_event_file


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    path: Path
    event_date: str
    encoded: bytes


_STOP = object()


class JsonlEventStore:


    def __init__(
        self,
        base_dir: Path,
        run_id: str,
        fsync_every: int = 200,
        *,
        compress_rotated_files: bool = True,
        compression_level: int = 6,
        queue_max_events: int = 20_000,
        write_batch_events: int = 500,
        write_flush_seconds: float = 0.025,
    ) -> None:
        self.run_id = run_id
        self.run_dir = base_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fsync_every = max(1, fsync_every)
        self._compress_rotated_files = compress_rotated_files
        self._compression_level = compression_level
        self._queue_max_events = max(1, queue_max_events)
        self._write_batch_events = max(1, write_batch_events)
        self._write_flush_seconds = max(0.001, write_flush_seconds)
        self._handles: dict[Path, IO[bytes]] = {}
        self._counts: defaultdict[Path, int] = defaultdict(int)
        self._latest_date: str | None = None
        self._compression_tasks: list[asyncio.Task[None]] = []
        self._rotated_archives: list[dict] = []
        self._queue: asyncio.Queue[_PendingWrite | object] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._worker_failure: BaseException | None = None
        self._closed = False
        self._queue_high_watermark = 0
        self._written_events = 0
        self._written_batches = 0
        self._largest_batch = 0
        self._last_batch_write_ms: float | None = None
        self._max_batch_write_ms = 0.0
        self._late_date_events = 0

    def _path_for(self, event: EventEnvelope) -> Path:
        date = datetime.fromtimestamp(event.received_wall_ns / 1e9, tz=UTC).strftime("%Y-%m-%d")
        safe_source = event.source.replace("/", "_")
        return self.run_dir / f"{date}__{safe_source}.jsonl"

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        self._queue = asyncio.Queue(maxsize=self._queue_max_events)
        self._worker = asyncio.create_task(
            self._writer_loop(), name=f"jsonl_writer_{self.run_id}"
        )

    async def append(self, event: EventEnvelope) -> None:
        if self._closed:
            raise RuntimeError("JSONL event store is already closed")
        self._raise_background_failure()
        self._ensure_worker()
        assert self._queue is not None
        encoded = orjson.dumps(event.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE)
        pending = _PendingWrite(
            path=self._path_for(event),
            event_date=datetime.fromtimestamp(event.received_wall_ns / 1e9, tz=UTC).strftime(
                "%Y-%m-%d"
            ),
            encoded=encoded,
        )
        await self._queue.put(pending)
        self._queue_high_watermark = max(self._queue_high_watermark, self._queue.qsize())
        self._raise_background_failure()

    async def flush(self) -> None:

        if self._queue is not None:
            await self._queue.join()
        self._raise_background_failure()

    async def write_manifest(self, manifest: dict) -> None:
        await self.flush()
        path = self.run_dir / "manifest.json"
        payload = orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        await asyncio.to_thread(self._write_manifest_atomic, path, payload)

    @staticmethod
    def _write_manifest_atomic(path: Path, payload: bytes) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def event_counts(self) -> dict[str, int]:
        return {path.name: count for path, count in sorted(self._counts.items())}

    def rotated_archives(self) -> list[dict]:
        return sorted(self._rotated_archives, key=lambda item: item["source"])

    def metrics(self) -> dict[str, int | float | None]:
        return {
            "queue_depth": self._queue.qsize() if self._queue is not None else 0,
            "queue_capacity": self._queue_max_events,
            "queue_high_watermark": self._queue_high_watermark,
            "written_events": self._written_events,
            "written_batches": self._written_batches,
            "largest_batch": self._largest_batch,
            "last_batch_write_ms": self._last_batch_write_ms,
            "max_batch_write_ms": self._max_batch_write_ms,
            "late_date_events": self._late_date_events,
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        try:
            await self.flush()
            if self._queue is not None and self._worker is not None:
                await self._queue.put(_STOP)
                await self._worker
        except BaseException as exc:
            failure = exc
        finally:
            await asyncio.to_thread(self._close_paths, list(self._handles))
            if self._compression_tasks:
                results = await asyncio.gather(*self._compression_tasks, return_exceptions=True)
                if failure is None:
                    failure = next(
                        (result for result in results if isinstance(result, BaseException)), None
                    )
        if failure is not None:
            raise failure

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        queue = self._queue
        stop_after_batch = False
        try:
            while not stop_after_batch:
                item = await queue.get()
                if item is _STOP:
                    queue.task_done()
                    break
                batch = [item]
                deadline = asyncio.get_running_loop().time() + self._write_flush_seconds
                while len(batch) < self._write_batch_events:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                    if next_item is _STOP:
                        queue.task_done()
                        stop_after_batch = True
                        break
                    batch.append(next_item)
                started = time.perf_counter_ns()
                try:
                    rotated = await asyncio.to_thread(self._write_batch, batch)
                finally:
                    for _ in batch:
                        queue.task_done()
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                self._written_events += len(batch)
                self._written_batches += 1
                self._largest_batch = max(self._largest_batch, len(batch))
                self._last_batch_write_ms = elapsed_ms
                self._max_batch_write_ms = max(self._max_batch_write_ms, elapsed_ms)
                if self._compress_rotated_files and rotated:
                    self._compression_tasks.append(
                        asyncio.create_task(
                            self._compress_paths(rotated),
                            name=f"compress_{self.run_id}_{self._latest_date or 'initial'}",
                        )
                    )
                self._raise_compression_failure()
        except BaseException as exc:
            self._worker_failure = exc
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()
            raise

    def _write_batch(self, batch: list[_PendingWrite]) -> list[Path]:
        rotated: list[Path] = []
        grouped: dict[Path, bytearray] = {}
        added_counts: defaultdict[Path, int] = defaultdict(int)
        for item in batch:


            if self._latest_date is not None and item.event_date < self._latest_date:
                self._late_date_events += 1
            if self._latest_date is None or item.event_date > self._latest_date:
                self._write_grouped(grouped, added_counts)
                grouped.clear()
                added_counts.clear()
                rotated.extend(self._close_before(item.event_date))
                self._latest_date = item.event_date
            grouped.setdefault(item.path, bytearray()).extend(item.encoded)
            added_counts[item.path] += 1
        self._write_grouped(grouped, added_counts)
        return rotated

    def _write_grouped(
        self,
        grouped: dict[Path, bytearray],
        added_counts: dict[Path, int],
    ) -> None:
        for path, payload in grouped.items():
            handle = self._handles.get(path)
            if handle is None:
                handle = path.open("ab", buffering=0)
                self._handles[path] = handle
            previous = self._counts[path]
            handle.write(payload)
            self._counts[path] += added_counts[path]
            if self._counts[path] // self._fsync_every > previous // self._fsync_every:
                os.fsync(handle.fileno())

    def _close_before(self, event_date: str) -> list[Path]:
        paths = [path for path in self._handles if path.name[:10] < event_date]
        self._close_paths(paths)
        return paths

    def _close_paths(self, paths: list[Path]) -> None:
        for path in paths:
            handle = self._handles.pop(path, None)
            if handle is None:
                continue
            try:
                os.fsync(handle.fileno())
            finally:
                handle.close()

    async def _compress_paths(self, paths: list[Path]) -> None:
        for path in paths:
            result = await asyncio.to_thread(
                compress_event_file,
                path,
                compression_level=self._compression_level,
            )
            self._rotated_archives.append(result)

    def _raise_compression_failure(self) -> None:
        for task in self._compression_tasks:
            if task.done() and not task.cancelled():
                task.result()

    def _raise_background_failure(self) -> None:
        if self._worker_failure is not None:
            raise self._worker_failure
        if self._worker is not None and self._worker.done() and not self._worker.cancelled():
            self._worker.result()
        self._raise_compression_failure()
