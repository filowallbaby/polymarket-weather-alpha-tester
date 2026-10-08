from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import orjson

from ..domain import EventEnvelope


class ClickHouseEventSink:


    columns = [
        "run_id",
        "schema_version",
        "source",
        "event_type",
        "received_at",
        "received_wall_ns",
        "received_mono_ns",
        "request_started_wall_ns",
        "response_received_wall_ns",
        "provider_timestamp_ns",
        "observation_timestamp_ns",
        "station_id",
        "market_slug",
        "asset_id",
        "payload_sha256",
        "payload_json",
        "parse_errors",
    ]

    def __init__(self, flush_rows: int = 1000, flush_seconds: float = 2.0) -> None:
        try:
            import clickhouse_connect
        except ImportError as exc:
            raise RuntimeError(
                "Install the clickhouse extra: pip install -e '.[clickhouse]'"
            ) from exc

        url = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
        from urllib.parse import urlparse

        parsed = urlparse(url)
        self._client = clickhouse_connect.get_client(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8123,
            username=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            database=os.getenv("CLICKHOUSE_DATABASE", "weather_alpha"),
        )
        self._flush_rows = flush_rows
        self._flush_seconds = flush_seconds
        self._rows: list[list[Any]] = []
        self._lock = asyncio.Lock()
        self._closed = False
        self._flush_task = asyncio.create_task(self._flush_loop())

    @staticmethod
    def _row(event: EventEnvelope) -> list[Any]:
        received = datetime.fromtimestamp(event.received_wall_ns / 1e9, tz=UTC)
        return [
            event.run_id,
            event.schema_version,
            event.source,
            event.event_type,
            received,
            event.received_wall_ns,
            event.received_mono_ns,
            event.request_started_wall_ns,
            event.response_received_wall_ns,
            event.provider_timestamp_ns,
            event.observation_timestamp_ns,
            event.station_id,
            event.market_slug,
            event.asset_id,
            event.payload_sha256,
            orjson.dumps(event.payload).decode("utf-8"),
            event.parse_errors,
        ]

    async def append(self, event: EventEnvelope) -> None:
        should_flush = False
        async with self._lock:
            self._rows.append(self._row(event))
            should_flush = len(self._rows) >= self._flush_rows
        if should_flush:
            await self.flush()

    async def _flush_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._flush_seconds)
            await self.flush()

    async def flush(self) -> None:
        async with self._lock:
            if not self._rows:
                return
            rows, self._rows = self._rows, []
        try:
            await asyncio.to_thread(
                self._client.insert,
                "events",
                rows,
                column_names=self.columns,
            )
        except Exception:
            async with self._lock:
                self._rows = rows + self._rows
            raise

    async def close(self) -> None:
        self._closed = True
        self._flush_task.cancel()
        try:
            await self._flush_task
        except asyncio.CancelledError:
            pass
        await self.flush()
        await asyncio.to_thread(self._client.close)
