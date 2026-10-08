from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class TimestampPair:
    wall_ns: int
    mono_ns: int

    @property
    def utc_iso(self) -> str:
        return datetime.fromtimestamp(self.wall_ns / 1_000_000_000, tz=UTC).isoformat()


def now_pair() -> TimestampPair:
    return TimestampPair(wall_ns=time.time_ns(), mono_ns=time.monotonic_ns())


def ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).isoformat()


def datetime_to_ns(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000_000)


async def sleep_aligned(period_seconds: float, phase_seconds: float = 0.15) -> None:

    now = time.time()
    target = ((now // period_seconds) + 1) * period_seconds + phase_seconds
    await asyncio.sleep(max(0.0, target - now))
