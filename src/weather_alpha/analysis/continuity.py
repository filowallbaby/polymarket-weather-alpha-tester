from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ContinuityGap:
    segment_before: int
    last_heartbeat_ns: int
    resumed_heartbeat_ns: int
    gap_seconds: float
    warmup_until_ns: int

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "segment_before": self.segment_before,
            "last_heartbeat_ns": self.last_heartbeat_ns,
            "resumed_heartbeat_ns": self.resumed_heartbeat_ns,
            "gap_seconds": self.gap_seconds,
            "warmup_until_ns": self.warmup_until_ns,
            "last_heartbeat_utc": _iso_utc(self.last_heartbeat_ns),
            "resumed_heartbeat_utc": _iso_utc(self.resumed_heartbeat_ns),
            "warmup_until_utc": _iso_utc(self.warmup_until_ns),
        }


def _iso_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC).isoformat()


def detect_continuity_gaps(
    events: list[dict[str, Any]],
    *,
    max_gap_seconds: float,
    resume_warmup_seconds: float,
) -> list[ContinuityGap]:

    heartbeat_ns = sorted(
        {
            int(event.get("received_wall_ns") or 0)
            for event in events
            if event.get("source") == "system"
            and event.get("event_type") == "heartbeat"
            and int(event.get("received_wall_ns") or 0) > 0
        }
    )
    gaps: list[ContinuityGap] = []
    for previous, current in zip(heartbeat_ns, heartbeat_ns[1:]):
        gap_seconds = (current - previous) / 1e9
        if gap_seconds <= max_gap_seconds:
            continue
        gaps.append(
            ContinuityGap(
                segment_before=len(gaps),
                last_heartbeat_ns=previous,
                resumed_heartbeat_ns=current,
                gap_seconds=gap_seconds,
                warmup_until_ns=current + int(resume_warmup_seconds * 1e9),
            )
        )
    return gaps


def continuity_at(timestamp_ns: int, gaps: list[ContinuityGap]) -> tuple[int, bool]:

    segment = 0
    for gap in gaps:
        if timestamp_ns <= gap.last_heartbeat_ns:
            break
        if timestamp_ns < gap.warmup_until_ns:
            return segment + 1, False
        segment += 1
    return segment, True
