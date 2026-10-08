from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Iterable

import orjson

from ..storage.archive import event_files, open_event_file


def resolve_run_dir(data_dir: Path, run_id: str | None) -> Path:
    runs_dir = data_dir / "runs"
    if run_id:
        path = runs_dir / run_id
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_dir}")
    return candidates[-1]


def iter_events(run_dir: Path, sources: set[str] | None = None) -> Iterator[dict[str, Any]]:

    for path in event_files(run_dir):
        if sources and not any(f"__{source}.jsonl" in path.name for source in sources):
            continue
        with open_event_file(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from exc
                if sources is None or event.get("source") in sources:
                    yield event


def load_events(run_dir: Path, sources: set[str] | None = None) -> list[dict[str, Any]]:
    events = list(iter_events(run_dir, sources))
    events.sort(key=lambda item: item.get("received_wall_ns", 0))
    return events


def load_events_across_runs(
    data_dir: Path,
    sources: set[str] | None = None,
    *,
    received_at_or_before_ns: int | None = None,
) -> list[dict[str, Any]]:

    runs_dir = data_dir / "runs"
    if not runs_dir.exists():
        return []
    events: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for event in load_events(run_dir, sources):
            if (
                received_at_or_before_ns is None
                or event.get("received_wall_ns", 0) <= received_at_or_before_ns
            ):
                events.append(event)
    events.sort(key=lambda item: item.get("received_wall_ns", 0))
    return events


def weather_observation_key(event: dict[str, Any]) -> str:
    payload = event.get("payload", {})
    station_id = str(event.get("station_id") or payload.get("station_id") or "")
    observation_time = str(
        event.get("observation_timestamp_ns") or payload.get("observation_time") or ""
    )
    raw = str(payload.get("raw_observation") or "").strip()
    if raw:
        normalized_raw = re.sub(r"\s+", " ", raw.upper()).strip()
        normalized_raw = re.sub(r"^(?:METAR|SPECI)\s+", "", normalized_raw)
        normalized_raw = normalized_raw.removesuffix("$").rstrip()
        return f"{station_id}|{observation_time}|raw:{normalized_raw}"
    return (
        f"{station_id}|{observation_time}|source:{event.get('source')}|"
        f"temp:{payload.get('temperature_c')}"
    )


def dedupe_weather_observations(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    first_seen: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=weather_event_sort_key):
        first_seen.setdefault(weather_observation_key(event), event)
    return sorted(first_seen.values(), key=weather_event_sort_key)


def weather_event_sort_key(event: dict[str, Any]) -> tuple[int, int]:

    return (
        int(event.get("received_wall_ns") or 0),
        int(event.get("observation_timestamp_ns") or 0),
    )


def latest_before(events: list[dict[str, Any]], timestamp_ns: int) -> dict[str, Any] | None:
    result = None
    for event in events:
        if event["received_wall_ns"] > timestamp_ns:
            break
        result = event
    return result


def first_after(events: Iterable[dict[str, Any]], timestamp_ns: int) -> dict[str, Any] | None:
    for event in events:
        if event["received_wall_ns"] >= timestamp_ns:
            return event
    return None
