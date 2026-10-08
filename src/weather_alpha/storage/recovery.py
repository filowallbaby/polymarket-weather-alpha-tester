from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from .archive import event_files, open_event_file


def recover_interrupted_run(data_dir: Path, run_id: str) -> dict[str, Any]:

    run_dir = data_dir / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    if manifest.get("status") != "running":
        raise ValueError(f"Run {run_id} is not marked running")
    pid = int(manifest.get("pid") or 0)
    if pid and _pid_running(pid):
        raise ValueError(f"Run {run_id} process {pid} is still active")

    counts: dict[str, int] = {}
    first_received_ns: int | None = None
    last_received_ns: int | None = None
    for path in event_files(run_dir):
        count = 0
        with open_event_file(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from exc
                received_ns = int(event.get("received_wall_ns") or 0)
                if received_ns:
                    first_received_ns = (
                        received_ns
                        if first_received_ns is None
                        else min(first_received_ns, received_ns)
                    )
                    last_received_ns = (
                        received_ns
                        if last_received_ns is None
                        else max(last_received_ns, received_ns)
                    )
                count += 1
        counts[path.name.removesuffix(".gz")] = count

    recovered_at = datetime.now(UTC).isoformat()
    manifest.update(
        {
            "status": "interrupted",
            "completed_at": recovered_at,
            "event_counts": counts,
            "failure": {
                "type": "ExternalInterruption",
                "message": "Process ended before it could finalize the manifest; raw streams were verified during recovery.",
            },
            "recovery": {
                "recovered_at": recovered_at,
                "verified_files": len(counts),
                "verified_events": sum(counts.values()),
                "first_received_wall_ns": first_received_ns,
                "last_received_wall_ns": last_received_ns,
            },
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest["recovery"]


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.recovery.tmp")
    temporary.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
