from __future__ import annotations

import gzip

import orjson
import pytest

from weather_alpha.analysis.io import load_events
from weather_alpha.storage.archive import archive_run, event_files


def _write_run(data_dir, run_id: str, status: str = "completed"):
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_bytes(orjson.dumps({"run_id": run_id, "status": status}))
    events = [
        {"source": "system", "event_type": "heartbeat", "received_wall_ns": index}
        for index in range(3)
    ]
    raw = b"".join(orjson.dumps(event) + b"\n" for event in events)
    (run_dir / "2026-07-20__system.jsonl").write_bytes(raw)
    return run_dir, events


def test_archive_run_verifies_compresses_and_remains_readable(tmp_path) -> None:
    data_dir = tmp_path / "data"
    run_dir, expected = _write_run(data_dir, "run")

    summary = archive_run(data_dir, "run")

    assert summary["compressed_bytes"] < summary["original_bytes"]
    assert not (run_dir / "2026-07-20__system.jsonl").exists()
    archive = run_dir / "2026-07-20__system.jsonl.gz"
    assert archive.exists()
    assert load_events(run_dir) == expected
    assert event_files(run_dir) == [archive]
    manifest = orjson.loads((run_dir / "manifest.json").read_bytes())
    assert manifest["archive"]["originals_kept"] is False


def test_event_files_prefers_plain_copy_when_both_exist(tmp_path) -> None:
    data_dir = tmp_path / "data"
    run_dir, expected = _write_run(data_dir, "run")
    source = run_dir / "2026-07-20__system.jsonl"
    with gzip.open(run_dir / "2026-07-20__system.jsonl.gz", "wb") as handle:
        handle.write(source.read_bytes())

    assert load_events(run_dir) == expected
    assert event_files(run_dir) == [source]


def test_archive_run_rejects_active_or_failed_run(tmp_path) -> None:
    data_dir = tmp_path / "data"
    _write_run(data_dir, "running", status="running")

    with pytest.raises(ValueError, match="Only completed, gracefully cancelled"):
        archive_run(data_dir, "running")


def test_archive_run_accepts_gracefully_cancelled_run(tmp_path) -> None:
    data_dir = tmp_path / "data"
    run_dir, expected = _write_run(data_dir, "cancelled", status="cancelled")

    summary = archive_run(data_dir, "cancelled")

    assert summary["run_id"] == "cancelled"
    assert not list(run_dir.glob("*.jsonl"))
    assert list(run_dir.glob("*.jsonl.gz"))
    assert load_events(run_dir) == expected
