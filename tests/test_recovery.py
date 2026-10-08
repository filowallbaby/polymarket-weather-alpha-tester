from __future__ import annotations

import orjson
import pytest

from weather_alpha.storage.recovery import recover_interrupted_run


def test_recover_interrupted_run_verifies_events_and_finalizes_manifest(tmp_path) -> None:
    data_dir = tmp_path / "data"
    run_dir = data_dir / "runs" / "stale"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_bytes(
        orjson.dumps({"run_id": "stale", "status": "running", "pid": 999_999_999})
    )
    events = [
        {"source": "system", "event_type": "heartbeat", "received_wall_ns": value}
        for value in (10, 20, 30)
    ]
    (run_dir / "2026-07-21__system.jsonl").write_bytes(
        b"".join(orjson.dumps(event) + b"\n" for event in events)
    )

    summary = recover_interrupted_run(data_dir, "stale")
    manifest = orjson.loads((run_dir / "manifest.json").read_bytes())

    assert summary["verified_events"] == 3
    assert manifest["status"] == "interrupted"
    assert manifest["failure"]["type"] == "ExternalInterruption"
    assert manifest["event_counts"] == {"2026-07-21__system.jsonl": 3}


def test_recovery_refuses_non_running_manifest(tmp_path) -> None:
    data_dir = tmp_path / "data"
    run_dir = data_dir / "runs" / "complete"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_bytes(
        orjson.dumps({"run_id": "complete", "status": "completed"})
    )

    with pytest.raises(ValueError, match="not marked running"):
        recover_interrupted_run(data_dir, "complete")
