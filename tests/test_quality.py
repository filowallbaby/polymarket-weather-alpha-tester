from __future__ import annotations

import csv
import json
from datetime import UTC, datetime

import orjson

from weather_alpha.analysis.quality import audit_run
from weather_alpha.config import load_config
from weather_alpha.domain import EventEnvelope


def test_quality_audit_is_machine_readable_and_fails_missing_sources(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    run_id = "quality-run"
    run_dir = cfg.data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8"
    )
    heartbeat = EventEnvelope.build(
        run_id=run_id,
        source="system",
        event_type="heartbeat",
        payload={"clock_step_since_start_ns": 0},
    ).model_dump(mode="json")
    (run_dir / "2026-07-20__system.jsonl").write_bytes(orjson.dumps(heartbeat) + b"\n")

    paths = audit_run(cfg, run_id)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert summary["failures"]
    report = paths["quality"].read_text(encoding="utf-8")
    assert "clock_stability,PASS" in report
    assert "market_buckets,FAIL" in report


def test_quality_warns_short_ws_gap_and_partial_wu_edge_date(tmp_path) -> None:
    base = load_config(project_root=".")
    cfg = base.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "reports_dir": tmp_path / "reports",
            "quality": base.quality.model_copy(
                update={"minimum_ws_health_coverage": 0.5}
            ),
        }
    )
    run_id = "partial-edge-run"
    run_dir = cfg.data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8"
    )
    start_ns = int(datetime(2026, 7, 20, tzinfo=UTC).timestamp() * 1e9)

    def write_events(source: str, events: list[dict]) -> None:
        path = run_dir / f"2026-07-20__{source}.jsonl"
        path.write_bytes(b"".join(orjson.dumps(event) + b"\n" for event in events))

    write_events(
        "system",
        [
            EventEnvelope.build(
                run_id=run_id,
                source="system",
                event_type="heartbeat",
                payload={"clock_step_since_start_ns": 0},
                received_wall_ns=start_ns,
            ).model_dump(mode="json"),
            EventEnvelope.build(
                run_id=run_id,
                source="system",
                event_type="heartbeat",
                payload={"clock_step_since_start_ns": 0},
                received_wall_ns=start_ns + int(70e9),
            ).model_dump(mode="json"),
        ],
    )
    write_events(
        "polymarket_ws",
        [
            EventEnvelope.build(
                run_id=run_id,
                source="polymarket_ws",
                event_type="subscribed",
                payload={},
                received_wall_ns=start_ns + int(10e9),
            ).model_dump(mode="json"),
            EventEnvelope.build(
                run_id=run_id,
                source="polymarket_ws",
                event_type="heartbeat_rtt",
                payload={"rtt_ms": 1.0},
                received_wall_ns=start_ns + int(60.5e9),
            ).model_dump(mode="json"),
        ],
    )
    write_events(
        "wunderground_browser",
        [
            EventEnvelope.build(
                run_id=run_id,
                source="wunderground_browser",
                event_type="resolution_snapshot",
                station_id="EHAM",
                payload={
                    "local_date": "2026-07-20",
                    "observation_count": 1,
                    "unit_verified": True,
                    "station_mismatch_count": 0,
                    "resolution_final": False,
                },
                received_wall_ns=start_ns + int(20e9),
            ).model_dump(mode="json")
        ],
    )

    paths = audit_run(cfg, run_id)
    with paths["quality"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eham = {
        row["check"]: row
        for row in rows
        if row["station_id"] == "EHAM"
    }
    assert eham["wu_finalization"]["status"] == "PASS"
    assert eham["wu_partial_dates"]["status"] == "WARN"
    assert eham["ws_health_gap"]["status"] == "WARN"
    assert eham["ws_health_coverage"]["status"] == "PASS"
