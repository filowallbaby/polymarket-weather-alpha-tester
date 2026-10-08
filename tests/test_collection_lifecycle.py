from __future__ import annotations

import asyncio
import json

import pytest

from weather_alpha.collect import _wait_for_stop_conditions, run_collection
from weather_alpha.config import load_config


class _Http:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def close(self) -> None:
        pass


class _IdleCollector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def run(self) -> None:
        await asyncio.Event().wait()


class _StoppedCollector(_IdleCollector):
    async def run(self) -> None:
        return None


def _patch_collectors(monkeypatch, *, stopped: bool = False) -> None:
    monkeypatch.setattr("weather_alpha.collect.TimedHttpClient", _Http)
    names = [
        "Wis2Collector",
        "AviationWeatherCollector",
        "NwsTgftpCollector",
        "KnmiCollector",
        "WundergroundCollector",
        "ForecastCollector",
        "PolymarketCollector",
    ]
    for name in names:
        replacement = (
            _StoppedCollector if stopped and name == "AviationWeatherCollector" else _IdleCollector
        )
        monkeypatch.setattr(f"weather_alpha.collect.{name}", replacement)


def test_duration_run_writes_completed_manifest(monkeypatch, tmp_path) -> None:
    _patch_collectors(monkeypatch)
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    run_id = asyncio.run(run_collection(cfg, duration_seconds=0.02))
    manifest = json.loads(
        (cfg.data_dir / "runs" / run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["completed_at"]
    assert manifest["failure"] is None
    assert "sleep_inhibition" in manifest
    assert manifest["sleep_inhibition"]["released"] is True
    assert manifest["process_memory"]["pid"]
    assert sum(manifest["event_counts"].values()) >= 1


def test_required_collector_exit_marks_run_failed(monkeypatch, tmp_path) -> None:
    _patch_collectors(monkeypatch, stopped=True)
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    with pytest.raises(RuntimeError, match="stopped unexpectedly: awc"):
        asyncio.run(run_collection(cfg, duration_seconds=1.0))

    run_dir = next((cfg.data_dir / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def test_stop_request_file_gracefully_finalizes_run(monkeypatch, tmp_path) -> None:
    _patch_collectors(monkeypatch)
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )

    async def scenario() -> str:
        task = asyncio.create_task(run_collection(cfg))
        runs_dir = cfg.data_dir / "runs"
        while not runs_dir.exists() or not any(runs_dir.iterdir()):
            await asyncio.sleep(0.01)
        run_dir = next(runs_dir.iterdir())
        (run_dir / "stop.requested").write_text("{}", encoding="utf-8")
        return await asyncio.wait_for(task, timeout=5)

    run_id = asyncio.run(scenario())
    manifest = json.loads(
        (cfg.data_dir / "runs" / run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["completion_reason"] == "stop_request_file"


def test_low_disk_guard_finalizes_and_stops_supervisor(monkeypatch, tmp_path) -> None:
    _patch_collectors(monkeypatch)
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    cfg = cfg.model_copy(
        update={
            "collector": cfg.collector.model_copy(
                update={"minimum_free_disk_gb": 1000.0}
            )
        }
    )

    run_id = asyncio.run(run_collection(cfg))
    manifest = json.loads(
        (cfg.data_dir / "runs" / run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["completion_reason"] == "low_disk_space"
    assert (cfg.data_dir / "collector_supervisor.stop").exists()


def test_private_memory_guard_requests_graceful_recycle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "weather_alpha.collect.process_memory",
        lambda: {"private_bytes": 300_000_000},
    )
    stop = asyncio.Event()
    reason = asyncio.run(
        _wait_for_stop_conditions(
            stop,
            tmp_path / "missing-stop-file",
            tmp_path,
            minimum_free_disk_gb=1,
            maximum_process_private_memory_mb=256,
            resource_check_seconds=1,
        )
    )
    assert reason == "memory_limit"
    assert stop.is_set()
