from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .domain import EventEnvelope, new_run_id
from .http_client import TimedHttpClient
from .power import inhibit_system_sleep, release_system_sleep_inhibition
from .resources import process_memory
from .sources.awc import AviationWeatherCollector
from .sources.forecasts import ForecastCollector
from .sources.knmi import KnmiCollector
from .sources.nws_tgftp import NwsTgftpCollector
from .sources.polymarket import PolymarketCollector
from .sources.wis2 import Wis2Collector
from .sources.wunderground import WundergroundCollector
from .storage.clickhouse import ClickHouseEventSink
from .storage.jsonl import JsonlEventStore
from .storage.market_capture import MarketCaptureSink
from .storage.multi import MultiSink
from .timebase import now_pair

logger = logging.getLogger(__name__)


async def run_collection(cfg: AppConfig, duration_seconds: float | None = None) -> str:
    run_id = new_run_id()
    sleep_inhibition = inhibit_system_sleep()
    json_store = JsonlEventStore(
        cfg.data_dir,
        run_id,
        cfg.collector.raw_fsync_every,
        compress_rotated_files=cfg.collector.compress_rotated_files,
        compression_level=cfg.collector.compression_level,
        queue_max_events=cfg.collector.raw_queue_max_events,
        write_batch_events=cfg.collector.raw_write_batch_events,
        write_flush_seconds=cfg.collector.raw_write_flush_seconds,
    )
    best_effort_sinks = []
    if cfg.clickhouse.enabled:
        best_effort_sinks.append(
            ClickHouseEventSink(
                flush_rows=cfg.clickhouse.flush_rows,
                flush_seconds=cfg.clickhouse.flush_seconds,
            )
        )
    base_sink = MultiSink([json_store], best_effort_sinks)
    sink = MarketCaptureSink(base_sink, cfg.market, run_id)
    http = TimedHttpClient(cfg.collector.request_timeout_seconds)
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "config_hash": cfg.config_hash(),
        "stations": {key: station.model_dump() for key, station in cfg.stations.items()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "sleep_inhibition": sleep_inhibition,
    }
    await json_store.write_manifest(manifest)

    collectors = {
        "wis2": Wis2Collector(cfg, run_id, sink).run(),
        "awc": AviationWeatherCollector(cfg, run_id, sink, http).run(),
        "nws_tgftp": NwsTgftpCollector(cfg, run_id, sink, http).run(),
        "knmi": KnmiCollector(cfg, run_id, sink, http).run(),
        "wunderground": WundergroundCollector(cfg, run_id, sink).run(),
        "forecasts": ForecastCollector(cfg, run_id, sink, http).run(),
        "polymarket": PolymarketCollector(cfg, run_id, sink, http).run(),
        "heartbeat": _heartbeat(cfg, run_id, sink, json_store),
    }
    tasks = {name: asyncio.create_task(coro, name=name) for name, coro in collectors.items()}


    supervised_names = {"polymarket", "heartbeat"}
    source_enabled = {
        "wis2": any(station.sources.wis2 for station in cfg.stations.values()),
        "awc": any(station.sources.awc for station in cfg.stations.values()),
        "nws_tgftp": any(station.sources.nws_tgftp for station in cfg.stations.values()),
        "wunderground": any(
            station.sources.wunderground for station in cfg.stations.values()
        ),
        "forecasts": any(station.sources.forecasts for station in cfg.stations.values()),
        "knmi": bool(
            any(station.sources.knmi for station in cfg.stations.values())
            and (
                os.getenv("KNMI_OPEN_DATA_API_KEY", "").strip()
                or os.getenv("KNMI_API_KEY", "").strip()
            )
        ),
    }
    supervised_names.update(name for name, enabled in source_enabled.items() if enabled)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    stop_request_path = cfg.data_dir / "runs" / run_id / "stop.requested"

    def request_stop() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    logger.info("Collection run %s started", run_id)
    status = "completed"
    failure: BaseException | None = None
    stop_task = asyncio.create_task(
        _wait_for_stop_conditions(
            stop,
            stop_request_path,
            cfg.data_dir,
            cfg.collector.minimum_free_disk_gb,
            cfg.collector.maximum_process_private_memory_mb,
            cfg.collector.resource_check_seconds,
        ),
        name="stop_wait",
    )
    effective_duration_seconds = (
        duration_seconds
        if duration_seconds is not None
        else cfg.collector.maximum_run_seconds
    )
    duration_task = asyncio.create_task(
        asyncio.sleep(effective_duration_seconds), name="duration_wait"
    )
    completion_reason = "duration" if duration_seconds is not None else "planned_recycle"
    try:
        watched = {stop_task, *(tasks[name] for name in supervised_names)}
        watched.add(duration_task)
        done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            completion_reason = stop_task.result()
        if stop_task not in done and duration_task not in done:
            stopped = next(task for task in done if task in tasks.values())
            name = stopped.get_name()
            exc = stopped.exception()
            if exc is None:
                raise RuntimeError(f"Required collector stopped unexpectedly: {name}")
            raise RuntimeError(f"Required collector failed: {name}") from exc
    except asyncio.CancelledError as exc:
        status = "cancelled"
        failure = exc
        raise
    except BaseException as exc:
        status = "failed"
        failure = exc
        raise
    finally:
        stop_task.cancel()
        duration_task.cancel()
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(stop_task, *(tasks.values()), return_exceptions=True)
        await asyncio.gather(duration_task, return_exceptions=True)
        await http.close()
        close_failure: BaseException | None = None
        try:

            await sink.close()
        except BaseException as exc:
            close_failure = exc
            status = "failed"
            if failure is None:
                failure = exc
        sleep_inhibition["released"] = release_system_sleep_inhibition()
        manifest.update(
            {
                "status": status,
                "completion_reason": completion_reason,
                "completed_at": datetime.now(UTC).isoformat(),
                "event_counts": json_store.event_counts(),
                "rotated_archives": json_store.rotated_archives(),
                "storage": json_store.metrics(),
                "market_capture": sink.metrics(),
                "process_memory": process_memory(),
                "failure": (
                    {"type": type(failure).__name__, "message": str(failure)[:1000]}
                    if failure is not None
                    else None
                ),
            }
        )
        await json_store.write_manifest(manifest)
        if completion_reason == "low_disk_space":
            supervisor_stop_path = cfg.data_dir / "collector_supervisor.stop"
            supervisor_stop_path.write_text(
                f"low disk safety stop at {datetime.now(UTC).isoformat()}\n",
                encoding="utf-8",
            )
        logger.info("Collection run %s stopped", run_id)
        if close_failure is not None and failure is close_failure:
            raise close_failure
    return run_id


async def _wait_for_stop_conditions(
    stop: asyncio.Event,
    request_path: Path,
    data_dir: Path,
    minimum_free_disk_gb: float,
    maximum_process_private_memory_mb: int,
    resource_check_seconds: float,
) -> str:

    next_resource_check = 0.0
    minimum_free_bytes = int(minimum_free_disk_gb * 1_000_000_000)
    maximum_private_bytes = maximum_process_private_memory_mb * 1_000_000
    while True:
        if request_path.exists():
            stop.set()
            return "stop_request_file"
        if stop.is_set():
            return "operating_system_signal"
        monotonic_now = time.monotonic()
        if monotonic_now >= next_resource_check:
            if shutil.disk_usage(data_dir).free < minimum_free_bytes:
                stop.set()
                return "low_disk_space"
            memory = process_memory()
            private_bytes = memory.get("private_bytes")
            if isinstance(private_bytes, int) and private_bytes >= maximum_private_bytes:
                stop.set()
                return "memory_limit"
            next_resource_check = monotonic_now + resource_check_seconds
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass


async def _heartbeat(
    cfg: AppConfig,
    run_id: str,
    sink: MarketCaptureSink,
    json_store: JsonlEventStore,
) -> None:
    baseline = now_pair()
    while True:
        current = now_pair()
        wall_elapsed = current.wall_ns - baseline.wall_ns
        mono_elapsed = current.mono_ns - baseline.mono_ns
        await sink.append(
            EventEnvelope.build(
                run_id=run_id,
                source="system",
                event_type="heartbeat",
                payload={
                    "utc": datetime.now(UTC).isoformat(),
                    "config_hash": cfg.config_hash(),
                    "wall_minus_monotonic_ns": current.wall_ns - current.mono_ns,
                    "clock_step_since_start_ns": wall_elapsed - mono_elapsed,
                    "storage": json_store.metrics(),
                    "market_capture": sink.metrics(),
                    "process_memory": process_memory(),
                },
                received_wall_ns=current.wall_ns,
                received_mono_ns=current.mono_ns,
            )
        )
        await asyncio.sleep(cfg.collector.heartbeat_seconds)
