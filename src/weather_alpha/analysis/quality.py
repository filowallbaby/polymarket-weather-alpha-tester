from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig, StationConfig
from ..storage.archive import event_files
from .io import iter_events, resolve_run_dir


def audit_run(cfg: AppConfig, run_id: str | None = None) -> dict[str, Path]:

    run_dir = resolve_run_dir(cfg.data_dir, run_id)
    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        status: str,
        detail: str,
        *,
        station_id: str | None = None,
        value: Any = None,
    ) -> None:
        rows.append(
            {
                "check": check,
                "status": status,
                "station_id": station_id,
                "value": value,
                "detail": detail,
            }
        )

    manifest = _read_manifest(run_dir / "manifest.json")
    small_sources = {
        "awc_metar",
        "awc_peer_metar",
        "wis2_metar",
        "wis2_peer_metar",
        "knmi_10m",
        "nws_tgftp_metar",
        "polymarket_gamma",
        "polymarket_ws",
        "system",
        "wunderground_browser",
        *{
            f"forecast_{provider}"
            for station in cfg.stations.values()
            for provider in station.forecast_providers
        },
    }


    events = list(iter_events(run_dir, small_sources))
    valid_book_assets: defaultdict[str, set[str]] = defaultdict(set)
    book_events = 0
    for event in iter_events(run_dir, {"polymarket_book"}):
        if event.get("event_type") != "book_state":
            continue
        book_events += 1
        if event.get("payload", {}).get("depth_valid") is not True:
            continue
        station_id = str(event.get("station_id") or "")
        asset_id = str(event.get("asset_id") or "")
        if station_id and asset_id:
            valid_book_assets[station_id].add(asset_id)

    status = manifest.get("status")
    recovered = bool(manifest.get("recovery", {}).get("verified_events"))
    manifest_ok = status == "completed" or (status == "interrupted" and recovered)
    add(
        "manifest_status",
        "PASS" if manifest_ok else "FAIL",
        f"status={status!r}, recovery_verified={recovered}",
    )
    raw_files = event_files(run_dir)
    verified_events = manifest.get("recovery", {}).get("verified_events")
    if verified_events is None:
        counts = manifest.get("event_counts") or {}
        verified_events = sum(int(value) for value in counts.values()) if counts else None
    observed_events = int(verified_events or (len(events) + book_events))
    add(
        "raw_files",
        "PASS" if raw_files and observed_events else "FAIL",
        f"files={len(raw_files)}, verified_events={observed_events}",
        value=observed_events,
    )

    collector_errors = [
        event
        for event in events
        if event.get("event_type")
        in {"collector_error", "connection_error", "decode_error", "parse_error"}
    ]
    add(
        "collector_errors",
        "WARN" if collector_errors else "PASS",
        f"count={len(collector_errors)}",
        value=len(collector_errors),
    )

    heartbeats = [
        event
        for event in events
        if event.get("source") == "system" and event.get("event_type") == "heartbeat"
    ]
    add(
        "system_heartbeat",
        "PASS" if heartbeats else "FAIL",
        f"count={len(heartbeats)}",
        value=len(heartbeats),
    )
    cumulative_clock_drift = [
        float(event.get("payload", {}).get("clock_step_since_start_ns", 0)) / 1e6
        for event in heartbeats
    ]
    adjacent_clock_steps = [
        abs(current - previous)
        for previous, current in zip(cumulative_clock_drift, cumulative_clock_drift[1:])
    ]
    max_clock_step = max(adjacent_clock_steps, default=0.0 if heartbeats else None)
    add(
        "clock_stability",
        "PASS"
        if max_clock_step is not None and max_clock_step <= cfg.quality.max_clock_step_ms
        else "FAIL",
        f"max_step_ms={max_clock_step}, limit_ms={cfg.quality.max_clock_step_ms}",
        value=max_clock_step,
    )

    ws_health = sorted(
        (
            event
            for event in events
            if event.get("source") == "polymarket_ws"
            and event.get("event_type") in {"subscribed", "heartbeat_rtt"}
        ),
        key=lambda event: event["received_wall_ns"],
    )
    ws_gaps = [
        (later["received_wall_ns"] - earlier["received_wall_ns"]) / 1e9
        for earlier, later in zip(ws_health, ws_health[1:], strict=False)
    ]
    max_ws_gap = max(ws_gaps, default=0.0 if ws_health else None)
    recovery = manifest.get("recovery", {})
    run_first_ns = int(recovery.get("first_received_wall_ns") or 0) or min(
        (int(event.get("received_wall_ns") or 0) for event in events),
        default=0,
    )
    run_last_ns = int(recovery.get("last_received_wall_ns") or 0) or max(
        (int(event.get("received_wall_ns") or 0) for event in events),
        default=0,
    )
    ws_intervals = list(ws_gaps)
    if ws_health and run_first_ns:
        ws_intervals.append(max(0.0, (ws_health[0]["received_wall_ns"] - run_first_ns) / 1e9))
    if ws_health and run_last_ns:
        ws_intervals.append(max(0.0, (run_last_ns - ws_health[-1]["received_wall_ns"]) / 1e9))
    run_span_seconds = (
        (run_last_ns - run_first_ns) / 1e9 if run_last_ns > run_first_ns else 0.0
    )
    unhealthy_seconds = sum(
        max(0.0, gap - cfg.execution.max_ws_health_age_seconds) for gap in ws_intervals
    )
    ws_health_coverage = (
        max(0.0, 1.0 - unhealthy_seconds / run_span_seconds)
        if ws_health and run_span_seconds > 0
        else None
    )

    for station in cfg.stations.values():
        _audit_station(
            cfg,
            station,
            events,
            add,
            max_ws_gap=max_ws_gap,
            ws_health_coverage=ws_health_coverage,
            ws_unhealthy_seconds=unhealthy_seconds,
            run_last_ns=run_last_ns,
            valid_book_assets=valid_book_assets.get(station.station_id, set()),
            knmi_configured=bool(
                os.getenv("KNMI_OPEN_DATA_API_KEY", "").strip()
                or os.getenv("KNMI_API_KEY", "").strip()
            ),
        )

    report_dir = cfg.reports_dir / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quality_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["check", "status", "station_id", "value", "detail"]
        )
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]
    summary_path = report_dir / "quality_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "ok": not failures,
                "checks": len(rows),
                "failures": failures,
                "warnings": warnings,
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"quality": report_path, "summary": summary_path}


def _audit_station(
    cfg: AppConfig,
    station: StationConfig,
    events: list[dict[str, Any]],
    add: Any,
    *,
    max_ws_gap: float | None,
    ws_health_coverage: float | None,
    ws_unhealthy_seconds: float,
    run_last_ns: int,
    valid_book_assets: set[str],
    knmi_configured: bool,
) -> None:
    station_events = [event for event in events if event.get("station_id") == station.station_id]

    def count(source: str, event_type: str) -> int:
        return sum(
            event.get("source") == source and event.get("event_type") == event_type
            for event in station_events
        )

    if station.sources.wis2:
        value = count("wis2_metar", "weather_observation")
        connected = sum(
            event.get("source") == "wis2_metar"
            and event.get("event_type") in {"mqtt_connected", "mqtt_reconnected"}
            for event in events
        )
        add(
            "wis2_observations",
            "PASS" if value else "WARN" if connected else "FAIL",
            f"count={value}, mqtt_connections={connected}",
            station_id=station.station_id,
            value=value,
        )

    if station.sources.awc:
        value = count("awc_metar", "weather_observation")
        add(
            "awc_observations",
            "PASS" if value else "FAIL",
            f"count={value}",
            station_id=station.station_id,
            value=value,
        )
    if station.sources.nws_tgftp:
        value = count("nws_tgftp_metar", "weather_observation")
        add(
            "nws_observations",
            "PASS" if value else "FAIL",
            f"count={value}",
            station_id=station.station_id,
            value=value,
        )
    if station.sources.knmi:
        value = count("knmi_10m", "weather_observation")
        status = "PASS" if value else "FAIL" if knmi_configured else "WARN"
        detail = (
            f"count={value}"
            if knmi_configured
            else "KNMI_OPEN_DATA_API_KEY not configured"
        )
        add("knmi_observations", status, detail, station_id=station.station_id, value=value)

    if station.sources.forecasts:
        for provider in station.forecast_providers:
            value = count(f"forecast_{provider}", "forecast_snapshot")
            add(
                f"forecast_{provider}",
                "PASS" if value else "FAIL",
                f"count={value}",
                station_id=station.station_id,
                value=value,
            )

    if station.sources.wunderground:
        snapshots = [
            event
            for event in station_events
            if event.get("source") == "wunderground_browser"
            and event.get("event_type") == "resolution_snapshot"
        ]
        usable = [
            event for event in snapshots if event.get("payload", {}).get("observation_count", 0) > 0
        ]
        units_ok = bool(usable) and all(
            event.get("payload", {}).get("unit_verified") is True for event in usable
        )
        mismatches = sum(
            int(event.get("payload", {}).get("station_mismatch_count", 0)) for event in snapshots
        )
        add(
            "wu_snapshots",
            "PASS" if usable else "FAIL",
            f"snapshots={len(snapshots)}, usable={len(usable)}",
            station_id=station.station_id,
            value=len(usable),
        )
        add(
            "wu_units",
            "PASS" if units_ok else "FAIL",
            f"verified={units_ok}",
            station_id=station.station_id,
        )
        add(
            "wu_station_filter",
            "WARN" if mismatches else "PASS",
            f"rejected={mismatches}",
            station_id=station.station_id,
            value=mismatches,
        )
        local_today = datetime.now(ZoneInfo(station.timezone)).date().isoformat()
        past_dates = {
            str(event.get("payload", {}).get("local_date") or "")
            for event in snapshots
            if str(event.get("payload", {}).get("local_date") or "") < local_today
        }
        missing_final = sorted(
            date
            for date in past_dates
            if not any(
                event.get("payload", {}).get("local_date") == date
                and event.get("payload", {}).get("resolution_final") is True
                for event in snapshots
            )
        )
        finalization_grace_seconds = (
            cfg.collector.wunderground_poll_seconds + cfg.collector.request_timeout_seconds
        )
        required_missing: list[str] = []
        partial_edge_dates: list[str] = []
        timezone = ZoneInfo(station.timezone)
        for local_date in missing_final:
            local_day_end = datetime.combine(
                date.fromisoformat(local_date) + timedelta(days=1),
                time.min,
                tzinfo=timezone,
            )
            required_after_ns = int(
                (local_day_end.timestamp() + finalization_grace_seconds) * 1e9
            )
            if run_last_ns >= required_after_ns:
                required_missing.append(local_date)
            else:
                partial_edge_dates.append(local_date)
        add(
            "wu_finalization",
            "FAIL" if required_missing else "PASS",
            f"required_missing_dates={required_missing}",
            station_id=station.station_id,
        )
        if partial_edge_dates:
            add(
                "wu_partial_dates",
                "WARN",
                f"run_ended_before_finalization={partial_edge_dates}",
                station_id=station.station_id,
                value=len(partial_edge_dates),
            )

    tracked = {
        event.get("asset_id")
        for event in station_events
        if event.get("source") == "polymarket_gamma"
        and event.get("event_type") == "tracked_token"
        and event.get("asset_id")
    }
    valid_books = valid_book_assets
    add(
        "market_buckets",
        "PASS" if len(tracked) >= cfg.quality.minimum_market_buckets else "FAIL",
        f"tracked={len(tracked)}, minimum={cfg.quality.minimum_market_buckets}",
        station_id=station.station_id,
        value=len(tracked),
    )
    missing_books = tracked - valid_books
    add(
        "book_coverage",
        "PASS" if tracked and not missing_books else "FAIL",
        f"valid={len(valid_books)}, missing={len(missing_books)}",
        station_id=station.station_id,
        value=len(valid_books),
    )
    gap_status = "FAIL"
    if max_ws_gap is not None:
        if max_ws_gap <= cfg.quality.max_ws_health_gap_seconds:
            gap_status = "PASS"
        elif max_ws_gap <= cfg.analysis.continuity_gap_seconds:
            gap_status = "WARN"
    add(
        "ws_health_gap",
        gap_status,
        (
            f"max_gap_seconds={max_ws_gap}, warning_above="
            f"{cfg.quality.max_ws_health_gap_seconds}, failure_above="
            f"{cfg.analysis.continuity_gap_seconds}"
        ),
        station_id=station.station_id,
        value=max_ws_gap,
    )
    add(
        "ws_health_coverage",
        (
            "PASS"
            if ws_health_coverage is not None
            and ws_health_coverage >= cfg.quality.minimum_ws_health_coverage
            else "FAIL"
        ),
        (
            f"coverage={ws_health_coverage}, minimum="
            f"{cfg.quality.minimum_ws_health_coverage}, "
            f"unhealthy_seconds={ws_unhealthy_seconds}"
        ),
        station_id=station.station_id,
        value=ws_health_coverage,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
