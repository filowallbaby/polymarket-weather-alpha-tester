from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime

import orjson

from weather_alpha.analysis.predictive import build_predictive_dataset
from weather_alpha.config import load_config
from weather_alpha.domain import EventEnvelope


def _event(
    *,
    run_id: str,
    source: str,
    event_type: str,
    wall_ns: int,
    payload: dict,
    station_id="KLGA",
    asset_id=None,
):
    return EventEnvelope.build(
        run_id=run_id,
        source=source,
        event_type=event_type,
        payload=payload,
        station_id=station_id,
        asset_id=asset_id,
        received_wall_ns=wall_ns,
        received_mono_ns=wall_ns,
    ).model_dump(mode="json")


def test_predictive_reports_build_from_synthetic_run(tmp_path) -> None:
    cfg = load_config(project_root=".")
    cfg = cfg.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "reports_dir": tmp_path / "reports",
        }
    )
    run_id = "synthetic"
    run_dir = cfg.data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    base = int(datetime(2026, 7, 16, 16, 0, tzinfo=UTC).timestamp() * 1e9)
    asset = "asset-84"
    common_book = {
        "event_date": "2026-07-16",
        "bucket_label": "84°F",
        "bids": [{"price": "0.18", "size": "1000"}],
        "asks": [{"price": "0.22", "size": "1000"}],
        "best_bid": "0.18",
        "best_ask": "0.22",
        "mid": "0.20",
        "depth_valid": True,
        "source_event_type": "book",
    }
    events = [
        _event(
            run_id=run_id,
            source="awc_metar",
            event_type="weather_observation",
            wall_ns=base - 11 * 60 * 60 * 1_000_000_000,
            payload={
                "station_id": "KLGA",
                "temperature_c": 20.0,
                "observation_time": "2026-07-16T05:00:00+00:00",
                "raw_observation": "KLGA 160500Z 00000KT 10SM CLR 20/15 A3000",
            },
        ),
        _event(
            run_id=run_id,
            source="forecast_ecmwf",
            event_type="forecast_snapshot",
            wall_ns=base - 10 * 60 * 1_000_000_000,
            payload={
                "summary": {"today_hourly_max_c": 31.0, "remaining_hourly_max_c": 30.5},
                "forecast": {"hourly": {"time": ["2026-07-16T13:00"], "temperature_2m": [30.0]}},
            },
        ),
        _event(
            run_id=run_id,
            source="awc_peer_metar",
            event_type="peer_weather_observation",
            wall_ns=base - 5 * 60 * 1_000_000_000,
            payload={
                "peer_station_id": "KJFK",
                "target_station_id": "KLGA",
                "temperature_c": 30.5,
                "dewpoint_c": 20.0,
                "latitude": 40.6413,
                "longitude": -73.7781,
                "observation_time": "2026-07-16T15:55:00+00:00",
            },
        ),
        _event(
            run_id=run_id,
            source="polymarket_ws",
            event_type="heartbeat_rtt",
            wall_ns=base - 1_000_000_000,
            payload={"rtt_ms": 10.0},
            station_id=None,
        ),
        _event(
            run_id=run_id,
            source="polymarket_book",
            event_type="book_state",
            wall_ns=base - 120 * 1_000_000_000,
            payload=common_book,
            asset_id=asset,
        ),
        _event(
            run_id=run_id,
            source="forecast_ecmwf",
            event_type="forecast_revision",
            wall_ns=base - 60 * 1_000_000_000,
            payload={
                "provider": "ecmwf",
                "local_date": "2026-07-16",
                "market_unit": "F",
                "previous_content_sha256": "old",
                "content_sha256": "new",
                "previous_today_max_c": 28.0,
                "current_today_max_c": 28.9,
                "previous_market_max": 82.4,
                "current_market_max": 84.02,
                "previous_bucket_value": 82,
                "current_bucket_value": 84,
                "bucket_changed": True,
            },
        ),
        _event(
            run_id=run_id,
            source="awc_metar",
            event_type="weather_observation",
            wall_ns=base,
            payload={
                "station_id": "KLGA",
                "temperature_c": 29.0,
                "dewpoint_c": 20.0,
                "wind_direction_deg": 150.0,
                "wind_speed_kt": 12.0,
                "observation_time": "2026-07-16T16:00:00+00:00",
                "raw_observation": "KLGA 161600Z 15012KT 10SM CLR 29/20 A3000",
            },
        ),
        _event(
            run_id=run_id,
            source="polymarket_book",
            event_type="book_state",
            wall_ns=base + 2 * 1_000_000_000,
            payload={**common_book, "best_bid": "0.48", "best_ask": "0.52", "mid": "0.50"},
            asset_id=asset,
        ),
        _event(
            run_id=run_id,
            source="wunderground_browser",
            event_type="resolution_snapshot",
            wall_ns=base + 5 * 1_000_000_000,
            payload={
                "local_date": "2026-07-16",
                "daily_max_from_network": 84.0,
                "latest_observation_timestamp_ns": base,
                "resolution_final": True,
            },
        ),
    ]
    by_source = defaultdict(list)
    for event in events:
        by_source[event["source"]].append(event)
    for source, source_events in by_source.items():
        path = run_dir / f"2026-07-16__{source}.jsonl"
        path.write_bytes(b"".join(orjson.dumps(event) + b"\n" for event in source_events))

    predictive_paths = build_predictive_dataset(cfg, run_id)

    feature_text = predictive_paths["features"].read_text(encoding="utf-8")
    assert "peer_KJFK_temperature_c" in feature_text
    assert "model_ecmwf_today_max_c" in feature_text
    with predictive_paths["features"].open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))
    forecast_row = next(row for row in feature_rows if row["decision_mode"] == "forecast_update")
    assert forecast_row["trigger_source"] == "forecast_ecmwf"
    assert float(forecast_row["model_ecmwf_today_max_c"]) == 30.0
    with predictive_paths["forecast_reactions"].open(newline="", encoding="utf-8") as handle:
        reaction_rows = list(csv.DictReader(handle))
    t0 = next(row for row in reaction_rows if row["horizon_seconds"] == "0")
    assert t0["current_bucket_label"] in {"84°F", "84Â°F"}
    assert float(t0["market_ask"]) == 0.22
    assert float(t0["market_ask_size"]) == 1000.0
    assert t0["revision_correct"] == "True"
