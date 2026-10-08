from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, datetime

import orjson

from weather_alpha.analysis.predictive import build_predictive_dataset
from weather_alpha.config import load_config
from weather_alpha.domain import EventEnvelope


def _event(run_id: str, source: str, event_type: str, wall_ns: int, payload: dict, **ids):
    return EventEnvelope.build(
        run_id=run_id,
        source=source,
        event_type=event_type,
        payload=payload,
        received_wall_ns=wall_ns,
        received_mono_ns=wall_ns,
        **ids,
    ).model_dump(mode="json")


def _write_run(cfg, run_id: str, events: list[dict]) -> None:
    run_dir = cfg.data_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    by_source = defaultdict(list)
    for event in events:
        by_source[event["source"]].append(event)
    for source, source_events in by_source.items():
        path = run_dir / f"2026-07-16__{source}.jsonl"
        path.write_bytes(b"".join(orjson.dumps(event) + b"\n" for event in source_events))


def test_restart_context_seeds_max_and_joins_later_finalization(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    early = int(datetime(2026, 7, 16, 5, 0, tzinfo=UTC).timestamp() * 1e9)
    midday = int(datetime(2026, 7, 16, 16, 0, tzinfo=UTC).timestamp() * 1e9)
    run1 = "20260716T050000Z-run1"
    run2 = "20260716T160000Z-run2"
    _write_run(
        cfg,
        run1,
        [
            _event(
                run1,
                "awc_metar",
                "weather_observation",
                early,
                {
                    "station_id": "KLGA",
                    "temperature_c": 20.0,
                    "observation_time": "2026-07-16T05:00:00+00:00",
                    "raw_observation": "KLGA 160500Z 00000KT 10SM CLR 20/15 A3000",
                },
                station_id="KLGA",
            )
        ],
    )
    book = {
        "event_date": "2026-07-16",
        "bucket_label": "84°F",
        "bids": [{"price": "0.18", "size": "1000"}],
        "asks": [{"price": "0.22", "size": "1000"}],
        "best_bid": "0.18",
        "best_ask": "0.22",
        "mid": "0.20",
        "depth_valid": True,
        "source_event_type": "book",
        "accepting_orders": True,
        "fees_enabled": True,
        "fee_rate": 0.05,
        "fee_exponent": 1,
    }
    _write_run(
        cfg,
        run2,
        [
            _event(
                run2,
                "polymarket_ws",
                "heartbeat_rtt",
                midday - 2_000_000_000,
                {"rtt_ms": 10.0},
            ),
            _event(
                run2,
                "polymarket_book",
                "book_state",
                midday - 1_000_000_000,
                book,
                station_id="KLGA",
                asset_id="asset-84",
            ),
            _event(
                run2,
                "awc_metar",
                "weather_observation",
                midday,
                {
                    "station_id": "KLGA",
                    "temperature_c": 29.0,
                    "observation_time": "2026-07-16T16:00:00+00:00",
                    "raw_observation": "KLGA 161600Z 15012KT 10SM CLR 29/20 A3000",
                },
                station_id="KLGA",
            ),
            _event(
                run2,
                "wunderground_browser",
                "resolution_snapshot",
                midday + 5_000_000_000,
                {
                    "local_date": "2026-07-16",
                    "daily_max_from_network": 84.0,
                    "resolution_final": True,
                },
                station_id="KLGA",
            ),
        ],
    )

    first_paths = build_predictive_dataset(cfg, run1)
    with first_paths["features"].open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    assert float(first_row["final_wu_max"]) == 84.0

    second_paths = build_predictive_dataset(cfg, run2)
    with second_paths["features"].open(newline="", encoding="utf-8") as handle:
        second_row = next(csv.DictReader(handle))
    assert second_row["day_history_complete"] == "True"
    assert float(second_row["running_max_market_unit"]) > 84.0
