from __future__ import annotations

import orjson

from weather_alpha.analysis.continuity import continuity_at, detect_continuity_gaps
from weather_alpha.analysis.predictive import _stream_market_quote_snapshots


def test_suspend_gap_marks_post_resume_warmup_ineligible() -> None:
    events = [
        {
            "source": "system",
            "event_type": "heartbeat",
            "received_wall_ns": int(seconds * 1e9),
        }
        for seconds in (1, 61, 121, 1_001, 1_061)
    ]
    gaps = detect_continuity_gaps(
        events,
        max_gap_seconds=180,
        resume_warmup_seconds=600,
    )

    assert len(gaps) == 1
    assert gaps[0].gap_seconds == 880.0
    assert continuity_at(int(100 * 1e9), gaps) == (0, True)
    assert continuity_at(int(500 * 1e9), gaps) == (1, False)
    assert continuity_at(int(1_500 * 1e9), gaps) == (1, False)
    assert continuity_at(int(1_601 * 1e9), gaps) == (1, True)


def test_order_books_are_joined_in_one_streaming_pass(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        _book_event(100, "bucket-a", 0.20),
        _book_event(200, "bucket-a", 0.30),
        _book_event(201, "bucket-b", 0.40),
    ]
    (run_dir / "2026-07-21__polymarket_book.jsonl").write_bytes(
        b"".join(orjson.dumps(event) + b"\n" for event in events)
    )
    decisions = [
        (("run", 150, "weather-a"), "KLGA", "2026-07-21", 150),
        (("run", 250, "weather-b"), "KLGA", "2026-07-21", 250),
    ]

    snapshots, labels = _stream_market_quote_snapshots(run_dir, decisions)

    first = snapshots[("run", 150, "weather-a")]
    second = snapshots[("run", 250, "weather-b")]
    assert first["mid"] == {"bucket-a": 0.20}
    assert first["coverage"]["ratio"] == 0.5
    assert second["mid"] == {"bucket-a": 0.30, "bucket-b": 0.40}
    assert second["coverage"]["ratio"] == 1.0
    assert labels[("KLGA", "2026-07-21")] == {"bucket-a", "bucket-b"}


def _book_event(received_wall_ns: int, label: str, mid: float) -> dict:
    return {
        "run_id": "run",
        "source": "polymarket_book",
        "event_type": "book_state",
        "station_id": "KLGA",
        "received_wall_ns": received_wall_ns,
        "payload": {
            "event_date": "2026-07-21",
            "bucket_label": label,
            "mid": mid,
            "best_bid": mid - 0.01,
            "best_ask": mid + 0.01,
            "fee_rate": 0.05,
            "fee_exponent": 1,
            "fees_enabled": True,
        },
    }
