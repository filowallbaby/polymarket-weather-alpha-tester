from __future__ import annotations

import json

import pytest

from weather_alpha.analysis.station_maximum_v6_update_alpha import (
    _annotate_forecast_novelty,
    _candidate_episodes,
    _information_delta_rows,
    _join_market_reactions,
    _stream_run_market_snapshots,
)


def test_information_delta_is_defined_before_outcome_join() -> None:
    rows = [
        {
            "decision_id": "before",
            "run_id": "run",
            "station_id": "EHAM",
            "local_date": "2026-07-30",
            "local_hour": 10.0,
            "received_wall_ns": 1_000_000_000,
            "decision_mode": "scheduled",
            "trigger_source": "timer",
            "v6_point_c": 25.0,
            "v6_distribution_json": json.dumps(
                {"25C": 0.6, "26C": 0.4}
            ),
            "market_distribution_json": json.dumps(
                {"25C": 0.5, "26C": 0.5}
            ),
            "winning_bucket": "26C",
            "settled": True,
            "rejection_reason": "",
        },
        {
            "decision_id": "after",
            "run_id": "run",
            "station_id": "EHAM",
            "local_date": "2026-07-30",
            "local_hour": 10.1,
            "received_wall_ns": 2_000_000_000,
            "decision_mode": "forecast_update",
            "trigger_source": "forecast_ecmwf",
            "forecast_content_changed": True,
            "v6_point_c": 25.3,
            "v6_distribution_json": json.dumps(
                {"25C": 0.4, "26C": 0.6}
            ),
            "market_distribution_json": json.dumps(
                {"25C": 0.5, "26C": 0.5}
            ),
            "winning_bucket": "26C",
            "settled": True,
            "rejection_reason": "",
        },
    ]

    events = _information_delta_rows(rows)

    assert len(events) == 1
    assert events[0]["forecast_provider"] == "ecmwf"
    assert events[0]["largest_probability_increase_bucket"] == "26C"
    assert events[0]["probability_total_variation"] == pytest.approx(0.2)
    assert events[0]["material_model_update"] is True
    assert events[0]["active_information_update"] is True
    assert events[0]["candidate_lead_update"] is True
    assert events[0]["qualified_candidate_update"] is True
    assert events[0]["candidate_bucket"] == "26C"
    assert events[0]["candidate_bucket_rank"] == 1
    assert events[0]["candidate_is_model_top"] is True
    assert events[0]["model_top_bucket"] == "26C"
    assert events[0]["modal_candidate_update"] is True
    assert events[0]["update_improved_final_outcome"] is True


def test_max_gap_tail_is_not_a_modal_candidate() -> None:
    rows = [
        {
            "decision_id": "before",
            "run_id": "run",
            "station_id": "KLAX",
            "local_date": "2026-07-30",
            "local_hour": 10.0,
            "received_wall_ns": 1_000_000_000,
            "decision_mode": "scheduled",
            "trigger_source": "timer",
            "v6_point_c": 25.0,
            "v6_distribution_json": json.dumps(
                {"78-79F": 0.55, "80-81F": 0.25, "82-83F": 0.20}
            ),
            "market_distribution_json": json.dumps(
                {"78-79F": 0.50, "80-81F": 0.35, "82-83F": 0.15}
            ),
            "rejection_reason": "",
        },
        {
            "decision_id": "after",
            "run_id": "run",
            "station_id": "KLAX",
            "local_date": "2026-07-30",
            "local_hour": 10.1,
            "received_wall_ns": 2_000_000_000,
            "decision_mode": "forecast_update",
            "trigger_source": "forecast_ecmwf",
            "forecast_content_changed": True,
            "v6_point_c": 25.2,
            "v6_distribution_json": json.dumps(
                {"78-79F": 0.56, "80-81F": 0.20, "82-83F": 0.24}
            ),
            "market_distribution_json": json.dumps(
                {"78-79F": 0.50, "80-81F": 0.35, "82-83F": 0.15}
            ),
            "rejection_reason": "",
        },
    ]

    event = _information_delta_rows(rows)[0]

    assert event["candidate_bucket"] == "82-83F"
    assert event["candidate_bucket_rank"] == 2
    assert event["qualified_candidate_update"] is True
    assert event["model_top_bucket"] == "78-79F"
    assert event["modal_candidate_update"] is True


def test_forecast_novelty_deduplicates_collector_restart() -> None:
    predictions = [
        {
            "decision_id": "first",
            "station_id": "EHAM",
            "decision_mode": "forecast_update",
            "trigger_source": "forecast_ecmwf",
            "received_wall_ns": 100,
        },
        {
            "decision_id": "restart",
            "station_id": "EHAM",
            "decision_mode": "forecast_update",
            "trigger_source": "forecast_ecmwf",
            "received_wall_ns": 200,
        },
    ]
    metadata = [
        {
            "decision_id": "first",
            "forecast_snapshot_kind": "revision",
            "forecast_content_sha256": "same",
            "forecast_previous_content_sha256": "old",
        },
        {
            "decision_id": "restart",
            "forecast_snapshot_kind": "initial",
            "forecast_content_sha256": "same",
            "forecast_previous_content_sha256": "",
        },
    ]

    annotated = _annotate_forecast_novelty(predictions, metadata)

    assert annotated[0]["forecast_content_changed"] is True
    assert annotated[1]["forecast_content_changed"] is False


def test_market_reaction_measures_full_vector_and_target_direction() -> None:
    events = [
        {
            "decision_id": "after",
            "station_id": "EHAM",
            "local_date": "2026-07-30",
            "material_model_update": True,
            "active_information_update": True,
            "probability_delta_json": json.dumps(
                {"25C": -0.2, "26C": 0.2}
            ),
            "market_distribution_t0_json": json.dumps(
                {"25C": 0.6, "26C": 0.4}
            ),
            "largest_probability_increase_bucket": "26C",
            "candidate_bucket": "26C",
            "weather_market_gap_t0": 0.2,
        }
    ]
    snapshots = [
        {
            "decision_id": "after",
            "horizon_seconds": 300,
            "target_timestamp_ns": 302_000_000_000,
            "snapshot_received_wall_ns": 301_000_000_000,
            "market_distribution_json": json.dumps(
                {"25C": 0.5, "26C": 0.5}
            ),
            "market_quote_coverage": 1.0,
        }
    ]

    reactions = _join_market_reactions(events, snapshots)

    assert len(reactions) == 1
    assert reactions[0]["target_bucket_market_probability_delta"] == pytest.approx(
        0.1
    )
    assert reactions[0]["market_model_delta_dot_product"] == pytest.approx(
        0.04
    )
    assert reactions[0]["market_followed_full_distribution"] is True
    assert reactions[0]["lead_confirmed"] is True
    assert reactions[0]["uncontaminated_by_later_model_update"] is True


def test_candidate_episode_allows_same_target_reinforcement() -> None:
    base = {
        "station_id": "EHAM",
        "local_date": "2026-07-30",
        "material_model_update": True,
        "winning_bucket": "26C",
        "settled": True,
    }
    events = [
        {
            **base,
            "decision_id": "start",
            "received_wall_ns": 100,
            "candidate_bucket": "26C",
            "model_top_bucket": "26C",
            "qualified_positive_gap_update_state": True,
            "modal_positive_gap_update_state": True,
            "candidate_lead_update": True,
            "qualified_candidate_update": True,
            "modal_candidate_update": True,
            "active_information_update": True,
            "weather_market_gap_t0": 0.10,
            "model_top_weather_market_gap_t0": 0.10,
            "candidate_bucket_model_probability": 0.60,
            "model_top_bucket_probability": 0.60,
        },
        {
            **base,
            "decision_id": "reinforce",
            "received_wall_ns": 200,
            "candidate_bucket": "26C",
            "model_top_bucket": "26C",
            "qualified_positive_gap_update_state": True,
            "modal_positive_gap_update_state": True,
            "candidate_lead_update": False,
            "qualified_candidate_update": False,
            "modal_candidate_update": False,
            "active_information_update": False,
            "weather_market_gap_t0": 0.12,
            "model_top_weather_market_gap_t0": 0.12,
            "candidate_bucket_model_probability": 0.62,
            "model_top_bucket_probability": 0.62,
        },
        {
            **base,
            "decision_id": "reverse",
            "received_wall_ns": 400,
            "candidate_bucket": "25C",
            "model_top_bucket": "25C",
            "qualified_positive_gap_update_state": False,
            "modal_positive_gap_update_state": False,
            "candidate_lead_update": False,
            "qualified_candidate_update": False,
            "modal_candidate_update": False,
            "active_information_update": True,
            "weather_market_gap_t0": -0.01,
            "model_top_weather_market_gap_t0": -0.01,
            "candidate_bucket_model_probability": 0.30,
            "model_top_bucket_probability": 0.30,
        },
    ]
    reactions = [
        {
            "decision_id": "start",
            "horizon_seconds": 5,
            "target_timestamp_ns": 300,
            "market_vector_valid": True,
            "model_top_bucket_market_probability_delta": 0.01,
        },
        {
            "decision_id": "start",
            "horizon_seconds": 30,
            "target_timestamp_ns": 500,
            "market_vector_valid": True,
            "model_top_bucket_market_probability_delta": 0.02,
        },
    ]

    episodes, episode_reactions = _candidate_episodes(
        events, reactions
    )

    assert len(episodes) == 1
    assert episodes[0]["confirmation_count"] == 2
    assert episodes[0]["active_confirmation_count"] == 1
    assert episodes[0]["target_won"] is True
    assert episode_reactions[0]["episode_survives_horizon"] is True
    assert episode_reactions[1]["episode_survives_horizon"] is False


def test_stream_market_snapshot_uses_last_book_before_horizon(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "2026-07-30__polymarket_book.jsonl"
    events = [
        _book_event(100, "25C", 0.60),
        _book_event(110, "26C", 0.40),
        _book_event(200, "25C", 0.55),
        _book_event(210, "26C", 0.45),
        _book_event(400, "25C", 0.50),
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    targets = [
        {
            "cache_key": ("decision", 5),
            "decision_id": "decision",
            "run_id": "run",
            "station_id": "EHAM",
            "local_date": "2026-07-30",
            "horizon_seconds": 5,
            "target_timestamp_ns": 300,
        }
    ]

    rows = _stream_run_market_snapshots(run_dir, targets)
    distribution = json.loads(rows[0]["market_distribution_json"])

    assert distribution == {"25C": 0.55, "26C": 0.45}
    assert rows[0]["snapshot_received_wall_ns"] == 210
    assert rows[0]["market_quote_coverage"] == 1.0


def _book_event(timestamp: int, label: str, mid: float) -> dict:
    return {
        "source": "polymarket_book",
        "event_type": "book_state",
        "received_wall_ns": timestamp,
        "station_id": "EHAM",
        "payload": {
            "event_date": "2026-07-30",
            "bucket_label": label,
            "mid": mid,
            "best_bid": mid - 0.01,
            "best_ask": mid + 0.01,
        },
    }
