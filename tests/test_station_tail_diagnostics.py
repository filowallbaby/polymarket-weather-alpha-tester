from __future__ import annotations

import pytest

from weather_alpha.analysis.station_tail_diagnostics import (
    _is_adjacent_upper_bucket,
    _prior_probability,
    _replay,
    _upper_tail_probability,
)


def test_prior_probability_uses_only_supplied_prior_rows() -> None:
    prior = [
        {"won": True, "raw_snapshot_probability": 0.60},
        {"won": False, "raw_snapshot_probability": 0.80},
    ]

    probability = _prior_probability(prior, current_raw=0.99)

    assert probability == (1.0 + 2.0 * 0.70) / 4.0


def test_upper_tail_probability_is_relative_to_candidate_bucket() -> None:
    distribution = {
        "70-71°F": 0.20,
        "72-73°F": 0.50,
        "74-75°F": 0.25,
        "76°F or higher": 0.05,
    }

    assert _upper_tail_probability(distribution, "72-73°F", "F") == 0.30
    assert _is_adjacent_upper_bucket("72-73°F", "74-75°F", "F") is True
    assert _is_adjacent_upper_bucket("72-73°F", "76°F or higher", "F") is False


def test_replay_selects_first_eligible_hour_and_one_trade_per_day() -> None:
    rows = [
        {
            "station_id": "KLAX",
            "local_date": "2026-07-01",
            "local_hour": 12,
            "chronological_prior_lane_signals": 20,
            "fixed_lane_probability": 0.75,
            "fixed_edge": 0.16,
            "synthetic_cost_per_share": 0.59,
            "won": True,
            "split": "development_diagnostic",
        },
        {
            "station_id": "KLAX",
            "local_date": "2026-07-01",
            "local_hour": 15,
            "chronological_prior_lane_signals": 30,
            "fixed_lane_probability": 0.95,
            "fixed_edge": 0.19,
            "synthetic_cost_per_share": 0.76,
            "won": False,
            "split": "development_diagnostic",
        },
    ]

    trades = _replay(
        rows,
        probability_field="fixed_lane_probability",
        edge_field="fixed_edge",
        minimum_edge=0.15,
        maximum_edge=0.20,
    )

    assert len(trades) == 1
    assert trades[0]["local_hour"] == 12
    assert trades[0]["synthetic_value_per_share"] == pytest.approx(0.41)
