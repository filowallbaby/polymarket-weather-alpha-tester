from __future__ import annotations

import json

import pytest

from weather_alpha.analysis.station_tail_paper_v3 import (
    _model_record,
    _outcome_class,
    _recalibrated_distribution,
)


def _candidate() -> dict[str, object]:
    return {
        "station_id": "KLAX",
        "policy_id": "klax_midday_convergence_v2",
        "local_hour": 12,
        "candidate_bucket": "72-73°F",
        "winning_bucket": "74-75°F",
        "candidate_price_proxy": 0.50,
        "raw_snapshot_probability": 0.60,
        "chronological_lane_probability": 0.70,
        "market_minus_weather_upper_tail": 0.05,
        "forecast_minus_candidate_high": 2.0,
        "forecast_spread_remaining_max_c": 1.5,
        "candidate_high_minus_current_max": 2.0,
        "raw_probability_rise_again": 0.65,
        "hours_since_running_max": 0.5,
        "peer_temperature_change_1h_mean": 0.4,
        "weather_distribution": {
            "70-71°F": 0.10,
            "72-73°F": 0.60,
            "74-75°F": 0.25,
            "76°F or higher": 0.05,
        },
        "market_distribution": {
            "70-71°F": 0.15,
            "72-73°F": 0.50,
            "74-75°F": 0.25,
            "76°F or higher": 0.10,
        },
        "won": False,
    }


def test_model_record_contains_regularized_snapshot_context() -> None:
    record = _model_record(_candidate())

    assert record["market_logit"] == pytest.approx(0.0)
    assert record["weather_logit"] > 0.0
    assert record["lane_logit"] > record["weather_logit"]
    assert record["station_id"] == "KLAX"
    assert record["local_hour_category"] == "12"


def test_recalibrated_distribution_is_coherent() -> None:
    distribution = _recalibrated_distribution(
        _candidate(),
        candidate_probability=0.65,
        upper_probability=0.25,
        lower_probability=0.10,
        unit="F",
    )

    assert distribution["72-73°F"] == pytest.approx(0.65)
    assert distribution["70-71°F"] == pytest.approx(0.10)
    assert (
        distribution["74-75°F"] + distribution["76°F or higher"]
    ) == pytest.approx(0.25)
    assert sum(distribution.values()) == pytest.approx(1.0)
    json.dumps(distribution)


def test_outcome_class_detects_adjacent_upper_miss() -> None:
    assert _outcome_class(_candidate()) == "upper"
