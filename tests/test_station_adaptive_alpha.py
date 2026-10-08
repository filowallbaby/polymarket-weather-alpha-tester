from __future__ import annotations

from weather_alpha.analysis.station_adaptive_alpha import (
    STATION_ADAPTIVE_LANES,
    _edge_band_rejections,
    _summarize_lane,
    _weather_centered_locked_probability,
)
from weather_alpha.analysis.station_specific_alpha import (
    evaluate_station_policy_snapshot,
)


def _lane(policy_id: str) -> dict[str, object]:
    for lanes in STATION_ADAPTIVE_LANES.values():
        for lane in lanes:
            if lane["policy_id"] == policy_id:
                return lane
    raise AssertionError(f"missing lane {policy_id}")


def test_all_airports_remain_enabled_in_adaptive_configuration() -> None:
    assert set(STATION_ADAPTIVE_LANES) == {
        "EHAM",
        "EGLC",
        "KLAX",
        "KLGA",
    }
    assert all(STATION_ADAPTIVE_LANES.values())


def test_lax_has_midday_entry_and_late_fallback() -> None:
    midday = _lane("klax_midday_convergence_v2")
    fallback = _lane("klax_late_certainty_fallback_v2")

    assert midday["target_local_hour"] == 12
    assert midday["minimum_weather_top_probability"] == 0.60
    assert fallback["target_local_hour"] == 15
    assert fallback["minimum_weather_top_probability"] == 0.80


def test_amsterdam_agreement_and_disagreement_use_separate_lanes() -> None:
    agreement = _lane("eham_late_convergence_v2")
    disagreement = _lane("eham_late_disagreement_v2")
    weather = {"20Â°C": 0.70, "21Â°C": 0.30}

    agreed = evaluate_station_policy_snapshot(
        agreement,
        weather,
        {"20Â°C": 0.65, "21Â°C": 0.35},
    )
    disagreed = evaluate_station_policy_snapshot(
        disagreement,
        weather,
        {"20Â°C": 0.20, "21Â°C": 0.80},
    )

    assert agreed["eligible"] is True
    assert disagreed["eligible"] is True
    assert agreed["candidate_bucket"] == "20Â°C"
    assert disagreed["candidate_bucket"] == "20Â°C"


def test_locked_replay_applies_same_positive_edge_rule_as_paper() -> None:
    rows = [
        {
            "won": True,
            "candidate_price_proxy": 0.60,
            "candidate_weather_probability": 0.70,
            "fixed_cost_per_share": 0.05,
            "synthetic_value_per_share": 0.35,
        },
        {
            "won": True,
            "candidate_price_proxy": 0.80,
            "candidate_weather_probability": 0.90,
            "fixed_cost_per_share": 0.05,
            "synthetic_value_per_share": 0.15,
        },
    ]

    summary = _summarize_lane(rows, locked_probability=0.70)

    assert summary["signals"] == 2
    assert summary["locked_edge_replay_signals"] == 1
    assert summary["locked_edge_replay_total_value"] == 0.35


def test_probability_prior_preserves_station_weather_direction() -> None:
    lax_like = [
        {
            "won": index < 9,
            "candidate_weather_probability": 0.90,
        }
        for index in range(10)
    ]
    amsterdam_disagreement_like = [
        {
            "won": index < 3,
            "candidate_weather_probability": 0.65,
        }
        for index in range(10)
    ]

    lax_probability = _weather_centered_locked_probability(lax_like)
    amsterdam_probability = _weather_centered_locked_probability(
        amsterdam_disagreement_like
    )

    assert lax_probability > 0.80
    assert amsterdam_probability < 0.40


def test_edge_band_is_lower_inclusive_and_upper_exclusive() -> None:
    deployment = {
        "minimum_edge_inclusive": 0.15,
        "maximum_edge_exclusive": 0.20,
    }

    assert _edge_band_rejections(0.15, deployment) == []
    assert _edge_band_rejections(0.1999, deployment) == []
    assert _edge_band_rejections(0.1499, deployment) == [
        "edge_below_deployed_band"
    ]
    assert _edge_band_rejections(0.20, deployment) == [
        "edge_above_deployed_band"
    ]
