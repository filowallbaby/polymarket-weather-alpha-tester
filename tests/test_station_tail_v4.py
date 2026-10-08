from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from weather_alpha.analysis.station_tail_v4 import (
    FEATURE_NAMES,
    MINIMUM_LANE_DAYS,
    MINIMUM_STATION_DAYS,
    LaneOffset,
    OffsetModel,
    StationTailV4Bundle,
    _feature_record,
    _fit_bundle,
    _predict_candidate,
    _project_coefficients,
    _replay_lower_bound_strategy,
    _warmup_rejections,
    _weather_confidence,
)


def _candidate(
    *,
    station_id: str = "KLAX",
    policy_id: str = "klax_midday_convergence_v2",
    local_date: str = "2026-06-01",
    won: bool = True,
    price: float = 0.45,
) -> dict[str, object]:
    return {
        "station_id": station_id,
        "policy_id": policy_id,
        "local_date": local_date,
        "local_hour": 12,
        "received_wall_ns": 1,
        "candidate_bucket": "72-73F",
        "candidate_price_proxy": price,
        "fixed_cost_per_share": 0.02,
        "synthetic_cost_per_share": price + 0.02,
        "raw_snapshot_probability": 0.60,
        "forecast_consensus_prediction": 72.5,
        "current_running_max": 71.0,
        "forecast_spread_remaining_max_c": 1.0,
        "hours_since_running_max": 0.5,
        "temperature_change_1h": 0.3,
        "temperature_c": 21.0,
        "peer_temperature_change_1h_mean": 0.2,
        "peer_temperature_mean_c": 21.5,
        "peer_station_coverage": 1.0,
        "peer_history_point_in_time": True,
        "won": won,
        "winning_bucket": "72-73F" if won else "74-75F",
    }


def test_physical_feature_geometry_uses_celsius_for_fahrenheit_market() -> None:
    record = dict(zip(FEATURE_NAMES, _feature_record(_candidate()), strict=True))

    assert record["forecast_above_high_c"] == 0.0
    assert record["forecast_below_low_c"] == 0.0
    assert record["forecast_inside_margin_c"] > 0.0
    assert record["upper_headroom_c"] == pytest.approx((73.0 - 71.0) * 5 / 9)
    assert record["peer_temperature_delta_c"] == pytest.approx(0.5)


def test_projected_coefficients_respect_physical_signs() -> None:
    coefficients = np.ones(len(FEATURE_NAMES) + 1)
    projected = _project_coefficients(coefficients)

    assert projected[0] == 1.0
    assert projected[1] == 1.0
    assert projected[2] == 0.0
    assert projected[3] == 0.0
    assert projected[4] == 1.0


def test_forecast_spread_only_reduces_weather_correction_confidence() -> None:
    tight = _candidate()
    wide = {**tight, "forecast_spread_remaining_max_c": 8.0}

    assert _weather_confidence(tight) > _weather_confidence(wide)
    assert _weather_confidence(wide) > 0.0


def test_warmup_is_station_and_lane_specific() -> None:
    model = OffsetModel(
        coefficients=np.zeros(len(FEATURE_NAMES) + 1),
        covariance=np.eye(len(FEATURE_NAMES) + 1),
        independent_days=100,
        effective_days=100.0,
        wins=50,
        losses=50,
    )
    bundle = StationTailV4Bundle(
        schema_version=1,
        deployed_at_ns=0,
        trained_through_local_date="2026-06-01",
        training_candidates=100,
        input_sha256="x",
        feature_names=FEATURE_NAMES,
        feature_means=np.zeros(len(FEATURE_NAMES)),
        feature_scales=np.ones(len(FEATURE_NAMES)),
        global_model=model,
        station_models={"KLAX": model},
        lane_offsets={"lane": LaneOffset(0.0, 0.01, MINIMUM_LANE_DAYS - 1)},
        station_day_counts={"KLAX": MINIMUM_STATION_DAYS - 1},
        lane_day_counts={"lane": MINIMUM_LANE_DAYS - 1},
        station_outcomes={"KLAX": (20, 20)},
    )

    assert _warmup_rejections(bundle, "KLAX", "lane") == [
        "insufficient_v4_station_days",
        "insufficient_v4_lane_days",
    ]


def test_market_offset_model_is_airport_specific_and_shrunken() -> None:
    rows: list[dict[str, object]] = []
    for station, policy, favorable in (
        ("KLAX", "klax", True),
        ("EHAM", "eham", False),
    ):
        for index in range(30):
            row = _candidate(
                station_id=station,
                policy_id=policy,
                local_date=f"2026-05-{index + 1:02d}",
                won=(index % 3 != 0) if favorable else (index % 3 == 0),
                price=0.50,
            )
            rows.append(row)

    bundle = _fit_bundle(rows, 0, "test")

    assert set(bundle.station_models) == {"EHAM", "KLAX"}
    assert not np.allclose(
        bundle.station_models["EHAM"].coefficients,
        bundle.station_models["KLAX"].coefficients,
    )
    assert all(
        math.isfinite(float(value))
        for model in bundle.station_models.values()
        for value in model.coefficients
    )


def test_trade_replay_uses_first_positive_lower_bound_not_edge_band() -> None:
    first = {
        **_candidate(),
        "v4_estimated_edge": 0.03,
        "v4_lower_bound_edge": 0.001,
        "v4_candidate_probability": 0.50,
        "v4_probability_lower_bound": 0.471,
        "synthetic_value_per_share": 0.53,
    }
    later = {
        **first,
        "received_wall_ns": 2,
        "v4_estimated_edge": 0.30,
        "v4_lower_bound_edge": 0.20,
    }

    trades = _replay_lower_bound_strategy([later, first])

    assert len(trades) == 1
    assert trades[0]["received_wall_ns"] == 1


def test_probability_lower_bound_never_exceeds_point_estimate() -> None:
    rows = []
    first_day = date(2026, 4, 1)
    for index in range(60):
        rows.append(
            _candidate(
                local_date=(first_day + timedelta(days=index)).isoformat(),
                won=index % 2 == 0,
            )
        )
    bundle = _fit_bundle(rows, 0, "test")
    prediction = _predict_candidate(bundle, rows[-1])

    assert prediction["probability_lower_bound"] <= prediction[
        "candidate_probability"
    ]
