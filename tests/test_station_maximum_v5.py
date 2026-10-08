from __future__ import annotations

import pandas as pd
import pytest

from weather_alpha.analysis.station_maximum_v5 import (
    INTERACTION_FEATURES,
    _build_feature_timelines,
    _distribution_from_residuals,
    _forecast_weight,
    _forced_entry_sensitivity,
    _match_feature_row,
    _market_label_vocabulary,
    _seasonal_residuals,
    build_physical_interaction_frame,
)
from weather_alpha.config import StationConfig


def _station(unit: str = "F") -> StationConfig:
    return StationConfig(
        key="test",
        city="Test",
        city_slug="test",
        station_id="KLAX",
        station_name="Test",
        timezone="America/Los_Angeles",
        latitude=33.94,
        longitude=-118.40,
        market_unit=unit,
        resolution_url="https://example.com",
        peer_station_ids=["KHHR"],
    )


def test_physical_interactions_align_fahrenheit_market_to_celsius() -> None:
    frame = pd.DataFrame(
        [
            {
                "temperature_c": 20.0,
                "resolution_running_max_market_unit": 69.8,
                "final_wu_max": 73.4,
                "temperature_change_1h": 1.0,
                "peer_temperature_mean_c": 21.0,
                "peer_temperature_max_c": 22.0,
                "peer_running_max_mean_c": 21.5,
                "peer_temperature_change_1h_mean": 0.5,
                "solar_clear_sky_factor": 0.8,
                "minutes_to_sunset": 360,
                "minutes_from_solar_noon": -60,
                "day_of_year_sin": 0.5,
                "day_of_year_cos": 0.5,
                "local_time_sin": -1.0,
                "local_time_cos": 0.0,
            }
        ]
    )

    result = build_physical_interaction_frame(
        frame, _station(), ncei=False
    )

    assert result.loc[0, "running_max_c"] == pytest.approx(21.0)
    assert result.loc[0, "target_c"] == pytest.approx(23.0)
    assert result.loc[0, "remaining_increment_c"] == pytest.approx(2.0)
    assert result.loc[0, "peer_mean_delta_c"] == pytest.approx(1.0)
    assert result.loc[0, "thermal_solar_interaction"] == pytest.approx(0.8)
    assert set(INTERACTION_FEATURES).issubset(result.columns)


def test_forecast_weight_is_convex_and_ridge_shrunk() -> None:
    useful = pd.DataFrame(
        {
            "base_point_c": [20.0, 20.0, 20.0],
            "forecast_point_c": [22.0, 22.0, 22.0],
            "target_c": [22.0, 22.0, 22.0],
        }
    )
    harmful = useful.assign(target_c=[19.0, 19.0, 19.0])

    useful_weight = _forecast_weight(useful)
    harmful_weight = _forecast_weight(harmful)

    assert 0.0 < useful_weight < 1.0
    assert harmful_weight == 0.0


def test_residual_distribution_cannot_put_mass_below_running_maximum() -> None:
    distribution = _distribution_from_residuals(
        center_c=20.0,
        current_max_c=21.0,
        residuals=[-10.0, 0.0, 1.0],
        labels=["68F or below", "69-70F", "71-72F", "73F or higher"],
        unit="F",
    )

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["68F or below"] < distribution["69-70F"]


def test_seasonal_residuals_match_hour_and_wrap_new_year() -> None:
    rows = [
        {"local_hour": 15.0, "day_of_year": 360.0, "residual_c": 1.0},
        {"local_hour": 15.0, "day_of_year": 100.0, "residual_c": 2.0},
        {"local_hour": 12.0, "day_of_year": 5.0, "residual_c": 3.0},
    ] * 60

    selected = _seasonal_residuals(rows, local_hour=15.0, day_of_year=5.0)

    assert selected
    assert set(selected) == {1.0}


def test_feature_join_recovers_only_sub_millisecond_csv_rounding() -> None:
    exact_timestamp = 1_785_402_016_728_828_100
    row = {"temperature_c": "27.6"}
    rows = {
        (
            "EGLC",
            "2026-07-30",
            "forecast_update",
            str(exact_timestamp),
        ): row
    }
    timelines = _build_feature_timelines(rows)

    matched = _match_feature_row(
        rows,
        timelines,
        "EGLC",
        "2026-07-30",
        "forecast_update",
        exact_timestamp + 60,
    )
    rejected = _match_feature_row(
        rows,
        timelines,
        "EGLC",
        "2026-07-30",
        "forecast_update",
        exact_timestamp + 200_000,
    )

    assert matched is row
    assert rejected == {}


def test_forced_entry_sensitivity_uses_one_signal_per_station_day() -> None:
    rows = [
        {
            "station_id": "KLGA",
            "local_date": "2026-07-01",
            "v4_estimated_edge": 0.01,
            "synthetic_value_per_share": -0.50,
            "won": False,
        },
        {
            "station_id": "KLGA",
            "local_date": "2026-07-01",
            "v4_estimated_edge": 0.03,
            "synthetic_value_per_share": 0.40,
            "won": True,
        },
        {
            "station_id": "EHAM",
            "local_date": "2026-07-01",
            "v4_estimated_edge": 0.01,
            "synthetic_value_per_share": -0.20,
            "won": False,
        },
    ]

    sensitivity = _forced_entry_sensitivity(rows)

    zero_edge = sensitivity[0]
    two_percent = sensitivity[1]
    assert zero_edge["station_day_entries"] == 2
    assert zero_edge["synthetic_value_total"] == pytest.approx(0.20)
    assert two_percent["station_day_entries"] == 1
    assert two_percent["synthetic_value_total"] == pytest.approx(0.40)


def test_market_label_vocabulary_unions_asynchronous_book_states() -> None:
    rows = [
        {
            "station_id": "KLGA",
            "local_date": "2026-07-29",
            "market_distribution_json": '{"78-79F": 0.4}',
            "market_bid_json": '{"82-83F": 0.2}',
            "weather_distribution_json": '{"80-81F": 0.5}',
        },
        {
            "station_id": "KLGA",
            "local_date": "2026-07-29",
            "market_ask_json": '{"84F or higher": 0.1}',
        },
    ]

    labels = _market_label_vocabulary(rows)

    assert labels[("KLGA", "2026-07-29")] == {
        "78-79F",
        "80-81F",
        "82-83F",
        "84F or higher",
    }
