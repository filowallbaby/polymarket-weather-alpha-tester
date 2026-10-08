from datetime import UTC, datetime

import pytest

from weather_alpha.analysis.meteorology import (
    station_time_features,
    surface_observation_features,
)
from weather_alpha.config import load_config


def test_station_time_features_follow_local_solar_cycle() -> None:
    station = load_config(project_root=".").stations["los_angeles"]
    midday = station_time_features(datetime(2026, 7, 21, 20, 0, tzinfo=UTC), station)
    midnight = station_time_features(datetime(2026, 7, 21, 7, 0, tzinfo=UTC), station)
    assert midday["solar_elevation_deg"] > 60
    assert midday["is_daylight"] == 1.0
    assert midnight["solar_elevation_deg"] < 0
    assert midnight["is_daylight"] == 0.0
    assert -1 <= midday["day_of_year_sin"] <= 1
    assert -1 <= midday["local_time_cos"] <= 1


def test_surface_features_include_humidity_wind_cloud_and_weather() -> None:
    features = surface_observation_features(
        {
            "temperature_c": 30.0,
            "dewpoint_c": 20.0,
            "wind_direction_deg": 270.0,
            "wind_speed_kt": 10.0,
            "wind_gust_kt": 18.0,
            "visibility_sm": 4.0,
            "weather": "-RA BR",
            "clouds": [
                {"cover": "SCT", "base": 1500},
                {"cover": "BKN", "base": 3000},
            ],
        }
    )
    assert features["dewpoint_depression_c"] == 10.0
    assert features["relative_humidity_pct"] == pytest.approx(55.1, rel=0.02)
    assert features["wind_u_kt"] == pytest.approx(10.0)
    assert features["wind_v_kt"] == pytest.approx(0.0, abs=1e-10)
    assert features["wind_gust_excess_kt"] == 8.0
    assert features["cloud_fraction"] == 0.75
    assert features["ceiling_ft"] == 3000
    assert features["weather_precipitation"] == 1.0
    assert features["weather_fog_mist"] == 1.0
