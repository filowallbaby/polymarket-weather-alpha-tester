from weather_alpha.analysis.predictive import _peer_features_at
from weather_alpha.config import load_config


def test_peer_features_are_time_safe_and_compute_upwind_weight() -> None:
    station = load_config(project_root=".").stations["nyc"]
    timestamp_ns = 10_000_000_000_000
    peer_event = {
        "received_wall_ns": timestamp_ns - 5 * 60 * 1_000_000_000,
        "payload": {
            "peer_station_id": "KJFK",
            "temperature_c": 31.0,
            "dewpoint_c": 20.0,
            "wind_direction_deg": 180.0,
            "wind_speed_kt": 10.0,
            "latitude": 40.6413,
            "longitude": -73.7781,
        },
    }
    future_event = {
        "received_wall_ns": timestamp_ns + 1,
        "payload": {
            "peer_station_id": "KEWR",
            "temperature_c": 99.0,
            "latitude": 40.6895,
            "longitude": -74.1745,
        },
    }
    features = _peer_features_at(
        {"KJFK": [peer_event], "KEWR": [future_event]},
        timestamp_ns,
        station,
        target_temperature_c=29.0,
        target_wind_direction_deg=150.0,
        target_wind_speed_kt=12.0,
        max_age_minutes=90,
    )
    assert features["peer_KJFK_temperature_c"] == 31.0
    assert features["peer_KJFK_delta_c"] == 2.0
    assert features["peer_station_count"] == 1
    assert features["peer_station_coverage"] == 0.25
    assert "peer_KEWR_temperature_c" not in features
    assert features["peer_upwind_weighted_temperature_c"] == 31.0


def test_peer_features_reuse_known_coordinates_and_measure_trend() -> None:
    station = load_config(project_root=".").stations["nyc"]
    timestamp_ns = 20_000_000_000_000
    first = {
        "station_id": "KLGA",
        "received_wall_ns": timestamp_ns - 30 * 60 * 1_000_000_000,
        "observation_timestamp_ns": timestamp_ns - 30 * 60 * 1_000_000_000,
        "payload": {
            "peer_station_id": "KJFK",
            "raw_observation": "KJFK FIRST",
            "temperature_c": 25.0,
            "latitude": 40.6413,
            "longitude": -73.7781,
        },
    }
    latest_push = {
        "station_id": "KLGA",
        "received_wall_ns": timestamp_ns - 1 * 60 * 1_000_000_000,
        "observation_timestamp_ns": timestamp_ns - 1 * 60 * 1_000_000_000,
        "payload": {
            "peer_station_id": "KJFK",
            "raw_observation": "KJFK LATEST",
            "temperature_c": 27.0,

        },
    }
    features = _peer_features_at(
        {"KJFK": [first, latest_push]},
        timestamp_ns,
        station,
        target_temperature_c=26.0,
        target_wind_direction_deg=150.0,
        target_wind_speed_kt=12.0,
        max_age_minutes=90,
    )
    assert features["peer_KJFK_distance_km"] > 0
    assert features["peer_KJFK_slope_30m_c_per_hour"] > 4.0
    assert features["peer_distance_weighted_temperature_c"] == 27.0
