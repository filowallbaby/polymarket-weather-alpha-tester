from datetime import UTC, date, datetime

from weather_alpha.config import load_config
from weather_alpha.analysis.predictive import _running_wu_max_at
from weather_alpha.sources.wunderground import WundergroundCollector


def _station():
    return load_config(project_root=".").stations["amsterdam"]


def test_dated_url_replaces_existing_date() -> None:
    assert WundergroundCollector._dated_url(
        "https://www.wunderground.com/history/daily/nl/schiphol/EHAM/date/2026-7-1",
        date(2026, 7, 16),
    ).endswith("/EHAM/date/2026-7-16")


def test_extracts_row_oriented_history_payload() -> None:
    payload = {
        "observations": [
            {"valid_time_gmt": 1784217600, "temp": 21.0, "icaoCode": "EHAM"},
            {"valid_time_gmt": 1784218200, "temp": 22.0, "icaoCode": "EHAM"},
        ]
    }
    observations = WundergroundCollector._extract_observations(
        payload,
        "https://api.weather.com/v1/location/EHAM:9:NL/observations/historical.json?units=m",
        _station(),
    )
    assert [row["temperature"] for row in observations] == [21.0, 22.0]
    assert all(row["unit"] == "C" for row in observations)


def test_row_oriented_payload_uses_wu_obs_id_for_station_validation() -> None:
    payload = {
        "metadata": {"units": "e"},
        "observations": [{"valid_time_gmt": 1784217600, "temp": 70.0, "obs_id": "KLGA"}],
    }
    station = load_config(project_root=".").stations["nyc"]
    observations = WundergroundCollector._extract_observations(
        payload,
        "https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?units=e",
        station,
    )
    assert observations[0]["station_id"] == "KLGA"


def test_extracts_parallel_arrays_with_explicit_units() -> None:
    payload = {
        "validTimeUtc": [1784217600, 1784218200],
        "temperature": {"metric": [21.5, 22.25], "imperial": [70.7, 72.05]},
        "stationId": "EHAM",
    }
    observations = WundergroundCollector._extract_observations(
        payload,
        "https://api.weather.com/v3/wx/observations/historical?icaoCode=EHAM",
        _station(),
    )
    assert [row["temperature"] for row in observations] == [21.5, 22.25]
    assert all(row["unit"] == "C" for row in observations)


def test_rejects_forecast_payloads() -> None:
    payload = {"validTimeUtc": [1784217600], "temperature": [22]}
    observations = WundergroundCollector._extract_observations(
        payload,
        "https://api.weather.com/v3/wx/forecast/hourly/2day?icaoCode=EHAM",
        _station(),
    )
    assert observations == []


def test_summary_filters_station_and_requires_next_day_observation() -> None:
    target = date(2026, 7, 16)

    def stamp(day: int, hour: int) -> int:
        return int(datetime(2026, 7, day, hour, tzinfo=UTC).timestamp() * 1e9)

    network = [
        {
            "observations": [
                {
                    "timestamp_ns": stamp(16, 12),
                    "temperature": 22.0,
                    "unit": "C",
                    "station_id": "EHAM",
                },
                {
                    "timestamp_ns": stamp(16, 13),
                    "temperature": 99.0,
                    "unit": "C",
                    "station_id": "EHRD",
                },
                {
                    "timestamp_ns": stamp(17, 0),
                    "temperature": 18.0,
                    "unit": "C",
                    "station_id": "EHAM",
                },
            ]
        }
    ]
    summary = WundergroundCollector._summarize(_station(), network, "", target)
    assert summary["daily_max_from_network"] == 22.0
    assert summary["station_mismatch_count"] == 1
    assert summary["resolution_final"] is True
    assert summary["finalization_observation_timestamp_ns"] == stamp(17, 0)


def test_summary_can_use_next_day_evidence_from_an_earlier_page_cycle() -> None:
    target = date(2026, 7, 16)
    target_ns = int(datetime(2026, 7, 16, 12, tzinfo=UTC).timestamp() * 1e9)
    next_day_ns = int(datetime(2026, 7, 17, 0, tzinfo=UTC).timestamp() * 1e9)
    network = [
        {
            "observations": [
                {
                    "timestamp_ns": target_ns,
                    "temperature": 22.0,
                    "unit": "C",
                    "station_id": "EHAM",
                }
            ]
        }
    ]
    summary = WundergroundCollector._summarize(
        _station(),
        network,
        "",
        target,
        first_observation_by_local_date={date(2026, 7, 17): next_day_ns},
    )
    assert summary["resolution_final"] is True
    assert summary["finalization_observation_timestamp_ns"] == next_day_ns


def test_plain_temperature_without_verifiable_units_is_rejected() -> None:
    payload = {"validTimeUtc": [1784217600], "temperature": [22], "stationId": "EHAM"}
    observations = WundergroundCollector._extract_observations(
        payload,
        "https://api.weather.com/v3/wx/observations/historical?icaoCode=EHAM",
        _station(),
    )
    assert observations == []


def test_summary_recovers_explicit_visible_day_high_without_network_history() -> None:
    body = """Weather History
Day High & Low
HIGH
24°C
at 2:53 PM
LOW
18°C
"""

    summary = WundergroundCollector._summarize(
        _station(), [], body, date(2026, 8, 10)
    )

    assert summary["daily_max_from_network"] == 24.0
    assert summary["unit_verified"] is True
    assert summary["resolution_final"] is False


def test_visible_day_high_converts_explicit_celsius_to_market_fahrenheit() -> None:
    station = load_config(project_root=".").stations["los_angeles"]
    body = "Day High & Low\nHIGH\n24°C\nat 9:53 AM\nLOW\n21°C"

    summary = WundergroundCollector._summarize(
        station, [], body, date(2026, 8, 10)
    )

    assert summary["daily_max_from_network"] == 75.2
    assert summary["unit_verified"] is True


def test_running_wu_max_recovers_captured_redesign_page_point_in_time() -> None:
    event = {
        "received_wall_ns": 100,
        "payload": {
            "daily_max_from_network": None,
            "market_unit": "F",
            "body_text_sample": "Day High & Low\nHIGH\n24°C\nat 9:53 AM\nLOW\n21°C",
        },
    }

    assert _running_wu_max_at([event], 99) is None
    assert _running_wu_max_at([event], 100) == 75.2
