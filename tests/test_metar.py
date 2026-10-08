from datetime import UTC, datetime

import pytest

from weather_alpha.parsing.metar import (
    parse_awc_metar,
    parse_raw_metar_fields,
    parse_raw_temperature,
)
from weather_alpha.sources.nws_tgftp import (
    infer_metar_observation_time,
    parse_tgftp_station_text,
)


def test_exact_t_group_wins_over_integer_body() -> None:
    raw = "KLGA 161651Z 21008KT 10SM FEW040 29/21 A2992 RMK AO2 T02940211"
    temps = parse_raw_temperature(raw)
    assert temps["temperature_c_exact"] == 29.4
    assert temps["temperature_c_reported"] == 29.0
    normalized, errors = parse_awc_metar(
        {"icaoId": "KLGA", "rawOb": raw, "reportTime": "2026-07-16T16:51:00Z"}
    )
    assert not errors
    assert normalized["temperature_c"] == 29.4
    assert normalized["dewpoint_c"] == 21.1


def test_negative_temperature() -> None:
    raw = "EGLC 161650Z 02005KT 9999 SCT020 M03/M07 Q1012 RMK T10331072"
    temps = parse_raw_temperature(raw)
    assert temps["temperature_c_exact"] == -3.3
    assert temps["dewpoint_c_exact"] == -7.2
    assert temps["temperature_c_reported"] == -3.0


def test_raw_metar_decodes_fast_surface_context() -> None:
    raw = (
        "EHAM 212025Z 25012G22KT 4000 -RA BR FEW011 SCT018 BKN036 16/15 Q1008 "
        "TEMPO 2500"
    )
    fields = parse_raw_metar_fields(raw)
    assert fields["wind_direction_deg"] == 250.0
    assert fields["wind_speed_kt"] == 12.0
    assert fields["wind_gust_kt"] == 22.0
    assert fields["altimeter_hpa"] == 1008.0
    assert fields["visibility_sm"] == pytest.approx(2.4855, rel=1e-4)
    assert fields["weather"] == "-RA BR"
    assert fields["clouds"][0] == {"cover": "FEW", "base": 1100, "modifier": None}
    assert fields["clouds"][2]["base"] == 3600


def test_raw_us_visibility_and_altimeter_are_normalized() -> None:
    raw = "KLGA 211651Z 18012KT 1 1/2SM BR OVC008 25/24 A2992"
    fields = parse_raw_metar_fields(raw)
    assert fields["visibility_sm"] == 1.5
    assert fields["altimeter_hpa"] == pytest.approx(1013.2075, rel=1e-4)
    assert fields["clouds"][0]["base"] == 800


def test_tgftp_station_file_and_month_rollover() -> None:
    text = "2026/08/01 00:02\nKLAX 312353Z 25008KT 10SM CLR 22/16 A2998\n"
    receipt, raw, errors = parse_tgftp_station_text(text, "KLAX")
    assert not errors
    assert receipt == datetime(2026, 8, 1, 0, 2, tzinfo=UTC)
    assert raw and raw.startswith("KLAX")
    inferred = infer_metar_observation_time(raw, receipt)
    assert inferred == datetime(2026, 7, 31, 23, 53, tzinfo=UTC)
