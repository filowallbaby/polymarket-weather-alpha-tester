from weather_alpha.parsing.buckets import parse_temperature_bucket, round_half_up
from weather_alpha.parsing.rules import parse_weather_market_rules, validate_rules


def test_realistic_rules_parse() -> None:
    text = (
        "This market will resolve to the range containing the highest temperature recorded "
        "at the Amsterdam Airport Schiphol in degrees Celsius on 16 Jul '26. "
        "The resolution source is https://www.wunderground.com/history/daily/nl/schiphol/EHAM"
    )
    parsed = parse_weather_market_rules(text)
    assert parsed.station_id == "EHAM"
    assert parsed.unit == "C"
    assert parsed.event_date.isoformat() == "2026-07-16"
    assert not validate_rules(
        parsed,
        expected_station_id="EHAM",
        expected_unit="C",
        reject_station_mismatch=True,
        reject_unit_mismatch=True,
    )


def test_fatal_rule_mismatch() -> None:
    parsed = parse_weather_market_rules(
        "recorded at the LaGuardia Airport in degrees Fahrenheit. "
        "https://www.wunderground.com/history/daily/us/ny/new-york-city/KJFK"
    )
    errors = validate_rules(
        parsed,
        expected_station_id="KLGA",
        expected_unit="F",
        reject_station_mismatch=True,
        reject_unit_mismatch=True,
    )
    assert "fatal_station_mismatch" in errors


def test_bucket_forms_and_rounding() -> None:
    assert parse_temperature_bucket("88-89°F", "F").contains(89)
    assert parse_temperature_bucket("87°F or below", "F").contains(-100)
    assert parse_temperature_bucket("90°F or higher", "F").contains(100)
    assert parse_temperature_bucket("24°C", "C").contains(24)
    assert round_half_up(28.5) == 29
    assert round_half_up(-2.5) == -3
