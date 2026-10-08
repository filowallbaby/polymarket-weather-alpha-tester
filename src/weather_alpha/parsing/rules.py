from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from dateutil import parser as date_parser

_STATION_FROM_WU = re.compile(
    r"wunderground\.com/history/daily/[^\s)\]]*/(?P<icao>[A-Z0-9]{4})(?:\b|[/?#])", re.I
)
_STATION_TEXT = re.compile(
    r"recorded at the (?P<name>.+? Airport(?: Station)?|.+? Station) in degrees", re.I
)
_UNIT = re.compile(r"degrees (?P<unit>Celsius|Fahrenheit)", re.I)
_DATE_DMY = re.compile(
    r"on\s+(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})\s+['’]?(?P<year>\d{2,4})",
    re.I,
)
_DATE_MDY = re.compile(
    r"on\s+(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ParsedRules:
    station_id: str | None
    station_name: str | None
    unit: Literal["C", "F"] | None
    event_date: date | None
    resolution_url: str | None


def parse_weather_market_rules(text: str) -> ParsedRules:
    station_match = _STATION_FROM_WU.search(text)
    name_match = _STATION_TEXT.search(text)
    unit_match = _UNIT.search(text)
    date_match = _DATE_DMY.search(text) or _DATE_MDY.search(text)

    unit = None
    if unit_match:
        unit = "C" if unit_match.group("unit").lower().startswith("c") else "F"

    event_date = None
    if date_match:
        raw = f"{date_match.group('day')} {date_match.group('month')} {date_match.group('year')}"
        parsed = date_parser.parse(raw, dayfirst=True)
        event_date = parsed.date()

    resolution_url = station_match.group(0) if station_match else None
    return ParsedRules(
        station_id=station_match.group("icao").upper() if station_match else None,
        station_name=name_match.group("name").strip() if name_match else None,
        unit=unit,
        event_date=event_date,
        resolution_url=resolution_url,
    )


def validate_rules(
    parsed: ParsedRules,
    *,
    expected_station_id: str,
    expected_unit: str,
    reject_station_mismatch: bool,
    reject_unit_mismatch: bool,
) -> list[str]:
    errors: list[str] = []
    if parsed.station_id is None:
        errors.append("rules_missing_station_id")
        if reject_station_mismatch:
            errors.append("fatal_missing_station_id")
    elif parsed.station_id != expected_station_id:
        errors.append(f"station_mismatch:{parsed.station_id}!={expected_station_id}")
        if reject_station_mismatch:
            errors.append("fatal_station_mismatch")

    if parsed.unit is None:
        errors.append("rules_missing_unit")
        if reject_unit_mismatch:
            errors.append("fatal_missing_unit")
    elif parsed.unit != expected_unit:
        errors.append(f"unit_mismatch:{parsed.unit}!={expected_unit}")
        if reject_unit_mismatch:
            errors.append("fatal_unit_mismatch")
    return errors
