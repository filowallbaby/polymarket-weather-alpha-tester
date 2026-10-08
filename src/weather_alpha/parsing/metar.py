from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from dateutil import parser as date_parser

_T_GROUP = re.compile(r"(?:^|\s)T(?P<tsign>[01])(?P<t>\d{3})(?P<dsign>[01])(?P<d>\d{3})(?:\s|$)")
_MAIN_TEMP = re.compile(r"(?:^|\s)(?P<t>M?\d{2})/(?P<d>M?\d{2})(?:\s|$)")
_WIND = re.compile(
    r"(?:^|\s)(?P<direction>\d{3}|VRB)(?P<speed>\d{2,3})"
    r"(?:G(?P<gust>\d{2,3}))?KT(?:\s|$)"
)
_VISIBILITY_SM = re.compile(
    r"(?:^|\s)(?P<prefix>[MP])?(?:"
    r"(?P<whole>\d{1,2})(?:\s+(?P<numerator>\d)/(?P<denominator>\d))?"
    r"|(?P<fraction_numerator>\d)/(?P<fraction_denominator>\d)"
    r")SM(?:\s|$)"
)
_VISIBILITY_METERS = re.compile(r"^\d{4}$")
_ALTIMETER_INHG = re.compile(r"(?:^|\s)A(?P<value>\d{4})(?:\s|$)")
_PRESSURE_HPA = re.compile(r"(?:^|\s)Q(?P<value>\d{4})(?:\s|$)")
_CLOUD_LAYER = re.compile(
    r"^(?P<cover>FEW|SCT|BKN|OVC|VV)(?P<base>\d{3}|///)(?P<modifier>CB|TCU)?$"
)
_CLEAR_SKY = re.compile(r"^(?P<cover>SKC|CLR|NSC|NCD)$")
_PRESENT_WEATHER = re.compile(
    r"^(?P<weather>[+-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?"
    r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS){1,3})$"
)


def _signed_tenth(sign: str, digits: str) -> float:
    value = int(digits) / 10.0
    return -value if sign == "1" else value


def parse_raw_temperature(raw: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "temperature_c_exact": None,
        "dewpoint_c_exact": None,
        "temperature_c_reported": None,
        "dewpoint_c_reported": None,
    }
    match = _T_GROUP.search(raw)
    if match:
        result["temperature_c_exact"] = _signed_tenth(match.group("tsign"), match.group("t"))
        result["dewpoint_c_exact"] = _signed_tenth(match.group("dsign"), match.group("d"))

    main = _MAIN_TEMP.search(raw)
    if main:

        def decode(value: str) -> float:
            return float(-int(value[1:]) if value.startswith("M") else int(value))

        result["temperature_c_reported"] = decode(main.group("t"))
        result["dewpoint_c_reported"] = decode(main.group("d"))
    return result


def parse_raw_metar_fields(raw: str) -> dict[str, Any]:

    wind_match = _WIND.search(raw)
    direction = None
    speed = None
    gust = None
    if wind_match:
        raw_direction = wind_match.group("direction")
        direction = None if raw_direction == "VRB" else float(raw_direction)
        speed = float(wind_match.group("speed"))
        gust = float(wind_match.group("gust")) if wind_match.group("gust") else None

    visibility = None
    visibility_match = _VISIBILITY_SM.search(raw)
    if visibility_match:
        if visibility_match.group("whole") is not None:
            visibility = float(visibility_match.group("whole"))
            if visibility_match.group("numerator"):
                visibility += float(visibility_match.group("numerator")) / float(
                    visibility_match.group("denominator")
                )
        else:
            visibility = float(visibility_match.group("fraction_numerator")) / float(
                visibility_match.group("fraction_denominator")
            )
        if visibility_match.group("prefix") == "M":

            visibility *= 0.99
    elif re.search(r"(?:^|\s)CAVOK(?:\s|$)", raw):
        visibility = 10.0 / 1.609344
    else:
        metric_visibility = next(
            (token for token in raw.split() if _VISIBILITY_METERS.fullmatch(token)), None
        )
        if metric_visibility is not None:
            meters = min(10_000.0, float(metric_visibility))
            visibility = (meters / 1000.0) / 1.609344

    altimeter_hpa = None
    altimeter_match = _ALTIMETER_INHG.search(raw)
    pressure_match = _PRESSURE_HPA.search(raw)
    if altimeter_match:
        altimeter_hpa = (float(altimeter_match.group("value")) / 100.0) * 33.8638866667
    elif pressure_match:
        altimeter_hpa = float(pressure_match.group("value"))

    clouds: list[dict[str, Any]] = []
    for token in raw.split():
        match = _CLOUD_LAYER.fullmatch(token)
        if match is None:
            continue
        raw_base = match.group("base")
        clouds.append(
            {
                "cover": match.group("cover"),
                "base": None if raw_base == "///" else int(raw_base) * 100,
                "modifier": match.group("modifier"),
            }
        )
    clear_match = next(
        (match for token in raw.split() if (match := _CLEAR_SKY.fullmatch(token)) is not None),
        None,
    )
    if not clouds and clear_match:
        clouds.append({"cover": clear_match.group("cover"), "base": None, "modifier": None})

    weather_tokens = [
        match.group("weather")
        for token in raw.split()
        if (match := _PRESENT_WEATHER.fullmatch(token)) is not None
    ]
    return {
        "wind_direction_deg": direction,
        "wind_speed_kt": speed,
        "wind_gust_kt": gust,
        "visibility_sm": visibility,
        "altimeter_hpa": altimeter_hpa,
        "weather": " ".join(weather_tokens) or None,
        "clouds": clouds or None,
    }


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):

        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    try:
        parsed = date_parser.isoparse(str(value))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_awc_metar(item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    raw = str(_first(item, "rawOb", "raw_text", "rawText", "raw") or "")
    station_id = str(_first(item, "icaoId", "stationId", "station_id", "id") or "")
    raw_temps = parse_raw_temperature(raw)
    raw_fields = parse_raw_metar_fields(raw)

    observation_time = _parse_time(
        _first(item, "reportTime", "obsTime", "observation_time", "observationTime")
    )
    if not station_id:
        errors.append("missing_station_id")
    if observation_time is None:
        errors.append("missing_or_invalid_observation_time")
    if not raw:
        errors.append("missing_raw_observation")

    temp_decoded = _float(_first(item, "temp", "tempC", "temperature", "temperature_c"))
    dew_decoded = _float(_first(item, "dewp", "dewpoint", "dewpointC", "dewpoint_c"))
    temperature_c = raw_temps["temperature_c_exact"]
    if temperature_c is None:
        temperature_c = temp_decoded
    if temperature_c is None:
        temperature_c = raw_temps["temperature_c_reported"]

    dewpoint_c = raw_temps["dewpoint_c_exact"]
    if dewpoint_c is None:
        dewpoint_c = dew_decoded
    if dewpoint_c is None:
        dewpoint_c = raw_temps["dewpoint_c_reported"]

    clouds = _first(item, "clouds", "sky_condition", "skyConditions") or raw_fields["clouds"]
    normalized = {
        "station_id": station_id,
        "observation_time": observation_time.isoformat() if observation_time else None,
        "raw_observation": raw,
        "report_type": _first(item, "metarType", "reportType", "type"),
        "temperature_c": temperature_c,
        "temperature_c_exact": raw_temps["temperature_c_exact"],
        "temperature_c_reported": raw_temps["temperature_c_reported"],
        "dewpoint_c": dewpoint_c,
        "dewpoint_c_exact": raw_temps["dewpoint_c_exact"],
        "dewpoint_c_reported": raw_temps["dewpoint_c_reported"],
        "wind_direction_deg": _float(_first(item, "wdir", "windDir", "wind_direction"))
        if _first(item, "wdir", "windDir", "wind_direction") is not None
        else raw_fields["wind_direction_deg"],
        "wind_speed_kt": _float(_first(item, "wspd", "windSpeed", "wind_speed"))
        if _first(item, "wspd", "windSpeed", "wind_speed") is not None
        else raw_fields["wind_speed_kt"],
        "wind_gust_kt": _float(_first(item, "wgst", "windGust", "wind_gust"))
        if _first(item, "wgst", "windGust", "wind_gust") is not None
        else raw_fields["wind_gust_kt"],
        "visibility_sm": _float(
            _first(item, "visib", "visibility", "visibility_statute_mi")
        )
        if _first(item, "visib", "visibility", "visibility_statute_mi") is not None
        else raw_fields["visibility_sm"],
        "altimeter_hpa": _float(_first(item, "altim", "altimeter", "altimeter_hpa"))
        if _first(item, "altim", "altimeter", "altimeter_hpa") is not None
        else raw_fields["altimeter_hpa"],
        "sea_level_pressure_hpa": _float(_first(item, "slp", "seaLevelPressure")),
        "weather": _first(item, "wxString", "weather", "presentWeather")
        or raw_fields["weather"],
        "clouds": clouds,
        "latitude": _float(_first(item, "lat", "latitude")),
        "longitude": _float(_first(item, "lon", "longitude")),
        "elevation_m": _float(_first(item, "elev", "elevation", "elevation_m")),
        "source_fields": item,
    }
    return normalized, errors
