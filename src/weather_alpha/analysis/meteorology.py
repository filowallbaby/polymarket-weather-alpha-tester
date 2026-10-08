from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..config import StationConfig
from ..parsing.metar import parse_raw_metar_fields


_CLOUD_FRACTION = {
    "SKC": 0.0,
    "CLR": 0.0,
    "NSC": 0.0,
    "NCD": 0.0,
    "FEW": 0.1875,
    "SCT": 0.4375,
    "BKN": 0.75,
    "OVC": 1.0,
    "VV": 1.0,
}


def station_time_features(at: datetime, station: StationConfig) -> dict[str, float]:

    if at.tzinfo is None:
        raise ValueError("station_time_features requires a timezone-aware datetime")
    local = at.astimezone(ZoneInfo(station.timezone))
    utc = at.astimezone(UTC)
    days_in_year = 366 if _is_leap_year(local.year) else 365
    day_angle = 2.0 * math.pi * (local.timetuple().tm_yday - 1) / days_in_year
    local_minutes = (
        local.hour * 60.0 + local.minute + local.second / 60.0 + local.microsecond / 60_000_000.0
    )
    local_angle = 2.0 * math.pi * local_minutes / 1440.0

    utc_hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    fractional_year = 2.0 * math.pi / 365.0 * (
        utc.timetuple().tm_yday - 1 + (utc_hour - 12.0) / 24.0
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(fractional_year)
        - 0.032077 * math.sin(fractional_year)
        - 0.014615 * math.cos(2.0 * fractional_year)
        - 0.040849 * math.sin(2.0 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(fractional_year)
        + 0.070257 * math.sin(fractional_year)
        - 0.006758 * math.cos(2.0 * fractional_year)
        + 0.000907 * math.sin(2.0 * fractional_year)
        - 0.002697 * math.cos(3.0 * fractional_year)
        + 0.00148 * math.sin(3.0 * fractional_year)
    )
    utc_offset_hours = (local.utcoffset().total_seconds() / 3600.0) if local.utcoffset() else 0.0
    true_solar_minutes = (
        local_minutes
        + equation_of_time
        + 4.0 * station.longitude
        - 60.0 * utc_offset_hours
    ) % 1440.0
    hour_angle_deg = true_solar_minutes / 4.0 - 180.0
    latitude = math.radians(station.latitude)
    hour_angle = math.radians(hour_angle_deg)
    cos_zenith = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    solar_elevation_deg = math.degrees(math.asin(cos_zenith))
    clear_sky_factor = max(0.0, cos_zenith)

    sunset_hour_angle_arg = -math.tan(latitude) * math.tan(declination)
    if sunset_hour_angle_arg <= -1.0:
        sunset_minutes = 1440.0
    elif sunset_hour_angle_arg >= 1.0:
        sunset_minutes = 0.0
    else:
        sunset_hour_angle_deg = math.degrees(math.acos(sunset_hour_angle_arg))
        solar_noon_minutes = (
            720.0
            - 4.0 * station.longitude
            - equation_of_time
            + 60.0 * utc_offset_hours
        )
        sunset_minutes = solar_noon_minutes + 4.0 * sunset_hour_angle_deg

    return {
        "local_time_sin": math.sin(local_angle),
        "local_time_cos": math.cos(local_angle),
        "day_of_year_sin": math.sin(day_angle),
        "day_of_year_cos": math.cos(day_angle),
        "solar_elevation_deg": solar_elevation_deg,
        "solar_clear_sky_factor": clear_sky_factor,
        "is_daylight": float(solar_elevation_deg > 0.0),
        "minutes_to_sunset": sunset_minutes - local_minutes,
        "minutes_from_solar_noon": (true_solar_minutes - 720.0),
    }


def surface_observation_features(payload: dict[str, Any]) -> dict[str, float]:

    payload = enrich_surface_payload(payload)
    output: dict[str, float] = {}
    temperature = _number(payload.get("temperature_c"))
    dewpoint = _number(payload.get("dewpoint_c"))
    if temperature is not None and dewpoint is not None:
        output["dewpoint_depression_c"] = temperature - dewpoint
        rh = relative_humidity_pct(temperature, dewpoint)
        output["relative_humidity_pct"] = rh
        output["vapor_pressure_deficit_hpa"] = saturation_vapor_pressure_hpa(temperature) * (
            1.0 - rh / 100.0
        )

    direction = _number(payload.get("wind_direction_deg"))
    speed = _number(payload.get("wind_speed_kt"))
    if speed is None:
        speed = _number(payload.get("wind_speed"))
    if direction is not None:
        radians = math.radians(direction % 360.0)
        output["wind_direction_sin"] = math.sin(radians)
        output["wind_direction_cos"] = math.cos(radians)
        if speed is not None:

            output["wind_u_kt"] = -speed * math.sin(radians)
            output["wind_v_kt"] = -speed * math.cos(radians)
    gust = _number(payload.get("wind_gust_kt"))
    if speed is not None and gust is not None:
        output["wind_gust_excess_kt"] = max(0.0, gust - speed)

    visibility = _number(payload.get("visibility_sm"))
    if visibility is not None:
        output["visibility_sm"] = visibility

    output.update(cloud_features(payload.get("clouds")))
    output.update(weather_features(payload.get("weather")))
    return output


def cloud_features(raw_clouds: Any) -> dict[str, float]:
    if raw_clouds is None:
        return {}
    clouds = raw_clouds if isinstance(raw_clouds, list) else [raw_clouds]
    fractions: list[float] = []
    bases: list[float] = []
    ceilings: list[float] = []
    convective = False
    for layer in clouds:
        if not isinstance(layer, dict):
            continue
        cover = str(layer.get("cover") or layer.get("amount") or "").upper()
        fraction = _CLOUD_FRACTION.get(cover)
        if fraction is not None:
            fractions.append(fraction)
        base = _number(layer.get("base"))
        if base is None:
            for field in ("base_ft_agl", "baseFeetAGL", "altitude"):
                base = _number(layer.get(field))
                if base is not None:
                    break
        if base is not None:
            bases.append(base)
            if cover in {"BKN", "OVC", "VV"}:
                ceilings.append(base)
        modifier = str(layer.get("modifier") or layer.get("type") or "").upper()
        convective = convective or modifier in {"CB", "TCU"}
    if not fractions and not bases:
        return {}
    output = {
        "cloud_layer_count": float(len(clouds)),
        "cloud_fraction": max(fractions, default=0.0),
        "convective_cloud": float(convective),
    }
    if bases:
        output["lowest_cloud_base_ft"] = min(bases)
    if ceilings:
        output["ceiling_ft"] = min(ceilings)
    return output


def weather_features(raw_weather: Any) -> dict[str, float]:
    if raw_weather in (None, ""):
        return {}
    text = str(raw_weather).upper()
    tokens = set(re.findall(r"[A-Z]+", text))
    joined = "".join(tokens)
    return {
        "weather_precipitation": float(
            any(code in joined for code in ("RA", "DZ", "SN", "PL", "GR", "GS"))
        ),
        "weather_thunderstorm": float("TS" in joined),
        "weather_fog_mist": float("FG" in joined or "BR" in joined),
        "weather_haze_smoke": float("HZ" in joined or "FU" in joined),
        "weather_showers": float("SH" in joined),
        "weather_freezing": float("FZ" in joined),
    }


def relative_humidity_pct(temperature_c: float, dewpoint_c: float) -> float:
    numerator = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = math.exp((17.625 * temperature_c) / (243.04 + temperature_c))
    return min(100.0, max(0.0, 100.0 * numerator / denominator))


def saturation_vapor_pressure_hpa(temperature_c: float) -> float:
    return 6.1094 * math.exp((17.625 * temperature_c) / (243.04 + temperature_c))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def enrich_surface_payload(payload: dict[str, Any]) -> dict[str, Any]:

    raw = str(payload.get("raw_observation") or "")
    if not raw:
        return payload
    enriched = dict(payload)
    for name, value in parse_raw_metar_fields(raw).items():
        if enriched.get(name) in (None, "", []):
            enriched[name] = value
    return enriched


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
