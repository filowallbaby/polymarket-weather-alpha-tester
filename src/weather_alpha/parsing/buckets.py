from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from typing import Literal

_EXACT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])?\s*$", re.I)
_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])?\s*$", re.I)
_BELOW = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])?\s+or\s+below\s*$", re.I)
_HIGHER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])?\s+or\s+higher\s*$", re.I)


@dataclass(frozen=True, slots=True)
class TemperatureBucket:
    label: str
    low: float | None
    high: float | None
    unit: Literal["C", "F"] | None

    def contains(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True


def parse_temperature_bucket(
    label: str, default_unit: Literal["C", "F"] | None = None
) -> TemperatureBucket:
    for regex, kind in (
        (_BELOW, "below"),
        (_HIGHER, "higher"),
        (_RANGE, "range"),
        (_EXACT, "exact"),
    ):
        match = regex.match(label)
        if not match:
            continue
        groups = match.groups()
        if kind == "range":
            low, high, unit = float(groups[0]), float(groups[1]), groups[2]
            return TemperatureBucket(label, low, high, (unit or default_unit))
        value, unit = float(groups[0]), groups[1]
        if kind == "below":
            return TemperatureBucket(label, None, value, (unit or default_unit))
        if kind == "higher":
            return TemperatureBucket(label, value, None, (unit or default_unit))
        return TemperatureBucket(label, value, value, (unit or default_unit))
    raise ValueError(f"Unrecognized temperature bucket: {label!r}")


def c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def round_half_up(value: float) -> int:

    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
