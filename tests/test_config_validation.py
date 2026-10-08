from __future__ import annotations

import pytest
from pydantic import ValidationError

from weather_alpha.config import (
    CollectorConfig,
    ExecutionConfig,
    MarketConfig,
    StationConfig,
    load_config,
)


def test_reconnect_range_is_validated() -> None:
    with pytest.raises(ValidationError, match="reconnect_max_seconds"):
        CollectorConfig(reconnect_min_seconds=10, reconnect_max_seconds=1)


def test_execution_scenarios_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="markout_seconds"):
        ExecutionConfig(markout_seconds=[5, 5])


def test_polymarket_application_heartbeat_timeout_exceeds_interval() -> None:
    with pytest.raises(ValidationError, match="ws_application_pong_timeout_seconds"):
        MarketConfig(ws_application_ping_seconds=20, ws_application_pong_timeout_seconds=10)


def test_station_rejects_invalid_timezone() -> None:
    station = load_config(project_root=".").stations["nyc"].model_dump()
    station["timezone"] = "Mars/Olympus_Mons"
    with pytest.raises(ValidationError, match="Unknown IANA timezone"):
        StationConfig.model_validate(station)
