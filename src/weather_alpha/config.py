from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceFlags(BaseModel):
    wis2: bool = False
    awc: bool = True
    nws_tgftp: bool = True
    knmi: bool = False
    wunderground: bool = True
    forecasts: bool = True


class StationConfig(BaseModel):
    key: str = ""
    city: str
    city_slug: str
    station_id: str = Field(pattern=r"^[A-Z0-9]{4}$")
    station_name: str
    timezone: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation_m: float | None = None
    market_unit: Literal["C", "F"]
    expected_precision: float = Field(default=1.0, gt=0)
    resolution_url: str
    sources: SourceFlags = Field(default_factory=SourceFlags)
    knmi_station_code: int | None = None
    forecast_providers: list[str] = Field(default_factory=list)
    peer_station_ids: list[str] = Field(default_factory=list)


    polymarket_series_id: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_station(self) -> "StationConfig":
        if self.sources.knmi and self.knmi_station_code is None:
            raise ValueError("knmi_station_code is required when KNMI source is enabled")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {self.timezone}") from exc
        if not self.resolution_url.startswith("https://"):
            raise ValueError("resolution_url must use HTTPS")
        peers = [peer.upper() for peer in self.peer_station_ids]
        if len(peers) != len(set(peers)) or self.station_id in peers:
            raise ValueError("peer_station_ids must be unique and exclude the primary station")
        if any(len(peer) != 4 or not peer.isalnum() for peer in peers):
            raise ValueError("peer_station_ids must contain four-character station IDs")
        if len(self.forecast_providers) != len(set(self.forecast_providers)):
            raise ValueError("forecast_providers must be unique")
        return self


class CollectorConfig(BaseModel):
    awc_poll_seconds: float = Field(default=60, gt=0)


    nws_tgftp_poll_seconds: float = Field(default=300, ge=60)
    knmi_poll_seconds: float = Field(default=900, gt=0)
    knmi_download_timeout_seconds: float = Field(default=120, gt=0)
    wunderground_poll_seconds: float = Field(default=300, ge=60)
    wunderground_browser_recycle_cycles: int = Field(default=24, ge=2, le=288)
    forecast_poll_seconds: float = Field(default=300, ge=60)
    forecast_concurrency: int = Field(default=2, ge=1, le=16)
    forecast_retry_attempts: int = Field(default=3, ge=1, le=10)
    forecast_retry_base_seconds: float = Field(default=2, gt=0)
    forecast_retry_max_seconds: float = Field(default=30, gt=0)
    market_discovery_seconds: float = Field(default=300, gt=0)
    heartbeat_seconds: float = Field(default=60, gt=0)
    minimum_free_disk_gb: float = Field(default=10.0, ge=1.0, le=1000.0)


    maximum_process_private_memory_mb: int = Field(default=4096, ge=256, le=262_144)
    resource_check_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    maximum_run_seconds: float = Field(default=21_600, ge=300, le=604_800)
    raw_fsync_every: int = Field(default=200, ge=1)
    raw_queue_max_events: int = Field(default=20_000, ge=100, le=1_000_000)
    raw_write_batch_events: int = Field(default=500, ge=1, le=100_000)
    raw_write_flush_seconds: float = Field(default=0.025, gt=0, le=5)
    compress_rotated_files: bool = True
    compression_level: int = Field(default=6, ge=1, le=9)
    request_timeout_seconds: float = Field(default=20, gt=0)
    reconnect_min_seconds: float = Field(default=1, gt=0)
    reconnect_max_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_reconnect(self) -> "CollectorConfig":
        if self.reconnect_max_seconds < self.reconnect_min_seconds:
            raise ValueError("reconnect_max_seconds must be >= reconnect_min_seconds")
        if self.forecast_retry_max_seconds < self.forecast_retry_base_seconds:
            raise ValueError("forecast_retry_max_seconds must be >= forecast_retry_base_seconds")
        return self


class MarketConfig(BaseModel):
    candidate_days_ahead: int = Field(default=3, ge=0, le=14)
    reject_station_mismatch: bool = True
    reject_unit_mismatch: bool = True
    capture_mode: Literal["full", "compact"] = "compact"
    compact_interval_seconds: float = Field(default=5.0, ge=0.1, le=60)
    compact_depth_levels: int = Field(default=5, ge=1, le=25)
    capture_raw_ws_in_bursts: bool = False
    burst_before_seconds: float = Field(default=30.0, ge=0, le=600)
    burst_after_seconds: float = Field(default=120.0, ge=0, le=3600)
    burst_max_buffer_events: int = Field(default=10_000, ge=100, le=200_000)
    burst_max_buffer_bytes: int = Field(
        default=128_000_000, ge=1_000_000, le=2_000_000_000
    )
    ws_max_queue_messages: int = Field(default=64, ge=1, le=512)
    ws_max_message_bytes: int = Field(default=8_000_000, ge=100_000, le=50_000_000)
    ws_application_ping_seconds: float = Field(default=10.0, ge=1, le=300)
    ws_application_pong_timeout_seconds: float = Field(default=45.0, ge=5, le=600)

    @model_validator(mode="after")
    def validate_ws_heartbeat(self) -> "MarketConfig":
        if self.ws_application_pong_timeout_seconds <= self.ws_application_ping_seconds:
            raise ValueError(
                "ws_application_pong_timeout_seconds must exceed ws_application_ping_seconds"
            )
        return self


class ExecutionConfig(BaseModel):
    markout_seconds: list[int] = Field(
        default_factory=lambda: [5, 15, 30, 60, 300, 1800]
    )
    max_ws_health_age_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_scenarios(self) -> "ExecutionConfig":
        for name, values in (
            ("markout_seconds", self.markout_seconds),
        ):
            if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique positive values")
        return self


class PredictiveConfig(BaseModel):
    peer_max_age_minutes: int = Field(default=90, gt=0)
    scheduled_interval_minutes: int = Field(default=15, ge=1, le=180)


class PaperTradingConfig(BaseModel):


    enabled: bool = True
    edge_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    persistence_minutes: int = Field(default=15, ge=1, le=180)
    minimum_confirmations: int = Field(default=2, ge=2, le=20)
    minimum_model_probability: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_market_ask: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_market_ask: float = Field(default=0.95, ge=0.0, le=1.0)


    minimum_market_mid_mass: float = Field(default=0.95, ge=0.0, le=1.5)
    requested_shares: float = Field(default=10.0, gt=0.0, le=100_000.0)
    default_execution_buffer: float = Field(default=0.02, ge=0.0, le=0.25)
    minimum_forecast_providers: int = Field(default=2, ge=1, le=10)
    maximum_forecast_spread_c: float = Field(default=3.0, gt=0.0, le=20.0)
    maximum_forecast_age_minutes: float = Field(default=45.0, gt=0.0, le=720.0)
    minimum_peer_coverage: float = Field(default=0.25, ge=0.0, le=1.0)
    maximum_observation_age_minutes: float = Field(default=90.0, gt=0.0, le=720.0)
    minimum_calibration_days: int = Field(default=20, ge=5, le=1000)
    minimum_alpha_station_days: int = Field(default=20, ge=10, le=1000)
    maximum_entries_per_station_day: int = Field(default=1, ge=1, le=1)
    model_trees: int = Field(default=100, ge=50, le=1000)
    ncei_source_weight: float = Field(default=0.5, gt=0.0, le=1.0)
    recency_half_life_years: float = Field(default=6.0, gt=0.0, le=50.0)
    excluded_calibration_dates: list[str] = Field(
        default_factory=lambda: ["2026-05-19"]
    )

    @model_validator(mode="after")
    def validate_paper_rules(self) -> "PaperTradingConfig":
        if self.maximum_market_ask <= self.minimum_market_ask:
            raise ValueError("maximum_market_ask must exceed minimum_market_ask")
        if len(self.excluded_calibration_dates) != len(
            set(self.excluded_calibration_dates)
        ):
            raise ValueError("excluded_calibration_dates must be unique")
        return self


class AnalysisConfig(BaseModel):
    max_day_start_lag_minutes: int = Field(default=90, ge=0)
    continuity_gap_seconds: float = Field(default=180.0, gt=0)
    resume_warmup_seconds: float = Field(default=600.0, ge=0)


class QualityConfig(BaseModel):
    max_clock_step_ms: float = Field(default=100.0, gt=0)
    max_ws_health_gap_seconds: float = Field(default=30.0, gt=0)
    minimum_ws_health_coverage: float = Field(default=0.995, ge=0.0, le=1.0)
    minimum_market_buckets: int = Field(default=2, ge=2)


class ClickHouseConfig(BaseModel):
    enabled: bool = False
    flush_rows: int = Field(default=1000, ge=1)
    flush_seconds: float = Field(default=2, gt=0)


class AppConfig(BaseModel):
    project_root: Path
    data_dir: Path
    reports_dir: Path
    stations: dict[str, StationConfig]
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    predictive: PredictiveConfig = Field(default_factory=PredictiveConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)

    @model_validator(mode="after")
    def validate_stations(self) -> "AppConfig":
        if not self.stations:
            raise ValueError("At least one station must be configured")
        ids = [station.station_id for station in self.stations.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("station_id values must be unique")
        keys = [station.key for station in self.stations.values()]
        if set(keys) != set(self.stations):
            raise ValueError("Each station key must match its stations mapping key")
        return self

    @property
    def selected_station_ids(self) -> list[str]:
        return [s.station_id for s in self.stations.values()]

    def config_hash(self) -> str:
        blob = self.model_dump_json(exclude={"project_root"}, round_trip=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def load_config(
    app_path: str | Path = "config/app.yaml",
    stations_path: str | Path = "config/stations.yaml",
    project_root: str | Path | None = None,
) -> AppConfig:
    root = Path(project_root or os.getcwd()).resolve()
    app_file = (root / app_path).resolve() if not Path(app_path).is_absolute() else Path(app_path)
    stations_file = (
        (root / stations_path).resolve()
        if not Path(stations_path).is_absolute()
        else Path(stations_path)
    )
    app_data = _load_yaml(app_file)
    station_data = _load_yaml(stations_file).get("stations", {})
    stations: dict[str, StationConfig] = {}
    for key, raw in station_data.items():
        stations[key] = StationConfig(key=key, **raw)

    data_dir = Path(app_data.pop("data_dir", "data"))
    reports_dir = Path(app_data.pop("reports_dir", "reports"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir

    cfg = AppConfig(
        project_root=root,
        data_dir=data_dir,
        reports_dir=reports_dir,
        stations=stations,
        **app_data,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return cfg
