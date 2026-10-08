CREATE DATABASE IF NOT EXISTS weather_alpha;

CREATE TABLE IF NOT EXISTS weather_alpha.events
(
    run_id String,
    schema_version UInt16,
    source LowCardinality(String),
    event_type LowCardinality(String),
    received_at DateTime64(9, 'UTC'),
    received_wall_ns UInt64,
    received_mono_ns UInt64,
    request_started_wall_ns Nullable(UInt64),
    response_received_wall_ns Nullable(UInt64),
    provider_timestamp_ns Nullable(UInt64),
    observation_timestamp_ns Nullable(UInt64),
    station_id LowCardinality(Nullable(String)),
    market_slug Nullable(String),
    asset_id Nullable(String),
    payload_sha256 FixedString(64),
    payload_json String,
    parse_errors Array(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(received_at)
ORDER BY (source, event_type, coalesce(station_id, ''), received_at, payload_sha256);
