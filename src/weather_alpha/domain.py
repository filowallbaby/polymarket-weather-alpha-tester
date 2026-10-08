from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import orjson
from pydantic import BaseModel, Field

from .timebase import ns_to_iso, now_pair

SCHEMA_VERSION = 1


class EventEnvelope(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    source: str
    event_type: str
    received_at: str
    received_wall_ns: int
    received_mono_ns: int
    request_started_wall_ns: int | None = None
    response_received_wall_ns: int | None = None
    provider_timestamp_ns: int | None = None
    observation_timestamp_ns: int | None = None
    station_id: str | None = None
    market_slug: str | None = None
    asset_id: str | None = None
    payload_sha256: str
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None
    parse_errors: list[str] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        source: str,
        event_type: str,
        payload: Any,
        station_id: str | None = None,
        market_slug: str | None = None,
        asset_id: str | None = None,
        request_started_wall_ns: int | None = None,
        response_received_wall_ns: int | None = None,
        provider_timestamp_ns: int | None = None,
        observation_timestamp_ns: int | None = None,
        parse_errors: list[str] | None = None,
        received_wall_ns: int | None = None,
        received_mono_ns: int | None = None,
    ) -> "EventEnvelope":
        stamp = now_pair()
        wall_ns = received_wall_ns or stamp.wall_ns
        mono_ns = received_mono_ns or stamp.mono_ns
        try:
            raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        except TypeError:
            raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return cls(
            run_id=run_id,
            source=source,
            event_type=event_type,
            received_at=ns_to_iso(wall_ns) or datetime.now(UTC).isoformat(),
            received_wall_ns=wall_ns,
            received_mono_ns=mono_ns,
            request_started_wall_ns=request_started_wall_ns,
            response_received_wall_ns=response_received_wall_ns,
            provider_timestamp_ns=provider_timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
            station_id=station_id,
            market_slug=market_slug,
            asset_id=asset_id,
            payload_sha256=hashlib.sha256(raw).hexdigest(),
            payload=payload,
            parse_errors=parse_errors or [],
        )


def new_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
