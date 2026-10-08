from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import orjson

from ..config import MarketConfig
from ..domain import EventEnvelope
from ..parsing.buckets import parse_temperature_bucket, round_half_up
from .multi import EventSink

_MARKET_SOURCES = {"polymarket_ws_raw", "polymarket_book"}


@dataclass(slots=True)
class _BufferedEvent:
    event: EventEnvelope
    estimated_bytes: int


class MarketCaptureSink:


    def __init__(self, sink: EventSink, cfg: MarketConfig, run_id: str = "capture-summary") -> None:
        self.sink = sink
        self.cfg = cfg
        self.run_id = run_id
        self._buffer: deque[_BufferedEvent] = deque()
        self._buffer_bytes = 0
        self._burst_until_ns = 0
        self._last_compact_ns: dict[str, int] = {}
        self._metrics: Counter[str] = Counter()
        self._signal_trigger_seen: dict[str, int] = {}
        self._market_labels: dict[tuple[str, str], set[str]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def append(self, event: EventEnvelope) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("MarketCaptureSink is closed")
            self._record_market_label(event)
            event = self._annotate_forecast_revision(event)
            if self.cfg.capture_mode == "full":
                await self.sink.append(event)
                self._metrics["persisted_full"] += 1
                return

            if self._is_signal_trigger(event):
                is_forecast_revision = self._is_forecast_revision_trigger(event)
                if self._register_signal_trigger(event):
                    self._metrics["burst_triggers"] += 1
                    self._metrics[
                        "forecast_revision_burst_triggers"
                        if is_forecast_revision
                        else "weather_burst_triggers"
                    ] += 1
                    cutoff_ns = event.received_wall_ns - int(self.cfg.burst_before_seconds * 1e9)
                    await self._flush_pre_window(cutoff_ns)
                    self._burst_until_ns = max(
                        self._burst_until_ns,
                        event.received_wall_ns + int(self.cfg.burst_after_seconds * 1e9),
                    )
                else:
                    self._metrics[
                        "duplicate_forecast_revision_triggers"
                        if is_forecast_revision
                        else "duplicate_weather_triggers"
                    ] += 1
                    await self._drain_aged(event.received_wall_ns)
                await self.sink.append(event)
                return

            if not self._is_market_event(event):
                await self._drain_aged(event.received_wall_ns)
                await self.sink.append(event)
                return

            self._metrics["market_events_received"] += 1


            if (
                event.source == "polymarket_ws_raw"
                and not self.cfg.capture_raw_ws_in_bursts
            ):
                metric = (
                    "raw_burst_events_dropped"
                    if event.received_wall_ns <= self._burst_until_ns
                    else "raw_events_dropped"
                )
                self._metrics[metric] += 1
                self._metrics["raw_events_dropped_before_buffer"] += 1
                return
            if event.received_wall_ns <= self._burst_until_ns:
                await self._persist_burst_or_drop_raw(event, "burst_post")
                return

            await self._drain_aged(event.received_wall_ns)
            estimated_bytes = _estimated_event_bytes(event)
            self._buffer.append(_BufferedEvent(event, estimated_bytes))
            self._buffer_bytes += estimated_bytes
            self._metrics["max_buffer_events"] = max(
                self._metrics["max_buffer_events"], len(self._buffer)
            )
            self._metrics["max_buffer_bytes"] = max(
                self._metrics["max_buffer_bytes"], self._buffer_bytes
            )
            while (
                len(self._buffer) > self.cfg.burst_max_buffer_events
                or self._buffer_bytes > self.cfg.burst_max_buffer_bytes
            ):
                self._metrics["buffer_forced_drains"] += 1
                buffered = self._pop_buffered()
                self._metrics["buffer_forced_drain_bytes"] += buffered.estimated_bytes
                await self._persist_compact_or_drop(buffered.event)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            while self._buffer:
                await self._persist_compact_or_drop(self._pop_buffered().event)
            self._closed = True
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="system",
                    event_type="market_capture_summary",
                    payload=self.metrics(),
                )
            )
            await self.sink.close()

    def metrics(self) -> dict[str, Any]:
        result = dict(self._metrics)
        result.update(
            {
                "mode": self.cfg.capture_mode,
                "buffered_events": len(self._buffer),
                "buffered_bytes": self._buffer_bytes,
                "buffer_max_events": self.cfg.burst_max_buffer_events,
                "buffer_max_bytes": self.cfg.burst_max_buffer_bytes,
                "burst_active_until_ns": self._burst_until_ns or None,
                "compact_interval_seconds": self.cfg.compact_interval_seconds,
                "compact_depth_levels": self.cfg.compact_depth_levels,
                "capture_raw_ws_in_bursts": self.cfg.capture_raw_ws_in_bursts,
                "burst_before_seconds": self.cfg.burst_before_seconds,
                "burst_after_seconds": self.cfg.burst_after_seconds,
            }
        )
        return result

    async def _drain_aged(self, now_ns: int) -> None:
        cutoff_ns = now_ns - int(self.cfg.burst_before_seconds * 1e9)
        while self._buffer and self._buffer[0].event.received_wall_ns < cutoff_ns:
            await self._persist_compact_or_drop(self._pop_buffered().event)

    async def _flush_pre_window(self, cutoff_ns: int) -> None:
        while self._buffer:
            buffered = self._pop_buffered()
            if buffered.event.received_wall_ns >= cutoff_ns:
                await self._persist_burst_or_drop_raw(buffered.event, "burst_pre")
            else:
                await self._persist_compact_or_drop(buffered.event)

    def _pop_buffered(self) -> _BufferedEvent:
        buffered = self._buffer.popleft()
        self._buffer_bytes -= buffered.estimated_bytes
        return buffered

    async def _persist_burst_or_drop_raw(
        self, event: EventEnvelope, capture_mode: str
    ) -> None:
        if event.source == "polymarket_ws_raw" and not self.cfg.capture_raw_ws_in_bursts:
            self._metrics["raw_burst_events_dropped"] += 1
            return
        await self.sink.append(_with_capture(event, capture_mode))
        self._metrics["persisted_full"] += 1

    async def _persist_compact_or_drop(self, event: EventEnvelope) -> None:
        if event.source == "polymarket_ws_raw":
            self._metrics["raw_events_dropped"] += 1
            return
        if event.source != "polymarket_book" or not isinstance(event.payload, dict):
            await self.sink.append(event)
            self._metrics["persisted_full"] += 1
            return
        asset_key = event.asset_id or str(event.payload.get("asset_id") or "")
        interval_ns = int(self.cfg.compact_interval_seconds * 1e9)
        last_ns = self._last_compact_ns.get(asset_key)
        if last_ns is not None and event.received_wall_ns - last_ns < interval_ns:
            self._metrics["book_events_throttled"] += 1
            return
        payload = dict(event.payload)
        original_bid_levels = len(payload.get("bids") or [])
        original_ask_levels = len(payload.get("asks") or [])
        payload["bids"] = list(payload.get("bids") or [])[: self.cfg.compact_depth_levels]
        payload["asks"] = list(payload.get("asks") or [])[: self.cfg.compact_depth_levels]
        payload["_capture"] = {
            "mode": "compact",
            "original_bid_levels": original_bid_levels,
            "original_ask_levels": original_ask_levels,
        }
        await self.sink.append(_with_payload(event, payload))
        self._last_compact_ns[asset_key] = event.received_wall_ns
        self._metrics["persisted_compact_books"] += 1

    @staticmethod
    def _is_market_event(event: EventEnvelope) -> bool:
        return event.source in _MARKET_SOURCES

    @staticmethod
    def _is_forecast_revision_trigger(event: EventEnvelope) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        economic_change = payload.get("economic_bucket_changed")
        return bool(
            event.source.startswith("forecast_")
            and event.event_type == "forecast_revision"
            and (
                economic_change is True
                or (economic_change is None and payload.get("bucket_changed") is True)
            )
        )

    @classmethod
    def _is_signal_trigger(cls, event: EventEnvelope) -> bool:


        return cls._is_forecast_revision_trigger(event)

    def _record_market_label(self, event: EventEnvelope) -> None:
        if (
            event.source != "polymarket_gamma"
            or event.event_type != "tracked_token"
            or not isinstance(event.payload, dict)
        ):
            return
        station_id = str(event.station_id or event.payload.get("station_id") or "")
        local_date = str(event.payload.get("event_date") or "")
        label = str(event.payload.get("bucket_label") or "")
        if station_id and local_date and label:
            self._market_labels.setdefault((station_id, local_date), set()).add(label)

    def _annotate_forecast_revision(self, event: EventEnvelope) -> EventEnvelope:
        if (
            not event.source.startswith("forecast_")
            or event.event_type != "forecast_revision"
            or not isinstance(event.payload, dict)
        ):
            return event
        payload = dict(event.payload)
        station_id = str(event.station_id or "")
        local_date = str(payload.get("local_date") or "")
        unit = str(payload.get("market_unit") or "")
        labels = self._market_labels.get((station_id, local_date), set())

        def label_for(value: Any) -> str | None:
            try:
                represented = round_half_up(float(value))
            except (TypeError, ValueError):
                return None
            for label in labels:
                try:
                    if parse_temperature_bucket(label, unit).contains(represented):
                        return label
                except ValueError:
                    continue
            return None

        previous_label = label_for(payload.get("previous_market_max"))
        current_label = label_for(payload.get("current_market_max"))
        payload["previous_market_bucket_label"] = previous_label
        payload["current_market_bucket_label"] = current_label
        payload["economic_bucket_changed"] = (
            previous_label != current_label if previous_label and current_label else None
        )
        return _with_payload(event, payload)

    def _register_signal_trigger(self, event: EventEnvelope) -> bool:
        payload = event.payload if isinstance(event.payload, dict) else {}
        key = "|".join(
            [
                "forecast",
                str(event.station_id or ""),
                str(payload.get("local_date") or ""),
                str(payload.get("previous_market_bucket_label") or ""),
                str(payload.get("current_market_bucket_label") or ""),
                str(payload.get("previous_market_max") or ""),
                str(payload.get("current_market_max") or ""),
            ]
        )
        previous_seen = self._signal_trigger_seen.get(key)
        if previous_seen is not None and event.received_wall_ns - previous_seen <= 10_000_000_000:
            return False
        self._signal_trigger_seen[key] = event.received_wall_ns
        cutoff_ns = event.received_wall_ns - 48 * 60 * 60 * 1_000_000_000
        self._signal_trigger_seen = {
            identity: received_ns
            for identity, received_ns in self._signal_trigger_seen.items()
            if received_ns >= cutoff_ns
        }
        return True


def _with_capture(event: EventEnvelope, mode: str) -> EventEnvelope:
    if not isinstance(event.payload, dict):
        return event
    payload = dict(event.payload)
    payload["_capture"] = {"mode": mode}
    return _with_payload(event, payload)


def _estimated_event_bytes(event: EventEnvelope) -> int:
    try:
        return len(orjson.dumps(event.payload)) + 512
    except TypeError:
        return 4096


def _with_payload(event: EventEnvelope, payload: dict[str, Any]) -> EventEnvelope:
    return EventEnvelope.build(
        run_id=event.run_id,
        source=event.source,
        event_type=event.event_type,
        payload=payload,
        station_id=event.station_id,
        market_slug=event.market_slug,
        asset_id=event.asset_id,
        request_started_wall_ns=event.request_started_wall_ns,
        response_received_wall_ns=event.response_received_wall_ns,
        provider_timestamp_ns=event.provider_timestamp_ns,
        observation_timestamp_ns=event.observation_timestamp_ns,
        parse_errors=event.parse_errors,
        received_wall_ns=event.received_wall_ns,
        received_mono_ns=event.received_mono_ns,
    )
