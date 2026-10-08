from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import orjson
import websockets

from ..config import AppConfig, StationConfig
from ..domain import EventEnvelope
from ..http_client import TimedHttpClient
from ..orderbook import OrderBook
from ..parsing.rules import parse_weather_market_rules, validate_rules
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import now_pair

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrackedToken:
    asset_id: str
    paired_no_asset_id: str | None
    outcome: str
    neg_risk: bool
    market_id: str | None
    condition_id: str | None
    event_slug: str
    market_slug: str | None
    station_key: str
    station_id: str
    unit: str
    event_date: str | None
    bucket_label: str
    question: str
    accepting_orders: bool
    tick_size: float | None
    fees_enabled: bool
    fee_rate: float | None
    fee_exponent: float | None
    fee_rebate_rate: float | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class PolymarketCollector:
    GAMMA_URL = "https://gamma-api.polymarket.com/events"
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink, http: TimedHttpClient) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.http = http
        self._tokens: dict[str, TrackedToken] = {}
        self._books: dict[str, OrderBook] = {}
        self._ws_task: asyncio.Task[None] | None = None
        self._ping_sent_wall_ns: int | None = None
        self._ping_sent_mono_ns: int | None = None
        self._seen_gamma_hash: dict[str, str] = {}
        self._connection_id = 0

    async def run(self) -> None:
        try:
            while True:
                try:
                    new_tokens = await self.discover()
                    tokens_changed = set(new_tokens) != set(self._tokens)
                    ws_failed = self._ws_task is not None and self._ws_task.done()

                    if ws_failed and self._ws_task is not None:
                        try:
                            failure = self._ws_task.exception()
                        except asyncio.CancelledError:
                            failure = None
                        if isinstance(failure, CriticalSinkError):
                            raise failure
                        if failure is not None:
                            await self.sink.append(
                                EventEnvelope.build(
                                    run_id=self.run_id,
                                    source="polymarket_ws",
                                    event_type="collector_error",
                                    payload={"error": repr(failure), "action": "restart"},
                                )
                            )
                        self._ws_task = None

                    if tokens_changed:
                        self._tokens = new_tokens
                        self._books = {asset: OrderBook(asset) for asset in self._tokens}
                        if self._ws_task:
                            self._ws_task.cancel()
                            try:
                                await self._ws_task
                            except asyncio.CancelledError:
                                pass
                            self._ws_task = None

                    if self._tokens and self._ws_task is None:
                        self._ws_task = asyncio.create_task(self._run_ws())
                except asyncio.CancelledError:
                    raise
                except CriticalSinkError:
                    raise
                except Exception as exc:
                    logger.exception("Polymarket discovery collector failure")
                    await self.sink.append(
                        EventEnvelope.build(
                            run_id=self.run_id,
                            source="polymarket_gamma",
                            event_type="collector_error",
                            payload={"error": repr(exc), "action": "retry"},
                        )
                    )
                if self._ws_task is None:
                    await asyncio.sleep(self.cfg.collector.market_discovery_seconds)
                else:
                    await asyncio.wait(
                        {self._ws_task},
                        timeout=self.cfg.collector.market_discovery_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
        finally:
            if self._ws_task:
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass

    async def discover(self) -> dict[str, TrackedToken]:
        tokens: dict[str, TrackedToken] = {}
        for station in self.cfg.stations.values():
            events = await self._discover_station_events(station)
            for event in events:
                event_tokens = await self._parse_event(station, event)
                tokens.update({token.asset_id: token for token in event_tokens})
        return tokens

    async def _discover_station_events(self, station: StationConfig) -> list[dict[str, Any]]:
        local_now = datetime.now(ZoneInfo(station.timezone))
        events: list[dict[str, Any]] = []

        for offset in range(-1, self.cfg.market.candidate_days_ahead + 1):
            day = (local_now + timedelta(days=offset)).date()
            month = calendar.month_name[day.month].lower()
            slug = f"highest-temperature-in-{station.city_slug}-on-{month}-{day.day}-{day.year}"
            response = await self.http.get(self.GAMMA_URL, params={"slug": slug})
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="polymarket_gamma",
                    event_type="discovery_response",
                    payload={
                        "candidate_slug": slug,
                        "status": response.status,
                        "headers": response.headers,
                        "body_bytes": len(response.body),
                        "body_sha256": hashlib.sha256(response.body).hexdigest(),
                    },
                    station_id=station.station_id,
                    market_slug=slug,
                    request_started_wall_ns=response.request_started_wall_ns,
                    response_received_wall_ns=response.response_received_wall_ns,
                    received_wall_ns=response.response_received_wall_ns,
                    received_mono_ns=response.response_received_mono_ns,
                )
            )
            if response.status != 200:
                continue
            payload = response.json()
            body_hash = hashlib.sha256(response.body).hexdigest()
            if self._seen_gamma_hash.get(slug) != body_hash:
                self._seen_gamma_hash[slug] = body_hash
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="polymarket_gamma",
                        event_type="discovery_payload",
                        payload={
                            "candidate_slug": slug,
                            "body_sha256": body_hash,
                            "response": payload,
                        },
                        station_id=station.station_id,
                        market_slug=slug,
                        request_started_wall_ns=response.request_started_wall_ns,
                        response_received_wall_ns=response.response_received_wall_ns,
                        received_wall_ns=response.response_received_wall_ns,
                        received_mono_ns=response.response_received_mono_ns,
                    )
                )
            if isinstance(payload, dict):
                payload = payload.get("events") or payload.get("data") or [payload]
            if isinstance(payload, list):
                events.extend(item for item in payload if isinstance(item, dict))

        unique: dict[str, dict[str, Any]] = {}
        for event in events:
            key = str(event.get("id") or event.get("slug") or hash(orjson.dumps(event)))
            unique[key] = event
        return list(unique.values())

    async def _parse_event(
        self, station: StationConfig, event: dict[str, Any]
    ) -> list[TrackedToken]:
        event_slug = str(event.get("slug") or "")
        rules_text = str(event.get("description") or event.get("rules") or "")
        markets = event.get("markets") or []
        if isinstance(markets, str):
            try:
                markets = json.loads(markets)
            except json.JSONDecodeError:
                markets = []
        if not rules_text and markets:
            rules_text = str(markets[0].get("description") or "")
        parsed = parse_weather_market_rules(rules_text)
        event_date = parsed.event_date or self._event_date_from_slug(event_slug)
        errors = validate_rules(
            parsed,
            expected_station_id=station.station_id,
            expected_unit=station.market_unit,
            reject_station_mismatch=self.cfg.market.reject_station_mismatch,
            reject_unit_mismatch=self.cfg.market.reject_unit_mismatch,
        )
        fatal = any(error.startswith("fatal_") for error in errors)
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="polymarket_gamma",
                event_type="market_rules",
                payload={
                    "event_id": event.get("id"),
                    "event_slug": event_slug,
                    "title": event.get("title"),
                    "parsed_station_id": parsed.station_id,
                    "parsed_station_name": parsed.station_name,
                    "parsed_unit": parsed.unit,
                    "parsed_event_date": parsed.event_date.isoformat()
                    if parsed.event_date
                    else None,
                    "effective_event_date": event_date.isoformat() if event_date else None,
                    "resolution_url": parsed.resolution_url,
                    "fatal": fatal,
                    "rules_text": rules_text,
                },
                station_id=station.station_id,
                market_slug=event_slug,
                parse_errors=errors,
            )
        )
        if fatal:
            logger.error("Rejected %s due to rule mismatch: %s", event_slug, errors)
            return []

        tokens: list[TrackedToken] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("closed") is True or market.get("active") is False:
                continue
            if market.get("acceptingOrders") is False or market.get("accepting_orders") is False:
                continue
            token_ids = self._json_list(market.get("clobTokenIds") or market.get("clob_token_ids"))
            outcomes = self._json_list(market.get("outcomes"))
            if not token_ids:
                continue
            yes_index = 0
            no_index: int | None = None
            for index, outcome in enumerate(outcomes):
                normalized_outcome = str(outcome).casefold()
                if normalized_outcome == "yes":
                    yes_index = index
                elif normalized_outcome == "no":
                    no_index = index
            if yes_index >= len(token_ids):
                continue
            label = str(
                market.get("groupItemTitle")
                or market.get("group_item_title")
                or self._label_from_question(str(market.get("question") or ""))
            )
            fee_schedule = market.get("feeSchedule") or market.get("fee_schedule") or {}
            if not isinstance(fee_schedule, dict):
                fee_schedule = {}
            token = TrackedToken(
                asset_id=str(token_ids[yes_index]),
                paired_no_asset_id=(
                    str(token_ids[no_index])
                    if no_index is not None and no_index < len(token_ids)
                    else None
                ),
                outcome="Yes",
                neg_risk=bool(
                    market.get("negRisk")
                    or market.get("neg_risk")
                    or event.get("negRisk")
                    or event.get("enableNegRisk")
                ),
                market_id=str(market.get("id")) if market.get("id") is not None else None,
                condition_id=market.get("conditionId") or market.get("condition_id"),
                event_slug=event_slug,
                market_slug=market.get("slug"),
                station_key=station.key,
                station_id=station.station_id,
                unit=station.market_unit,
                event_date=event_date.isoformat() if event_date else None,
                bucket_label=label,
                question=str(market.get("question") or ""),
                accepting_orders=True,
                tick_size=self._number_or_none(
                    market.get("orderPriceMinTickSize") or market.get("order_price_min_tick_size")
                ),
                fees_enabled=bool(market.get("feesEnabled") or market.get("fees_enabled")),
                fee_rate=self._number_or_none(fee_schedule.get("rate")),
                fee_exponent=self._number_or_none(fee_schedule.get("exponent")),
                fee_rebate_rate=self._number_or_none(fee_schedule.get("rebateRate")),
            )
            tokens.append(token)
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="polymarket_gamma",
                    event_type="tracked_token",
                    payload=token.payload(),
                    station_id=station.station_id,
                    market_slug=event_slug,
                    asset_id=token.asset_id,
                )
            )
        return tokens

    async def _run_ws(self) -> None:
        delay = self.cfg.collector.reconnect_min_seconds
        while True:
            try:
                self._connection_id += 1
                await self._reset_books("connection_start")
                self._ping_sent_wall_ns = None
                self._ping_sent_mono_ns = None
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=self.cfg.market.ws_max_message_bytes,
                    max_queue=self.cfg.market.ws_max_queue_messages,
                    compression=None,
                ) as websocket:
                    subscription = {
                        "assets_ids": list(self._tokens),
                        "type": "market",
                        "custom_feature_enabled": True,
                    }
                    await websocket.send(orjson.dumps(subscription).decode())
                    await self.sink.append(
                        EventEnvelope.build(
                            run_id=self.run_id,
                            source="polymarket_ws",
                            event_type="subscribed",
                            payload={
                                "asset_count": len(self._tokens),
                                "connection_id": self._connection_id,
                            },
                        )
                    )
                    heartbeat = asyncio.create_task(self._heartbeat(websocket))
                    delay = self.cfg.collector.reconnect_min_seconds
                    try:
                        async for raw_message in websocket:
                            stamp = now_pair()
                            if raw_message == "PONG":
                                await self._record_pong(stamp.wall_ns, stamp.mono_ns)
                                continue
                            try:
                                decoded = orjson.loads(raw_message)
                            except orjson.JSONDecodeError as exc:
                                await self.sink.append(
                                    EventEnvelope.build(
                                        run_id=self.run_id,
                                        source="polymarket_ws",
                                        event_type="decode_error",
                                        payload={
                                            "raw": str(raw_message)[:2000],
                                            "error": repr(exc),
                                        },
                                        received_wall_ns=stamp.wall_ns,
                                        received_mono_ns=stamp.mono_ns,
                                    )
                                )
                                continue
                            messages = decoded if isinstance(decoded, list) else [decoded]
                            for message in messages:
                                if isinstance(message, dict):
                                    await self._handle_ws_message(
                                        message, stamp.wall_ns, stamp.mono_ns
                                    )
                    finally:
                        heartbeat.cancel()
                        try:
                            await heartbeat
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except CriticalSinkError:
                raise
            except Exception as exc:
                logger.warning("Polymarket WS disconnected: %r", exc)
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="polymarket_ws",
                        event_type="connection_error",
                        payload={"error": repr(exc), "retry_seconds": delay},
                    )
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.cfg.collector.reconnect_max_seconds)

    async def _heartbeat(self, websocket: Any) -> None:
        while True:
            await asyncio.sleep(self.cfg.market.ws_application_ping_seconds)
            stamp = now_pair()
            if self._ping_sent_mono_ns is not None:
                outstanding_seconds = (stamp.mono_ns - self._ping_sent_mono_ns) / 1e9
                if outstanding_seconds > self.cfg.market.ws_application_pong_timeout_seconds:
                    await websocket.close(code=1011, reason="application PONG timeout")
                    raise TimeoutError(
                        f"Polymarket application PONG missing for {outstanding_seconds:.1f}s"
                    )
                continue
            self._ping_sent_wall_ns = stamp.wall_ns
            self._ping_sent_mono_ns = stamp.mono_ns
            await websocket.send("PING")

    async def _record_pong(self, wall_ns: int, mono_ns: int) -> None:
        if self._ping_sent_mono_ns is None:
            return
        rtt_ns = mono_ns - self._ping_sent_mono_ns
        ping_sent_wall_ns = self._ping_sent_wall_ns
        self._ping_sent_wall_ns = None
        self._ping_sent_mono_ns = None
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="polymarket_ws",
                event_type="heartbeat_rtt",
                payload={
                    "rtt_ns": rtt_ns,
                    "rtt_ms": rtt_ns / 1e6,
                    "ping_sent_wall_ns": ping_sent_wall_ns,
                    "pong_received_wall_ns": wall_ns,
                },
                received_wall_ns=wall_ns,
                received_mono_ns=mono_ns,
            )
        )

    async def _handle_ws_message(self, message: dict[str, Any], wall_ns: int, mono_ns: int) -> None:
        event_type = str(message.get("event_type") or message.get("type") or "unknown")
        asset_id = str(message.get("asset_id") or "") or None
        token = self._tokens.get(asset_id or "")
        exchange_timestamp = self._exchange_timestamp_ns(message.get("timestamp"))


        if self.cfg.market.capture_raw_ws_in_bursts:
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="polymarket_ws_raw",
                    event_type=event_type,
                    payload=message,
                    station_id=token.station_id if token else None,
                    market_slug=token.event_slug if token else None,
                    asset_id=asset_id,
                    provider_timestamp_ns=exchange_timestamp,
                    received_wall_ns=wall_ns,
                    received_mono_ns=mono_ns,
                )
            )

        changed_assets: set[str] = set()
        errors: list[str] = []
        try:
            if event_type == "book" and asset_id in self._books:
                self._books[asset_id].apply_snapshot(message, wall_ns)
                changed_assets.add(asset_id)
            elif event_type == "price_change":
                for change in message.get("price_changes") or []:
                    changed_id = str(change.get("asset_id") or "")
                    if changed_id not in self._books:
                        continue
                    self._books[changed_id].apply_change(change, message.get("timestamp"), wall_ns)
                    changed_assets.add(changed_id)
            elif event_type == "best_bid_ask" and asset_id in self._books:

                changed_assets.add(asset_id)
        except Exception as exc:
            errors.append(repr(exc))

        for changed_id in changed_assets:
            tracked = self._tokens[changed_id]
            book = self._books[changed_id]
            normalized = self._book_payload(book, tracked, event_type)

            if event_type == "best_bid_ask":
                normalized["best_bid"] = message.get("best_bid")
                normalized["best_ask"] = message.get("best_ask")
                normalized["spread"] = message.get("spread")
                if (
                    normalized.get("best_bid") is not None
                    and normalized.get("best_ask") is not None
                ):
                    normalized["mid"] = str(
                        (float(normalized["best_bid"]) + float(normalized["best_ask"])) / 2
                    )
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="polymarket_book",
                    event_type="book_state",
                    payload=normalized,
                    station_id=tracked.station_id,
                    market_slug=tracked.event_slug,
                    asset_id=changed_id,
                    provider_timestamp_ns=exchange_timestamp,
                    parse_errors=errors,
                    received_wall_ns=wall_ns,
                    received_mono_ns=mono_ns,
                )
            )

    async def _reset_books(self, reason: str) -> None:
        stamp = now_pair()
        for asset_id, book in self._books.items():
            book.reset()
            tracked = self._tokens[asset_id]
            payload = self._book_payload(book, tracked, "connection_reset")
            payload["reset_reason"] = reason
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="polymarket_book",
                    event_type="book_state",
                    payload=payload,
                    station_id=tracked.station_id,
                    market_slug=tracked.event_slug,
                    asset_id=asset_id,
                    received_wall_ns=stamp.wall_ns,
                    received_mono_ns=stamp.mono_ns,
                )
            )

    def _book_payload(
        self, book: OrderBook, tracked: TrackedToken, source_event_type: str
    ) -> dict[str, Any]:
        normalized = book.normalized()
        normalized.update(
            {
                "bucket_label": tracked.bucket_label,
                "outcome": tracked.outcome,
                "paired_no_asset_id": tracked.paired_no_asset_id,
                "neg_risk": tracked.neg_risk,
                "station_key": tracked.station_key,
                "station_id": tracked.station_id,
                "unit": tracked.unit,
                "event_date": tracked.event_date,
                "event_slug": tracked.event_slug,
                "market_slug": tracked.market_slug,
                "accepting_orders": tracked.accepting_orders,
                "tick_size": tracked.tick_size,
                "fees_enabled": tracked.fees_enabled,
                "fee_rate": tracked.fee_rate,
                "fee_exponent": tracked.fee_exponent,
                "fee_rebate_rate": tracked.fee_rebate_rate,
                "connection_id": self._connection_id,
                "source_event_type": source_event_type,
                "depth_valid": (
                    book.has_snapshot
                    and not book.is_crossed
                    and source_event_type in {"book", "price_change"}
                ),
            }
        )
        return normalized

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _number_or_none(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _label_from_question(question: str) -> str:

        lower = question.lower()
        marker = " be "
        if marker in lower:
            start = lower.index(marker) + len(marker)
            tail = question[start:]
            for date_marker in (" on ", " during "):
                if date_marker in tail.lower():
                    return tail[: tail.lower().index(date_marker)].strip()
            return tail.strip(" ?")
        return question

    @staticmethod
    def _event_date_from_slug(slug: str):
        import re
        from datetime import datetime

        match = re.search(r"-on-([a-z]+)-(\d{1,2})-(\d{4})$", slug, re.I)
        if not match:
            return None
        try:
            return datetime.strptime(
                f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y"
            ).date()
        except ValueError:
            return None

    @staticmethod
    def _exchange_timestamp_ns(value: Any) -> int | None:
        try:
            timestamp = int(value)
        except (TypeError, ValueError):
            return None
        if timestamp < 10_000_000_000:
            return timestamp * 1_000_000_000
        if timestamp < 10_000_000_000_000:
            return timestamp * 1_000_000
        return timestamp
