from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import os
import re
import socket
import ssl
from datetime import UTC, datetime
from typing import Any

import orjson
from dateutil import parser as date_parser

from ..config import AppConfig
from ..domain import EventEnvelope
from ..parsing.metar import parse_awc_metar
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import datetime_to_ns, now_pair
from .nws_tgftp import infer_metar_observation_time

logger = logging.getLogger(__name__)


DEFAULT_TOPICS = (
    "origin/a/wis2/+/data/core/S/A/#",
    "origin/a/wis2/+/data/core/S/P/#",
)
_MAX_NOTIFICATION_BYTES = 10 * 1024 * 1024
_REPORT_START = re.compile(
    r"(?m)(?:^|\n)[ \t]*(?P<prefix>(?:METAR|SPECI)(?:[ \t]+COR)?[ \t]+)?"
    r"(?P<station>[A-Z][A-Z0-9]{3})[ \t]+(?P<time>\d{6}Z)\b"
)


class Wis2Collector:


    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.host = os.getenv("WIS2_BROKER_HOST", "globalbroker.meteo.fr").strip()
        self.port = _env_port("WIS2_BROKER_PORT", 8883)
        self.username = os.getenv("WIS2_USERNAME", "everyone")
        self.password = os.getenv("WIS2_PASSWORD", "everyone")
        configured_id = os.getenv("WIS2_CLIENT_ID", "").strip()
        stable_seed = f"{socket.gethostname()}|{cfg.project_root}|wis2".encode()
        self.client_id = configured_id or f"weather-alpha-{hashlib.sha256(stable_seed).hexdigest()[:24]}"
        configured_topics = [
            value.strip()
            for value in os.getenv("WIS2_TOPICS", "").split(",")
            if value.strip()
        ]
        self.topics = tuple(configured_topics or DEFAULT_TOPICS)
        self._targets = {
            station.station_id: station
            for station in cfg.stations.values()
            if station.sources.wis2
        }
        self._peer_targets: dict[str, list[str]] = {}
        for target_id, station in self._targets.items():
            for peer_id in station.peer_station_ids:
                peer_id = peer_id.upper()
                if peer_id != target_id:
                    self._peer_targets.setdefault(peer_id, []).append(target_id)
        self.station_ids = set(self._targets) | set(self._peer_targets)
        self._seen: set[tuple[str, str]] = set()

    async def run(self) -> None:
        if not self.station_ids:
            return
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=20_000)
        loop = asyncio.get_running_loop()
        client = self._build_client(loop, queue)
        try:
            await asyncio.to_thread(client.connect, self.host, self.port, 60)
            client.loop_start()
            event_type, payload = await asyncio.wait_for(queue.get(), timeout=30)
            if event_type != "connected":
                raise RuntimeError(f"WIS2 MQTT connection failed: {payload}")
            await self._record_status("mqtt_connected", payload)

            while True:
                event_type, payload = await queue.get()
                try:
                    if event_type == "message":
                        await self._handle_message(payload)
                    elif event_type == "connected":
                        await self._record_status("mqtt_reconnected", payload)
                    elif event_type == "subscribed":
                        await self._record_status("mqtt_subscribed", payload)
                    elif event_type == "disconnected":
                        await self._record_status("mqtt_disconnected", payload)
                    elif event_type == "connect_error":
                        raise RuntimeError(f"WIS2 MQTT connection failed: {payload}")
                    elif event_type == "queue_overflow":
                        await self._record_status("mqtt_queue_overflow", payload)
                except asyncio.CancelledError:
                    raise
                except CriticalSinkError:
                    raise
                except Exception as exc:
                    logger.exception("WIS2 notification processing failure")
                    await self.sink.append(
                        EventEnvelope.build(
                            run_id=self.run_id,
                            source="wis2_metar",
                            event_type="notification_parse_error",
                            payload={"error": repr(exc), "topic": payload.get("topic")},
                        )
                    )
                finally:
                    queue.task_done()
        finally:
            try:
                await asyncio.to_thread(client.disconnect)
            finally:
                client.loop_stop()

    def _build_client(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> Any:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("Install the paho-mqtt runtime dependency") from exc

        dropped = 0

        def publish(event_type: str, payload: Any) -> None:
            def put() -> None:
                nonlocal dropped
                try:
                    queue.put_nowait((event_type, payload))
                except asyncio.QueueFull:
                    dropped += 1
                    if dropped == 1 or dropped % 1000 == 0:
                        try:
                            queue.put_nowait(("queue_overflow", {"dropped": dropped}))
                        except asyncio.QueueFull:
                            pass

            loop.call_soon_threadsafe(put)

        def on_connect(
            client: Any,
            userdata: Any,
            flags: Any,
            reason_code: Any,
            properties: Any = None,
        ) -> None:
            if _reason_is_failure(reason_code):
                publish("connect_error", {"reason": str(reason_code)})
                return
            for topic in self.topics:
                client.subscribe(topic, qos=1)
            publish(
                "connected",
                {
                    "reason": str(reason_code),
                    "session_present": bool(getattr(flags, "session_present", False)),
                    "host": self.host,
                    "port": self.port,
                    "topics": list(self.topics),
                },
            )

        def on_subscribe(
            client: Any,
            userdata: Any,
            mid: int,
            reason_codes: Any,
            properties: Any = None,
        ) -> None:
            publish(
                "subscribed",
                {"message_id": mid, "reason_codes": [str(code) for code in reason_codes]},
            )

        def on_message(client: Any, userdata: Any, message: Any) -> None:
            stamp = now_pair()
            raw = bytes(message.payload)
            publish(
                "message",
                {
                    "topic": message.topic,
                    "payload": raw if len(raw) <= _MAX_NOTIFICATION_BYTES else b"",
                    "payload_bytes": len(raw),
                    "oversized": len(raw) > _MAX_NOTIFICATION_BYTES,
                    "qos": int(message.qos),
                    "retain": bool(message.retain),
                    "mid": int(message.mid),
                    "received_wall_ns": stamp.wall_ns,
                    "received_mono_ns": stamp.mono_ns,
                },
            )

        def on_disconnect(
            client: Any,
            userdata: Any,
            disconnect_flags: Any,
            reason_code: Any,
            properties: Any = None,
        ) -> None:
            publish("disconnected", {"reason": str(reason_code)})

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            protocol=mqtt.MQTTv5,
            transport="tcp",
        )
        client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.username_pw_set(self.username, self.password)
        client.reconnect_delay_set(
            min_delay=max(1, int(self.cfg.collector.reconnect_min_seconds)),
            max_delay=max(1, int(self.cfg.collector.reconnect_max_seconds)),
        )
        client.on_connect = on_connect
        client.on_subscribe = on_subscribe
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        return client

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if message["oversized"]:
            raise ValueError(
                f"WIS2 notification exceeds {_MAX_NOTIFICATION_BYTES} bytes: "
                f"{message['payload_bytes']}"
            )
        notification = orjson.loads(message["payload"])
        if not isinstance(notification, dict):
            raise ValueError("WIS2 notification is not an object")

        provider_time = _notification_time(notification)
        content, content_error = decode_inline_content(notification.get("content"))
        properties = notification.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        links = _safe_links(notification.get("links"))
        notification_id = str(notification.get("id") or "") or None
        data_id = properties.get("data_id") or properties.get("data-id")
        common = {
            "topic": message["topic"],
            "notification_id": notification_id,
            "data_id": data_id,
            "wigos_station_identifier": properties.get("wigos_station_identifier")
            or properties.get("wigos-station-identifier"),
            "inline_content": content is not None,
            "inline_bytes": len(content) if content is not None else 0,
            "inline_sha256": hashlib.sha256(content).hexdigest() if content is not None else None,
            "content_error": content_error,
            "links": links,
            "qos": message["qos"],
            "retain": message["retain"],
        }
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="wis2_metar",
                event_type="notification",
                payload=common,
                provider_timestamp_ns=datetime_to_ns(provider_time),
                received_wall_ns=message["received_wall_ns"],
                received_mono_ns=message["received_mono_ns"],
            )
        )
        if content is None:
            return
        bulletin = content.decode("utf-8", errors="replace")
        reports = extract_tac_reports(bulletin, self.station_ids)
        for station_id, raw_report in reports:
            seen_key = (station_id, raw_report)
            if seen_key in self._seen:
                continue
            self._seen.add(seen_key)
            reference = provider_time or datetime.fromtimestamp(
                message["received_wall_ns"] / 1_000_000_000, tz=UTC
            )
            observation_time = infer_metar_observation_time(raw_report, reference)
            normalized, errors = parse_awc_metar(
                {
                    "icaoId": station_id,
                    "rawOb": raw_report,
                    "reportTime": observation_time.isoformat() if observation_time else None,
                }
            )
            normalized.update(
                {
                    "wis2_topic": message["topic"],
                    "wis2_notification_id": notification_id,
                    "wis2_data_id": data_id,
                    "backfill": False,
                    "is_latest_in_response": True,
                }
            )
            if station_id in self._targets:
                await self._append_observation(
                    "wis2_metar",
                    "weather_observation",
                    station_id,
                    normalized,
                    errors,
                    provider_time,
                    observation_time,
                    message,
                )
            for target_id in self._peer_targets.get(station_id, []):
                await self._append_observation(
                    "wis2_peer_metar",
                    "peer_weather_observation",
                    target_id,
                    {
                        **normalized,
                        "peer_station_id": station_id,
                        "target_station_id": target_id,
                    },
                    errors,
                    provider_time,
                    observation_time,
                    message,
                )

    async def _append_observation(
        self,
        source: str,
        event_type: str,
        station_id: str,
        payload: dict[str, Any],
        errors: list[str],
        provider_time: datetime | None,
        observation_time: datetime | None,
        message: dict[str, Any],
    ) -> None:
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source=source,
                event_type=event_type,
                payload=payload,
                station_id=station_id,
                provider_timestamp_ns=datetime_to_ns(provider_time),
                observation_timestamp_ns=datetime_to_ns(observation_time),
                parse_errors=errors,
                received_wall_ns=message["received_wall_ns"],
                received_mono_ns=message["received_mono_ns"],
            )
        )

    async def _record_status(self, event_type: str, payload: dict[str, Any]) -> None:
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="wis2_metar",
                event_type=event_type,
                payload={**payload, "client_id": self.client_id},
            )
        )


def decode_inline_content(content: Any) -> tuple[bytes | None, str | None]:
    if content is None:
        return None, None
    if not isinstance(content, dict):
        return None, "content_not_object"
    value = content.get("value")
    if value is None:
        return None, "content_value_missing"
    if not isinstance(value, str):
        return None, "content_value_not_string"
    encoding = str(content.get("encoding") or "utf-8").lower()
    if encoding in {"base64", "base64url"}:
        try:
            if encoding == "base64url":
                value += "=" * (-len(value) % 4)
                return base64.urlsafe_b64decode(value), None
            return base64.b64decode(value, validate=True), None
        except (binascii.Error, ValueError):
            return None, "invalid_base64_content"
    if encoding in {"utf-8", "utf8", "text", "identity"}:
        return value.encode(), None
    return None, f"unsupported_content_encoding:{encoding}"


def extract_tac_reports(text: str, station_ids: set[str]) -> list[tuple[str, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    matches = list(_REPORT_START.finditer(normalized))
    reports: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        station_id = match.group("station")
        if station_id not in station_ids:
            continue
        start = match.start("prefix") if match.group("prefix") else match.start("station")
        equals = normalized.find("=", match.end())
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        end = equals if equals >= 0 and equals < next_start else next_start
        report = " ".join(normalized[start:end].split()).strip(" =")
        if report:
            reports.append((station_id, report))
    return reports


def _notification_time(notification: dict[str, Any]) -> datetime | None:
    properties = notification.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    candidates = (
        properties.get("datetime"),
        properties.get("pubtime"),
        properties.get("created"),
        notification.get("time"),
    )
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        try:
            parsed = date_parser.isoparse(value)
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _safe_links(raw_links: Any) -> list[dict[str, str]]:
    if not isinstance(raw_links, list):
        return []
    links: list[dict[str, str]] = []
    for item in raw_links[:20]:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href") or "")
        if not href:
            continue
        links.append(
            {
                "rel": str(item.get("rel") or ""),
                "type": str(item.get("type") or ""),
                "href": href[:2048],
            }
        )
    return links


def _reason_is_failure(reason_code: Any) -> bool:
    flag = getattr(reason_code, "is_failure", None)
    if flag is not None:
        return bool(flag)
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return str(reason_code).lower() not in {"success", "0"}


def _env_port(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port
