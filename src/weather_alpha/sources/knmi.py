from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import socket
import ssl
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..domain import EventEnvelope
from ..http_client import TimedHttpClient
from ..storage.multi import CriticalSinkError, MultiSink
from ..timebase import datetime_to_ns, sleep_aligned

logger = logging.getLogger(__name__)


class KnmiCollector:
    BASE = "https://api.dataplatform.knmi.nl/open-data/v1"
    DATASET = "10-minute-in-situ-meteorological-observations"
    VERSION = "1.0"

    def __init__(self, cfg: AppConfig, run_id: str, sink: MultiSink, http: TimedHttpClient) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.sink = sink
        self.http = http
        self.api_key = (
            os.getenv("KNMI_OPEN_DATA_API_KEY", "").strip()
            or os.getenv("KNMI_API_KEY", "").strip()
        )
        self.notification_key = os.getenv("KNMI_NOTIFICATION_API_KEY", "").strip()
        configured_client_id = os.getenv("KNMI_NOTIFICATION_CLIENT_ID", "").strip()
        stable_seed = f"{socket.gethostname()}|{cfg.project_root}".encode()
        self.notification_client_id = configured_client_id or (
            f"weather-alpha-{hashlib.sha256(stable_seed).hexdigest()[:24]}"
        )
        self.stations = [s for s in cfg.stations.values() if s.sources.knmi]
        self._last_filename: str | None = None

    async def run(self) -> None:
        if not self.stations:
            return
        if not self.api_key:
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="knmi_10m",
                    event_type="collector_disabled",
                    payload={"reason": "KNMI_OPEN_DATA_API_KEY is not configured"},
                )
            )
            return
        await self._poll_safely("startup")
        if self.notification_key:
            try:
                await self._run_notification_service()
                return
            except asyncio.CancelledError:
                raise
            except CriticalSinkError:
                raise
            except Exception as exc:
                logger.exception("KNMI notification service unavailable; using polling fallback")
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="knmi_10m",
                        event_type="notification_error",
                        payload={"error": repr(exc), "fallback_polling": True},
                    )
                )
        while True:
            await sleep_aligned(self.cfg.collector.knmi_poll_seconds, phase_seconds=0.4)
            await self._poll_safely("reconciliation")

    async def _poll_safely(self, trigger: str) -> None:
        try:
            await self.poll_once(trigger=trigger)
        except asyncio.CancelledError:
            raise
        except CriticalSinkError:
            raise
        except Exception as exc:
            logger.exception("KNMI collector failure")
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="knmi_10m",
                    event_type="collector_error",
                    payload={"error": repr(exc), "trigger": trigger},
                )
            )

    async def poll_once(self, *, trigger: str = "poll") -> None:
        list_url = f"{self.BASE}/datasets/{self.DATASET}/versions/{self.VERSION}/files"
        if self._last_filename is None:
            response = await self._list_files(
                list_url,
                {"maxKeys": 1, "orderBy": "created", "sorting": "desc"},
                trigger=trigger,
                page=1,
            )
            files = response.get("files", [])
            if files and files[0].get("filename"):
                await self._download_filename(
                    str(files[0]["filename"]), files[0], trigger=trigger
                )
            return


        begin = self._last_filename
        next_page_token: str | None = None
        page = 0
        while True:
            page += 1
            params: dict[str, Any] = {
                "maxKeys": 100,
                "orderBy": "filename",
                "sorting": "asc",
                "begin": begin,
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token
            payload = await self._list_files(
                list_url,
                params,
                trigger=trigger,
                page=page,
            )
            files = sorted(
                (
                    item
                    for item in payload.get("files", [])
                    if isinstance(item, dict)
                    and str(item.get("filename") or "") > str(self._last_filename or "")
                ),
                key=lambda item: str(item["filename"]),
            )
            for metadata in files:
                await self._download_filename(
                    str(metadata["filename"]), metadata, trigger=trigger
                )

            next_page_token = str(payload.get("nextPageToken") or "") or None
            if next_page_token:
                continue
            if payload.get("isTruncated") is True and files:
                begin = str(files[-1]["filename"])
                continue
            break

    async def _list_files(
        self,
        list_url: str,
        params: dict[str, Any],
        *,
        trigger: str,
        page: int,
    ) -> dict[str, Any]:
        response = await self.http.get(
            list_url,
            params=params,
            headers={"Authorization": self.api_key},
        )
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="knmi_10m",
                event_type="list_response",
                payload={
                    "url": response.url,
                    "status": response.status,
                    "headers": response.headers,
                    "body_bytes": len(response.body),
                    "trigger": trigger,
                    "page": page,
                    "begin": params.get("begin"),
                },
                request_started_wall_ns=response.request_started_wall_ns,
                response_received_wall_ns=response.response_received_wall_ns,
                received_wall_ns=response.response_received_wall_ns,
                received_mono_ns=response.response_received_mono_ns,
            )
        )
        if response.status != 200:
            raise RuntimeError(
                f"KNMI list files returned {response.status}: {response.body[:300]!r}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("KNMI list files response is not an object")
        return payload

    async def _download_filename(
        self,
        filename: str,
        metadata: dict[str, Any],
        *,
        trigger: str,
    ) -> None:
        if self._last_filename is not None and filename <= self._last_filename:
            return

        url_endpoint = (
            f"{self.BASE}/datasets/{self.DATASET}/versions/{self.VERSION}/files/{filename}/url"
        )
        url_response = await self.http.get(url_endpoint, headers={"Authorization": self.api_key})
        if url_response.status != 200:
            raise RuntimeError(
                f"KNMI temporary URL returned {url_response.status}: {url_response.body[:300]!r}"
            )
        temp_url = url_response.json().get("temporaryDownloadUrl")
        if not temp_url:
            raise ValueError("KNMI response did not include temporaryDownloadUrl")
        file_response = await self.http.get(
            temp_url,
            timeout_seconds=self.cfg.collector.knmi_download_timeout_seconds,
        )
        if file_response.status != 200:
            raise RuntimeError(f"KNMI file download returned {file_response.status}")

        blob_path = await asyncio.to_thread(self._store_blob, filename, file_response.body)
        file_sha256 = hashlib.sha256(file_response.body).hexdigest()
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="knmi_10m",
                event_type="dataset_file",
                payload={
                    "filename": filename,
                    "metadata": metadata,
                    "trigger": trigger,
                    "bytes": len(file_response.body),
                    "sha256": file_sha256,
                    "blob_path": str(blob_path),
                    "download_headers": file_response.headers,
                },
                request_started_wall_ns=file_response.request_started_wall_ns,
                response_received_wall_ns=file_response.response_received_wall_ns,
                received_wall_ns=file_response.response_received_wall_ns,
                received_mono_ns=file_response.response_received_mono_ns,
            )
        )

        observations = await asyncio.to_thread(self._parse_netcdf, file_response.body)
        for station in self.stations:
            parsed = observations.get(station.knmi_station_code)
            if parsed is None:
                await self.sink.append(
                    EventEnvelope.build(
                        run_id=self.run_id,
                        source="knmi_10m",
                        event_type="parse_error",
                        payload={
                            "filename": filename,
                            "station_code": station.knmi_station_code,
                            "error": "station_not_found_in_latest_file",
                            "available_station_codes": sorted(observations)[:100],
                        },
                        station_id=station.station_id,
                        received_wall_ns=file_response.response_received_wall_ns,
                        received_mono_ns=file_response.response_received_mono_ns,
                    )
                )
                continue
            obs_time = parsed.get("observation_time")
            obs_dt = datetime.fromisoformat(obs_time) if obs_time else None
            await self.sink.append(
                EventEnvelope.build(
                    run_id=self.run_id,
                    source="knmi_10m",
                    event_type="weather_observation",
                    payload={
                        "filename": filename,
                        **parsed,
                        "backfill": False,
                        "is_latest_in_response": True,
                    },
                    station_id=station.station_id,
                    request_started_wall_ns=file_response.request_started_wall_ns,
                    response_received_wall_ns=file_response.response_received_wall_ns,
                    provider_timestamp_ns=self._metadata_time_ns(metadata),
                    observation_timestamp_ns=datetime_to_ns(obs_dt),
                    received_wall_ns=file_response.response_received_wall_ns,
                    received_mono_ns=file_response.response_received_mono_ns,
                )
            )
        self._last_filename = filename

    async def _run_notification_service(self) -> None:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = self._build_mqtt_client(loop, queue)
        try:
            await asyncio.to_thread(self._connect_mqtt, client)
            client.loop_start()
            event_type, payload = await asyncio.wait_for(queue.get(), timeout=20)
            if event_type != "connected":
                raise RuntimeError(f"KNMI MQTT connection failed: {payload}")
            await self._record_notification_status("notification_connected", payload)

            while True:
                try:
                    event_type, payload = await asyncio.wait_for(
                        queue.get(), timeout=self.cfg.collector.knmi_poll_seconds
                    )
                except TimeoutError:
                    await self._poll_safely("reconciliation")
                    continue
                if event_type == "message":
                    try:
                        await self._handle_notification(payload)
                    except asyncio.CancelledError:
                        raise
                    except CriticalSinkError:
                        raise
                    except Exception as exc:
                        logger.exception("KNMI notification processing failure")
                        await self.sink.append(
                            EventEnvelope.build(
                                run_id=self.run_id,
                                source="knmi_10m",
                                event_type="notification_parse_error",
                                payload={"error": repr(exc)},
                            )
                        )
                elif event_type == "disconnected":
                    await self._record_notification_status("notification_disconnected", payload)
                elif event_type == "connected":
                    await self._record_notification_status("notification_reconnected", payload)
                elif event_type == "connect_error":
                    raise RuntimeError(f"KNMI MQTT connection failed: {payload}")
                elif event_type == "subscribed":
                    await self._record_notification_status("notification_subscribed", payload)
        finally:
            try:
                await asyncio.to_thread(client.disconnect)
            finally:
                client.loop_stop()

    def _build_mqtt_client(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> Any:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("Install KNMI support: pip install -e '.[knmi]'") from exc

        def publish(event_type: str, payload: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

        def on_connect(
            client: Any,
            userdata: Any,
            flags: Any,
            reason_code: Any,
            properties: Any = None,
        ) -> None:
            if getattr(reason_code, "is_failure", True):
                publish("connect_error", {"reason": str(reason_code)})
                return
            topic = f"dataplatform/file/v1/{self.DATASET}/{self.VERSION}/created"
            client.subscribe(topic, qos=1)
            publish(
                "connected",
                {
                    "reason": str(reason_code),
                    "session_present": bool(flags.session_present),
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
            publish("message", {"topic": message.topic, "payload": bytes(message.payload)})

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
            client_id=self.notification_client_id,
            protocol=mqtt.MQTTv5,
            transport="websockets",
        )
        client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.username_pw_set("token", self.notification_key)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect = on_connect
        client.on_subscribe = on_subscribe
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        return client

    def _connect_mqtt(self, client: Any) -> None:
        from paho.mqtt.packettypes import PacketTypes
        from paho.mqtt.properties import Properties

        properties = Properties(PacketTypes.CONNECT)
        properties.SessionExpiryInterval = 24 * 60 * 60
        client.connect(
            "mqtt.dataplatform.knmi.nl",
            port=443,
            keepalive=60,
            clean_start=False,
            properties=properties,
        )

    async def _handle_notification(self, message: dict[str, Any]) -> None:
        import orjson

        event = orjson.loads(message["payload"])
        if not isinstance(event, dict):
            raise ValueError("KNMI notification payload is not an object")
        data = event.get("data")
        if not isinstance(data, dict):
            raise ValueError("KNMI notification has no data object")
        if data.get("datasetName") != self.DATASET or str(data.get("datasetVersion")) != self.VERSION:
            raise ValueError("KNMI notification dataset/version mismatch")
        filename = str(data.get("filename") or "")
        if not filename or Path(filename).name != filename or not filename.endswith(".nc"):
            raise ValueError("KNMI notification filename is invalid")
        expected_url = (
            f"{self.BASE}/datasets/{self.DATASET}/versions/{self.VERSION}/files/{filename}/url"
        )
        if data.get("url") and str(data["url"]) != expected_url:
            raise ValueError("KNMI notification URL does not match the documented endpoint")
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="knmi_10m",
                event_type="file_notification",
                payload={
                    "topic": message["topic"],
                    "event_id": event.get("id"),
                    "event_time": event.get("time"),
                    "filename": filename,
                    "qos": 1,
                },
                provider_timestamp_ns=self._metadata_time_ns({"event_time": event.get("time")}),
            )
        )
        await self._download_filename(
            filename,
            {**data, "event_time": event.get("time")},
            trigger="notification",
        )

    async def _record_notification_status(self, event_type: str, payload: dict[str, Any]) -> None:
        await self.sink.append(
            EventEnvelope.build(
                run_id=self.run_id,
                source="knmi_10m",
                event_type=event_type,
                payload={**payload, "client_id": self.notification_client_id},
            )
        )

    def _store_blob(self, filename: str, body: bytes) -> Path:
        safe_name = Path(filename).name
        blob_dir = self.cfg.data_dir / "runs" / self.run_id / "blobs" / "knmi"
        blob_dir.mkdir(parents=True, exist_ok=True)
        destination = blob_dir / safe_name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(body)
            temporary.replace(destination)
        return destination

    @staticmethod
    def _metadata_time_ns(metadata: dict[str, Any]) -> int | None:
        for key in ("event_time", "created", "lastModified"):
            value = metadata.get(key)
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return datetime_to_ns(parsed)
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_netcdf(body: bytes) -> dict[int, dict[str, Any]]:
        try:
            import numpy as np
            import xarray as xr
        except ImportError as exc:
            raise RuntimeError("Install KNMI support: pip install -e '.[knmi]'") from exc

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(body)
            path = Path(tmp.name)
        try:


            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"numpy\.ndarray size changed, may indicate binary incompatibility.*",
                    category=RuntimeWarning,
                )
                ds = xr.open_dataset(path, decode_times=True)
            with ds:
                ds.load()
                station_name = KnmiCollector._find_name(
                    ds,
                    ["station", "station_code", "stn", "station_id", "wsi"],
                    dimensions_first=True,
                )
                time_name = KnmiCollector._find_name(
                    ds, ["time", "datetime", "observation_time"], dimensions_first=True
                )
                temp_name = KnmiCollector._find_name(
                    ds, ["ta", "temperature", "air_temperature", "t"]
                )
                dew_name = KnmiCollector._find_name(
                    ds, ["td", "dew_point_temperature", "dewpoint", "dew_point"], required=False
                )
                wind_speed_name = KnmiCollector._find_name(
                    ds, ["ff", "wind_speed", "windspeed"], required=False
                )
                wind_dir_name = KnmiCollector._find_name(
                    ds, ["dd", "wind_direction", "winddir"], required=False
                )
                pressure_name = KnmiCollector._find_name(
                    ds, ["pp", "air_pressure", "pressure", "p"], required=False
                )
                radiation_name = KnmiCollector._find_name(
                    ds,
                    ["qg", "global_radiation", "surface_downwelling_shortwave_flux"],
                    required=False,
                )

                station_values = np.asarray(ds[station_name].values).reshape(-1)

                times = np.asarray(ds[time_name].values).reshape(-1)
                latest_time = times[-1]
                if hasattr(latest_time, "astype"):
                    obs_time = np.datetime_as_string(latest_time, unit="s")
                else:
                    obs_time = str(latest_time)
                if not obs_time.endswith("Z") and "+" not in obs_time:
                    obs_time += "+00:00"
                obs_dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00")).astimezone(UTC)

                result: dict[int, dict[str, Any]] = {}
                for index, raw_station in enumerate(station_values):
                    code = KnmiCollector._station_code(raw_station)
                    if code is None:
                        continue
                    result[code] = {
                        "knmi_station_code": code,
                        "observation_time": obs_dt.isoformat(),
                        "temperature_c": KnmiCollector._value_at(
                            ds[temp_name], index, temp_name, quantity="temperature_c"
                        ),
                        "dewpoint_c": KnmiCollector._optional_value(
                            ds, dew_name, index, quantity="temperature_c"
                        ),
                        "wind_speed_kt": KnmiCollector._optional_value(
                            ds, wind_speed_name, index, quantity="wind_speed_kt"
                        ),
                        "wind_direction_deg": KnmiCollector._optional_value(
                            ds, wind_dir_name, index
                        ),
                        "pressure_hpa": KnmiCollector._optional_value(
                            ds, pressure_name, index, quantity="pressure_hpa"
                        ),
                        "global_radiation": KnmiCollector._optional_value(
                            ds, radiation_name, index
                        ),
                        "netcdf_variables": list(ds.data_vars),
                    }
                return result
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _find_name(
        ds: Any, candidates: list[str], required: bool = True, dimensions_first: bool = False
    ) -> str | None:
        names = (
            list(ds.dims) + list(ds.coords) + list(ds.data_vars)
            if dimensions_first
            else list(ds.data_vars) + list(ds.coords) + list(ds.dims)
        )
        lower = {name.lower(): name for name in names}
        for candidate in candidates:
            if candidate.lower() in lower:
                return lower[candidate.lower()]
        for name in names:
            lname = name.lower()
            if any(candidate.lower() in lname for candidate in candidates):
                return name
        if required:
            raise KeyError(f"Unable to find any of {candidates}; available names: {names}")
        return None

    @staticmethod
    def _station_code(raw: Any) -> int | None:
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None

        if "-" in text:
            segment = text.split("-")[-1]
            segment_digits = "".join(ch for ch in segment if ch.isdigit())
            if segment_digits:
                return int(segment_digits)
        return int(digits)

    @staticmethod
    def _value_at(
        array: Any,
        station_index: int,
        name: str,
        *,
        quantity: str | None = None,
    ) -> float | None:
        import numpy as np

        values = np.asarray(array.values)
        dims = list(array.dims)
        station_dim_index = 0
        for i, dim in enumerate(dims):
            if "station" in dim.lower() or dim.lower() in {"stn", "wsi"}:
                station_dim_index = i
                break
        slicer = [slice(None)] * values.ndim
        slicer[station_dim_index] = station_index
        selected = np.asarray(values[tuple(slicer)]).reshape(-1)
        if not selected.size:
            return None
        raw = selected[-1]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if np.isnan(value):
            return None
        units = str(array.attrs.get("units", "")).casefold().replace(" ", "")
        if quantity == "temperature_c" and units in {"k", "kelvin"}:
            value -= 273.15
        elif quantity == "wind_speed_kt" and units in {
            "m/s",
            "ms-1",
            "m.s-1",
            "m*s-1",
        }:
            value *= 1.9438444924406
        elif quantity == "pressure_hpa" and units in {"pa", "pascal", "pascals"}:
            value /= 100.0
        return value

    @staticmethod
    def _optional_value(
        ds: Any,
        name: str | None,
        station_index: int,
        *,
        quantity: str | None = None,
    ) -> float | None:
        return (
            None
            if name is None
            else KnmiCollector._value_at(
                ds[name], station_index, name, quantity=quantity
            )
        )
