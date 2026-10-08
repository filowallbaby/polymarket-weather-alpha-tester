from __future__ import annotations

import asyncio
import json
import os
import shutil
import ssl
import subprocess
import threading
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from statistics import median

from .config import AppConfig
from .http_client import TimedHttpClient, TimedResponse


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


async def run_doctor(cfg: AppConfig) -> list[Check]:

    checks: list[Check] = []
    expected = {"EHAM", "KLGA", "KLAX", "EGLC"}
    actual = set(cfg.selected_station_ids)
    peer_ids = {
        peer.upper() for station in cfg.stations.values() for peer in station.peer_station_ids
    }
    all_awc_ids = actual | peer_ids
    checks.append(Check("stations", len(cfg.stations) >= 4, f"configured={len(cfg.stations)}"))
    checks.append(
        Check("baseline_station_ids", expected.issubset(actual), f"actual={sorted(actual)}")
    )
    checks.append(
        Check(
            "peer_station_config",
            all(bool(station.peer_station_ids) for station in cfg.stations.values()),
            f"configured_peer_ids={sorted(peer_ids)}",
        )
    )
    checks.append(
        Check(
            "data_dir",
            cfg.data_dir.exists() and os.access(cfg.data_dir, os.W_OK),
            str(cfg.data_dir),
        )
    )
    free_bytes = shutil.disk_usage(cfg.data_dir).free
    minimum_free_bytes = cfg.collector.minimum_free_disk_gb * 1_000_000_000
    checks.append(
        Check(
            "disk_space",
            free_bytes >= minimum_free_bytes,
            f"free={free_bytes / 1_000_000_000:.2f} GB; "
            f"minimum={cfg.collector.minimum_free_disk_gb:.2f} GB",
        )
    )
    checks.append(_paper_model_check(cfg))
    user_agent = os.getenv("HTTP_USER_AGENT", "").strip()
    user_agent_ok = bool(user_agent) and not any(
        marker in user_agent.lower() for marker in ("contact=unset", "replace@example.com")
    )
    checks.append(
        Check(
            "http_user_agent",
            user_agent_ok,
            "configured" if user_agent_ok else "set a descriptive, contactable HTTP_USER_AGENT",
            required=False,
        )
    )

    http = TimedHttpClient(cfg.collector.request_timeout_seconds)
    clock_responses: list[tuple[str, TimedResponse]] = []
    try:
        awc: TimedResponse | None = None
        try:
            awc = await http.get(
                "https://aviationweather.gov/api/data/metar",
                params={"ids": ",".join(sorted(all_awc_ids)), "format": "json", "hours": "2"},
            )
            checks.append(
                Check(
                    "aviation_weather",
                    awc.status in {200, 204},
                    f"HTTP {awc.status}, {len(awc.body)} bytes",
                )
            )
            clock_responses.append(("awc", awc))
        except Exception as exc:
            checks.append(_exception_check("aviation_weather", exc))

        sample_station = sorted(actual)[0]
        try:
            tgftp = await http.get(
                f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{sample_station}.TXT"
            )
            checks.append(
                Check(
                    "nws_tgftp",
                    tgftp.status == 200 and sample_station.encode() in tgftp.body,
                    f"HTTP {tgftp.status}, {len(tgftp.body)} bytes, sample={sample_station}",
                )
            )
            clock_responses.append(("nws", tgftp))
        except Exception as exc:
            checks.append(_exception_check("nws_tgftp", exc))

        try:
            gamma = await http.get(
                "https://gamma-api.polymarket.com/events",
                params={"limit": 1, "active": "true", "closed": "false"},
            )
            checks.append(
                Check(
                    "polymarket_gamma",
                    gamma.status == 200,
                    f"HTTP {gamma.status}, {len(gamma.body)} bytes",
                )
            )
            clock_responses.append(("gamma", gamma))
        except Exception as exc:
            checks.append(_exception_check("polymarket_gamma", exc))

        sample = next(iter(cfg.stations.values()))
        try:
            forecast = await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": sample.latitude,
                    "longitude": sample.longitude,
                    "hourly": "temperature_2m",
                    "forecast_days": 1,
                    "timezone": sample.timezone,
                },
            )
            checks.append(
                Check(
                    "forecast_provider",
                    forecast.status == 200,
                    f"HTTP {forecast.status}, {len(forecast.body)} bytes",
                )
            )
            clock_responses.append(("forecast", forecast))
        except Exception as exc:
            checks.append(_exception_check("forecast_provider", exc))

        checks.append(_clock_offset_check(clock_responses))

        if any(s.sources.knmi for s in cfg.stations.values()):
            key = (
                os.getenv("KNMI_OPEN_DATA_API_KEY", "").strip()
                or os.getenv("KNMI_API_KEY", "").strip()
            )
            checks.append(
                Check(
                    "knmi_open_data_key",
                    bool(key),
                    "configured" if key else "missing; EHAM will use NOAA paths only",
                    required=False,
                )
            )
            if key:
                try:
                    knmi = await http.get(
                        "https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
                        "10-minute-in-situ-meteorological-observations/versions/1.0/files",
                        params={"maxKeys": 1, "orderBy": "created", "sorting": "desc"},
                        headers={"Authorization": key},
                    )
                    files = knmi.json().get("files", []) if knmi.status == 200 else []
                    checks.append(
                        Check(
                            "knmi_open_data_api",
                            knmi.status == 200 and bool(files),
                            f"HTTP {knmi.status}, files={len(files)}",
                        )
                    )
                except Exception as exc:
                    checks.append(_exception_check("knmi_open_data_api", exc))

            notification_key = os.getenv("KNMI_NOTIFICATION_API_KEY", "").strip()
            checks.append(
                Check(
                    "knmi_notification_key",
                    bool(notification_key),
                    "configured" if notification_key else "missing; using 15-minute reconciliation",
                    required=False,
                )
            )
            if notification_key:
                checks.append(await _knmi_notification_check(notification_key))

        if any(s.sources.wis2 for s in cfg.stations.values()):
            checks.append(await _wis2_check())

        playwright = await _playwright_check()
        playwright.required = any(s.sources.wunderground for s in cfg.stations.values())
        checks.append(playwright)
        checks.append(_clock_service_check())
    finally:
        await http.close()
    return checks


def _exception_check(name: str, exc: Exception) -> Check:
    detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
    return Check(name, False, detail)


def _paper_model_check(cfg: AppConfig) -> Check:
    if not cfg.paper_trading.enabled:
        return Check("paper_model", True, "disabled by configuration", required=False)
    summary_path = cfg.reports_dir / "paper_trading" / "paper_model_summary.json"
    bundle_path = cfg.reports_dir / "paper_trading" / "paper_model_bundle.pkl"
    if not summary_path.exists() or not bundle_path.exists():
        return Check(
            "paper_model",
            False,
            "missing frozen model; run weather-alpha fit-paper-model",
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _exception_check("paper_model", exc)
    fitted = {
        str(row.get("station_id"))
        for row in summary.get("stations", [])
        if row.get("status") == "fitted" and row.get("probability_gate_ready") is True
    }
    config_matches = summary.get("config_hash") == cfg.config_hash()
    all_stations_ready = set(cfg.selected_station_ids).issubset(fitted)
    live_orders_disabled = summary.get("live_orders_enabled") is False
    ok = bool(config_matches and all_stations_ready and live_orders_disabled)
    return Check(
        "paper_model",
        ok,
        f"stations_ready={sorted(fitted)}, config_matches={config_matches}, "
        f"live_orders_enabled={summary.get('live_orders_enabled')}, "
        f"deployed_at={summary.get('deployed_at_utc')}",
    )


async def _knmi_notification_check(token: str) -> Check:
    return await asyncio.to_thread(_knmi_notification_check_sync, token)


def _knmi_notification_check_sync(token: str) -> Check:
    try:
        import paho.mqtt.client as mqtt
        from paho.mqtt.packettypes import PacketTypes
        from paho.mqtt.properties import Properties
    except ImportError:
        return Check(
            "knmi_notification_api",
            False,
            "paho-mqtt missing; install .[knmi]",
        )

    completed = threading.Event()
    result = {"connected": False, "subscribed": False, "reason": "timeout"}
    topic = (
        "dataplatform/file/v1/10-minute-in-situ-meteorological-observations/1.0/created"
    )

    def on_connect(
        client: object,
        userdata: object,
        flags: object,
        reason_code: object,
        properties: object = None,
    ) -> None:
        result["reason"] = str(reason_code)
        if not getattr(reason_code, "is_failure", True):
            result["connected"] = True
            client.subscribe(topic, qos=1)
        else:
            completed.set()

    def on_subscribe(
        client: object,
        userdata: object,
        mid: int,
        reason_codes: object,
        properties: object = None,
    ) -> None:
        codes = list(reason_codes)
        result["subscribed"] = bool(codes) and not any(
            getattr(code, "is_failure", True) for code in codes
        )
        completed.set()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"weather-alpha-doctor-{uuid.uuid4().hex}",
        protocol=mqtt.MQTTv5,
        transport="websockets",
    )
    client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.username_pw_set("token", token)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    properties = Properties(PacketTypes.CONNECT)
    properties.SessionExpiryInterval = 60
    try:
        client.connect(
            "mqtt.dataplatform.knmi.nl",
            port=443,
            keepalive=60,
            clean_start=True,
            properties=properties,
        )
        client.loop_start()
        completed.wait(12)
    except Exception as exc:
        return _exception_check("knmi_notification_api", exc)
    finally:
        try:
            client.disconnect()
        finally:
            client.loop_stop()
    ok = bool(result["connected"] and result["subscribed"])
    return Check(
        "knmi_notification_api",
        ok,
        f"connected={result['connected']}, subscribed={result['subscribed']}, "
        f"reason={result['reason']}",
    )


async def _wis2_check() -> Check:
    return await asyncio.to_thread(_wis2_check_sync)


def _wis2_check_sync() -> Check:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return Check("wis2_mqtt", False, "paho-mqtt missing")

    completed = threading.Event()
    result = {"connected": False, "subscribed": False, "reason": "timeout"}
    host = os.getenv("WIS2_BROKER_HOST", "globalbroker.meteo.fr").strip()
    try:
        port = int(os.getenv("WIS2_BROKER_PORT", "8883"))
    except ValueError:
        return Check("wis2_mqtt", False, "WIS2_BROKER_PORT is not an integer")
    username = os.getenv("WIS2_USERNAME", "everyone")
    password = os.getenv("WIS2_PASSWORD", "everyone")
    topic = "origin/a/wis2/+/data/core/S/A/#"

    def on_connect(
        client: object,
        userdata: object,
        flags: object,
        reason_code: object,
        properties: object = None,
    ) -> None:
        result["reason"] = str(reason_code)
        if not getattr(reason_code, "is_failure", True):
            result["connected"] = True
            client.subscribe(topic, qos=1)
        else:
            completed.set()

    def on_subscribe(
        client: object,
        userdata: object,
        mid: int,
        reason_codes: object,
        properties: object = None,
    ) -> None:
        codes = list(reason_codes)
        result["subscribed"] = bool(codes) and not any(
            getattr(code, "is_failure", True) for code in codes
        )
        completed.set()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"weather-alpha-wis2-doctor-{uuid.uuid4().hex}",
        protocol=mqtt.MQTTv5,
        transport="tcp",
    )
    client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    try:
        client.connect(host, port=port, keepalive=60)
        client.loop_start()
        completed.wait(12)
    except Exception as exc:
        return _exception_check("wis2_mqtt", exc)
    finally:
        try:
            client.disconnect()
        finally:
            client.loop_stop()
    ok = bool(result["connected"] and result["subscribed"])
    return Check(
        "wis2_mqtt",
        ok,
        f"host={host}:{port}, connected={result['connected']}, "
        f"subscribed={result['subscribed']}, reason={result['reason']}",
    )


async def _playwright_check() -> Check:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return Check(
            "playwright",
            False,
            "package missing; install .[wunderground] and run playwright install chromium",
        )

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            await browser.close()
    except Exception as exc:
        return _exception_check("playwright", exc)
    return Check("playwright", True, "package and Chromium launch OK")


def _http_clock_offset(response: TimedResponse) -> float | None:
    raw = response.headers.get("Date")
    if not raw:
        return None
    try:
        server_ns = int(parsedate_to_datetime(raw).timestamp() * 1e9)
    except (TypeError, ValueError, OverflowError):
        return None
    midpoint_ns = (response.request_started_wall_ns + response.response_received_wall_ns) // 2
    return (midpoint_ns - server_ns) / 1e9


def _clock_offset_check(responses: list[tuple[str, TimedResponse]]) -> Check:
    samples = [
        (name, offset)
        for name, response in responses
        if (offset := _http_clock_offset(response)) is not None
    ]
    if len(samples) < 2:
        return Check(
            "clock_offset_http",
            False,
            f"insufficient independent Date headers: samples={len(samples)}",
        )
    middle = median(offset for _, offset in samples)
    details = ", ".join(f"{name}={offset:+.3f}s" for name, offset in samples)
    return Check(
        "clock_offset_http",
        abs(middle) <= 3.0,
        f"median={middle:+.3f}s from {len(samples)} endpoints; {details}",
    )


def _clock_service_check() -> Check:
    if chronyc := shutil.which("chronyc"):
        result = subprocess.run(
            [chronyc, "tracking"], capture_output=True, text=True, timeout=5, check=False
        )
        detail = (result.stdout or result.stderr).strip().replace("\n", "; ")[:500]
        normalized = detail.lower()
        synchronized = result.returncode == 0 and not any(
            marker in normalized for marker in ("not synchronised", "not synchronized")
        )
        return Check("clock_sync_service", synchronized, detail, required=False)
    if w32tm := shutil.which("w32tm"):
        result = subprocess.run(
            [w32tm, "/query", "/status"], capture_output=True, text=True, timeout=5, check=False
        )
        detail = (result.stdout or result.stderr).strip().replace("\n", "; ")[:500]
        normalized = detail.lower()
        synchronized = result.returncode == 0 and not any(
            marker in normalized
            for marker in (
                "not synchronized",
                "not synchronised",
                "last successful sync time: unspecified",
                "source: local cmos clock",
            )
        )
        return Check("clock_sync_service", synchronized, detail, required=False)
    return Check(
        "clock_sync_service",
        False,
        "chronyc/w32tm not found; verify NTP synchronization manually",
        required=False,
    )
