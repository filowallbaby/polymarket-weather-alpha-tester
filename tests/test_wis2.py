from __future__ import annotations

import asyncio
import base64

import orjson

from weather_alpha.config import load_config
from weather_alpha.sources.wis2 import (
    Wis2Collector,
    decode_inline_content,
    extract_tac_reports,
)


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


def _collector(tmp_path) -> tuple[Wis2Collector, _Sink]:
    base = load_config(project_root=".")
    stations = {
        key: station.model_copy(
            update={"sources": station.sources.model_copy(update={"wis2": True})}
        )
        for key, station in base.stations.items()
    }
    cfg = base.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "reports_dir": tmp_path / "reports",
            "stations": stations,
        }
    )
    sink = _Sink()
    return Wis2Collector(cfg, "run", sink), sink


def test_extract_tac_reports_handles_wrapped_metar_and_speci() -> None:
    bulletin = (
        "SAUS01 TEST 211700\r\n"
        "METAR KLGA 211651Z 18008KT 10SM FEW050\r\n"
        "SCT250 30/20 A2992 RMK T03000200=\r\n"
        "SPECI KJFK 211658Z 19010KT 8SM 29/21 A2991=\r\n"
        "METAR EGLL 211650Z AUTO 24005KT CAVOK 24/12 Q1014="
    )

    reports = extract_tac_reports(bulletin, {"KLGA", "KJFK"})

    assert reports == [
        (
            "KLGA",
            "METAR KLGA 211651Z 18008KT 10SM FEW050 SCT250 30/20 A2992 RMK T03000200",
        ),
        ("KJFK", "SPECI KJFK 211658Z 19010KT 8SM 29/21 A2991"),
    ]


def test_decode_inline_content_rejects_invalid_base64() -> None:
    assert decode_inline_content({"encoding": "base64", "value": "%%%"}) == (
        None,
        "invalid_base64_content",
    )


def test_wis2_notification_creates_target_and_peer_observations(tmp_path) -> None:
    collector, sink = _collector(tmp_path)
    bulletin = (
        "METAR KLGA 211651Z 18008KT 10SM FEW050 30/20 A2992 RMK T03000200="
        "\nMETAR KJFK 211651Z 19009KT 10SM FEW060 29/20 A2991="
    )
    notification = {
        "id": "urn:uuid:test",
        "properties": {
            "datetime": "2026-07-21T16:52:03Z",
            "data_id": "wis2/test/metar",
            "wigos-station-identifier": "0-20000-0-KLGA",
        },
        "links": [
            {
                "rel": "canonical",
                "type": "text/plain",
                "href": "https://publisher.example/metar.txt",
            }
        ],
        "content": {
            "encoding": "base64",
            "value": base64.b64encode(bulletin.encode()).decode(),
        },
    }
    message = {
        "topic": "origin/a/wis2/test/data/core/S/A/A/A/00/TEST",
        "payload": orjson.dumps(notification),
        "payload_bytes": 100,
        "oversized": False,
        "qos": 1,
        "retain": False,
        "mid": 1,
        "received_wall_ns": 1_774_112_000_000_000_000,
        "received_mono_ns": 123,
    }

    asyncio.run(collector._handle_message(message))

    observations = [event for event in sink.events if "observation" in event.event_type]
    target = next(event for event in observations if event.source == "wis2_metar")
    peer = next(event for event in observations if event.source == "wis2_peer_metar")
    assert target.station_id == "KLGA"
    assert target.payload["temperature_c"] == 30.0
    assert target.payload["temperature_c_exact"] == 30.0
    assert target.provider_timestamp_ns is not None
    assert target.observation_timestamp_ns is not None
    assert peer.station_id == "KLGA"
    assert peer.payload["peer_station_id"] == "KJFK"
    assert sink.events[0].event_type == "notification"
    assert sink.events[0].payload["inline_content"] is True
    assert sink.events[0].payload["links"][0]["rel"] == "canonical"


def test_wis2_deduplicates_same_report(tmp_path) -> None:
    collector, sink = _collector(tmp_path)
    report = "METAR KLGA 211651Z 18008KT 10SM FEW050 30/20 A2992="
    notification = {
        "content": {
            "encoding": "base64",
            "value": base64.b64encode(report.encode()).decode(),
        }
    }
    message = {
        "topic": "topic",
        "payload": orjson.dumps(notification),
        "payload_bytes": 100,
        "oversized": False,
        "qos": 1,
        "retain": False,
        "mid": 1,
        "received_wall_ns": 1_774_112_000_000_000_000,
        "received_mono_ns": 123,
    }

    asyncio.run(collector._handle_message(message))
    asyncio.run(collector._handle_message(message))

    observations = [event for event in sink.events if event.event_type == "weather_observation"]
    assert len(observations) == 1
