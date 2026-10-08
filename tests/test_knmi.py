from __future__ import annotations

import asyncio
import warnings

import numpy as np
import orjson
import pytest
import xarray as xr

from weather_alpha.config import load_config
from weather_alpha.http_client import TimedResponse
from weather_alpha.sources.knmi import KnmiCollector


class _Sink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


def _collector(monkeypatch, tmp_path) -> tuple[KnmiCollector, _Sink]:
    monkeypatch.setenv("KNMI_OPEN_DATA_API_KEY", "open-data-token")
    monkeypatch.setenv("KNMI_NOTIFICATION_API_KEY", "notification-token")
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    sink = _Sink()
    return KnmiCollector(cfg, "run", sink, object()), sink


def test_knmi_prefers_explicit_open_data_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KNMI_API_KEY", "legacy-token")
    collector, _ = _collector(monkeypatch, tmp_path)

    assert collector.api_key == "open-data-token"
    assert collector.notification_key == "notification-token"
    assert collector.notification_client_id.startswith("weather-alpha-")


def test_knmi_notification_validates_documented_event_and_download_url(
    monkeypatch, tmp_path
) -> None:
    collector, sink = _collector(monkeypatch, tmp_path)
    filename = "KMDS__OPER_P___10M_OBS_L2_202607211600.nc"
    downloaded = []

    async def fake_download(name, metadata, *, trigger):
        downloaded.append((name, metadata, trigger))

    collector._download_filename = fake_download
    event = {
        "id": "event-id",
        "time": "2026-07-21T16:01:00Z",
        "data": {
            "datasetName": collector.DATASET,
            "datasetVersion": collector.VERSION,
            "filename": filename,
            "url": (
                f"{collector.BASE}/datasets/{collector.DATASET}/versions/"
                f"{collector.VERSION}/files/{filename}/url"
            ),
        },
    }

    asyncio.run(
        collector._handle_notification(
            {
                "topic": (
                    f"dataplatform/file/v1/{collector.DATASET}/{collector.VERSION}/created"
                ),
                "payload": orjson.dumps(event),
            }
        )
    )

    assert downloaded[0][0] == filename
    assert downloaded[0][2] == "notification"
    assert sink.events[0].event_type == "file_notification"
    assert sink.events[0].provider_timestamp_ns is not None


def test_knmi_notification_rejects_untrusted_download_url(monkeypatch, tmp_path) -> None:
    collector, _ = _collector(monkeypatch, tmp_path)
    event = {
        "data": {
            "datasetName": collector.DATASET,
            "datasetVersion": collector.VERSION,
            "filename": "latest.nc",
            "url": "https://example.invalid/latest.nc",
        }
    }

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            collector._handle_notification(
                {"topic": "documented-topic", "payload": orjson.dumps(event)}
            )
        )


def test_knmi_reconciliation_downloads_every_unseen_file(monkeypatch, tmp_path) -> None:
    collector, sink = _collector(monkeypatch, tmp_path)
    filenames = [
        "KMDS__OPER_P___10M_OBS_L2_202607211520.nc",
        "KMDS__OPER_P___10M_OBS_L2_202607211530.nc",
        "KMDS__OPER_P___10M_OBS_L2_202607211540.nc",
    ]

    class FakeHttp:
        def __init__(self) -> None:
            self.params = []

        async def get(self, url, *, params=None, **_kwargs):
            self.params.append(params)
            return TimedResponse(
                status=200,
                headers={},
                body=orjson.dumps(
                    {
                        "files": [{"filename": filename} for filename in filenames],
                        "isTruncated": False,
                    }
                ),
                request_started_wall_ns=1,
                request_started_mono_ns=1,
                response_received_wall_ns=2,
                response_received_mono_ns=2,
                url=url,
            )

    http = FakeHttp()
    collector.http = http
    collector._last_filename = filenames[0]
    downloaded = []

    async def fake_download(name, metadata, *, trigger):
        downloaded.append((name, trigger))
        collector._last_filename = name

    collector._download_filename = fake_download
    asyncio.run(collector.poll_once(trigger="reconciliation"))

    assert downloaded == [
        (filenames[1], "reconciliation"),
        (filenames[2], "reconciliation"),
    ]
    assert http.params[0] == {
        "maxKeys": 100,
        "orderBy": "filename",
        "sorting": "asc",
        "begin": filenames[0],
    }
    assert sink.events[0].event_type == "list_response"


def test_knmi_reconciliation_advances_truncated_pages(monkeypatch, tmp_path) -> None:
    collector, _ = _collector(monkeypatch, tmp_path)
    names = [
        "KMDS__OPER_P___10M_OBS_L2_202607211520.nc",
        "KMDS__OPER_P___10M_OBS_L2_202607211530.nc",
        "KMDS__OPER_P___10M_OBS_L2_202607211540.nc",
    ]

    class FakeHttp:
        def __init__(self) -> None:
            self.responses = [
                {"files": [{"filename": names[0]}, {"filename": names[1]}], "isTruncated": True},
                {
                    "files": [{"filename": names[1]}, {"filename": names[2]}],
                    "isTruncated": False,
                },
            ]

        async def get(self, url, **_kwargs):
            return TimedResponse(
                status=200,
                headers={},
                body=orjson.dumps(self.responses.pop(0)),
                request_started_wall_ns=1,
                request_started_mono_ns=1,
                response_received_wall_ns=2,
                response_received_mono_ns=2,
                url=url,
            )

    collector.http = FakeHttp()
    collector._last_filename = names[0]
    downloaded = []

    async def fake_download(name, metadata, *, trigger):
        downloaded.append(name)
        collector._last_filename = name

    collector._download_filename = fake_download
    asyncio.run(collector.poll_once(trigger="reconciliation"))

    assert downloaded == names[1:]


def test_knmi_netcdf_parser_reads_station_temperature(tmp_path) -> None:
    path = tmp_path / "sample.nc"
    dataset = xr.Dataset(
        {
            "ta": (
                ("time", "station"),
                np.array([[293.15]]),
                {"units": "K"},
            ),
            "td": (("time", "station"), np.array([[283.15]]), {"units": "K"}),
            "ff": (("time", "station"), np.array([[5.0]]), {"units": "m/s"}),
            "pp": (("time", "station"), np.array([[102_000.0]]), {"units": "Pa"}),
        },
        coords={
            "time": np.array(["2026-07-21T16:00:00"], dtype="datetime64[s]"),
            "station": np.array([6240]),
        },
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"numpy\.ndarray size changed, may indicate binary incompatibility.*",
            category=RuntimeWarning,
        )
        dataset.to_netcdf(path)

    parsed = KnmiCollector._parse_netcdf(path.read_bytes())

    assert parsed[6240]["temperature_c"] == pytest.approx(20.0)
    assert parsed[6240]["dewpoint_c"] == pytest.approx(10.0)
    assert parsed[6240]["wind_speed_kt"] == pytest.approx(9.719222462203)
    assert parsed[6240]["pressure_hpa"] == pytest.approx(1020.0)
    assert parsed[6240]["observation_time"] == "2026-07-21T16:00:00+00:00"
