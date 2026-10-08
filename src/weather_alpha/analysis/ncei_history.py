from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

from ..config import AppConfig, StationConfig
from ..parsing.buckets import c_to_f
from .meteorology import station_time_features


NCEI_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
NCEI_GLOBAL_HOURLY_URL = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_number}.csv"
)
GHCNH_YEAR_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/"
    "by-year/{year}/psv/GHCNh_{ghcnh_id}_{year}.psv"
)
GOOD_QUALITY_CODES = frozenset({"0", "1", "4", "5", "9"})
DEFAULT_DECISION_HOURS = (9, 12, 15)


@dataclass(frozen=True)
class NceiStationRecord:
    icao: str
    usaf: str
    wban: str
    name: str
    country: str
    state: str
    latitude: float
    longitude: float
    elevation_m: float | None
    begin: date
    end: date

    @property
    def station_number(self) -> str:
        return f"{self.usaf}{self.wban}"


@dataclass(frozen=True)
class Observation:
    timestamp_ns: int
    temperature_c: float
    dewpoint_c: float | None
    sea_level_pressure_hpa: float | None
    wind_speed_m_s: float | None
    report_type: str
    temperature_quality: str


async def backfill_ncei_global_hourly(
    cfg: AppConfig,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    concurrency: int = 4,
    refresh: bool = False,
    retries: int = 4,
) -> dict[str, Path]:

    if start_year < 1901 or end_year < start_year:
        raise ValueError("NCEI history years must satisfy 1901 <= start_year <= end_year")
    if concurrency < 1 or concurrency > 12:
        raise ValueError("concurrency must be between 1 and 12")
    if retries < 1 or retries > 10:
        raise ValueError("retries must be between 1 and 10")

    root = cfg.data_dir / "historical" / "ncei_global_hourly"
    raw_root = root / "raw"
    inventory_root = root / "inventory"
    raw_root.mkdir(parents=True, exist_ok=True)
    inventory_root.mkdir(parents=True, exist_ok=True)
    requested_icaos = sorted(
        {
            station_id
            for station in cfg.stations.values()
            for station_id in (station.station_id, *station.peer_station_ids)
        }
    )
    headers = {
        "User-Agent": os.getenv(
            "HTTP_USER_AGENT",
            "polymarket-weather-alpha-tester/0.1.26 research-contact-not-configured",
        )
    }
    timeout = aiohttp.ClientTimeout(total=180, connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(limit=max(2, concurrency), ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        inventory_text, inventory_path, inventory_hash = await _load_inventory(
            session, inventory_root, refresh=refresh, retries=retries
        )
        records = parse_isd_inventory(inventory_text, requested_icaos)
        station_years = resolve_station_years(records, requested_icaos, start_year, end_year)
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(icao: str, year: int, record: NceiStationRecord) -> dict[str, Any]:
            destination = raw_root / icao / f"{year}.csv.gz"
            metadata = raw_root / icao / f"{year}.meta.json"
            url = NCEI_GLOBAL_HOURLY_URL.format(
                year=year, station_number=record.station_number
            )
            async with semaphore:
                return await _download_year(
                    session,
                    url=url,
                    destination=destination,
                    metadata_path=metadata,
                    icao=icao,
                    year=year,
                    record=record,
                    refresh=refresh,
                    retries=retries,
                )

        tasks = [
            asyncio.create_task(fetch_one(icao, year, record))
            for (icao, year), record in sorted(station_years.items())
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    downloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key, result in zip(sorted(station_years), results, strict=True):
        if isinstance(result, BaseException):
            failures.append(
                {"station_id": key[0], "year": key[1], "error": str(result)}
            )
        else:
            downloads.append(result)
    report_dir = cfg.reports_dir / "ncei_history"
    report_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = report_dir / "ncei_station_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "inventory_url": NCEI_INVENTORY_URL,
                "inventory_cache": str(inventory_path),
                "inventory_sha256": inventory_hash,
                "start_year": start_year,
                "end_year": end_year,
                "stations": [
                    _record_dict(record)
                    for record in sorted(
                        {item for item in station_years.values()},
                        key=lambda item: (item.icao, item.begin, item.station_number),
                    )
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = report_dir / "ncei_download_summary.json"
    total_uncompressed = sum(int(item["uncompressed_bytes"]) for item in downloads)
    total_compressed = sum(int(item["compressed_bytes"]) for item in downloads)
    summary = {
        "source": "NOAA NCEI Global Hourly / Integrated Surface Database",
        "source_url_template": NCEI_GLOBAL_HOURLY_URL,
        "start_year": start_year,
        "end_year": end_year,
        "years": end_year - start_year + 1,
        "requested_station_ids": requested_icaos,
        "station_count": len(requested_icaos),
        "expected_files": len(station_years),
        "successful_files": len(downloads),
        "downloaded_files": sum(not item["cache_hit"] for item in downloads),
        "cache_hits": sum(bool(item["cache_hit"]) for item in downloads),
        "uncompressed_bytes": total_uncompressed,
        "compressed_bytes": total_compressed,
        "failures": failures,
        "quality_policy": {
            "temperature_codes_accepted": sorted(GOOD_QUALITY_CODES),
            "temperature_codes_rejected": ["2", "3", "6", "7"],
            "missing_temperature": "+9999",
        },
        "downloads": downloads,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"NCEI backfill failed for {len(failures)} station-years; see {summary_path}"
        )
    return {"mapping": mapping_path, "summary": summary_path}


async def backfill_ghcnh_overlap(
    cfg: AppConfig,
    *,
    year: int = 2026,
    concurrency: int = 4,
    refresh: bool = False,
    retries: int = 4,
) -> dict[str, Path]:

    if year < 2024 or year > datetime.now(UTC).year:
        raise ValueError("GHCNh overlap year must be between 2024 and the current year")
    mapping_path = cfg.reports_dir / "ncei_history" / "ncei_station_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError("Run backfill-ncei-history before GHCNh overlap discovery")
    station_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["stations"]
    by_icao = {str(item["station_id"]): item for item in station_mapping}
    requested_icaos = sorted(
        {
            station_id
            for station in cfg.stations.values()
            for station_id in (station.station_id, *station.peer_station_ids)
        }
    )
    ghcnh_ids: dict[str, str] = {}
    expected_missing: dict[str, str] = {}
    for icao in requested_icaos:
        metadata = by_icao.get(icao)
        if metadata is None:
            raise ValueError(f"No cached NCEI mapping for {icao}")
        if icao == "EHAM":
            expected_missing[icao] = "not published in the NCEI 2026 GHCNh by-year feed"
            continue
        if icao.startswith("K"):
            ghcnh_ids[icao] = f"USW000{str(metadata['wban']).zfill(5)}"
        elif icao.startswith("EG"):
            ghcnh_ids[icao] = f"UKI0000{icao}"
        elif icao.startswith("EH"):
            ghcnh_ids[icao] = f"NLI0000{icao}"
        else:
            raise ValueError(f"No deterministic GHCNh identifier rule for {icao}")

    root = cfg.data_dir / "historical" / "ghcnh_overlap" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": os.getenv(
            "HTTP_USER_AGENT",
            "polymarket-weather-alpha-tester/0.1.26 research-contact-not-configured",
        ),
        "Connection": "close",
    }
    timeout = aiohttp.ClientTimeout(total=180, connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(
        limit=max(2, concurrency), ttl_dns_cache=300, force_close=True
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:

        async def fetch_one(icao: str, ghcnh_id: str) -> dict[str, Any]:
            url = GHCNH_YEAR_URL.format(year=year, ghcnh_id=ghcnh_id)
            destination = root / icao / f"{year}.psv.gz"
            metadata_path = root / icao / f"{year}.meta.json"
            async with semaphore:
                return await _download_ghcnh_year(
                    session,
                    url=url,
                    destination=destination,
                    metadata_path=metadata_path,
                    icao=icao,
                    ghcnh_id=ghcnh_id,
                    year=year,
                    refresh=refresh,
                    retries=retries,
                )

        tasks = [
            asyncio.create_task(fetch_one(icao, ghcnh_id))
            for icao, ghcnh_id in sorted(ghcnh_ids.items())
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    downloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for (icao, _), result in zip(sorted(ghcnh_ids.items()), results, strict=True):
        if isinstance(result, BaseException):
            failures.append({"station_id": icao, "error": str(result)})
        else:
            downloads.append(result)
    report_dir = cfg.reports_dir / "ncei_history"
    summary_path = report_dir / "ghcnh_overlap_download_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": "NOAA NCEI GHCNh",
                "year": year,
                "requested_station_count": len(requested_icaos),
                "successful_files": len(downloads),
                "expected_missing": expected_missing,
                "failures": failures,
                "uncompressed_bytes": sum(
                    int(item["uncompressed_bytes"]) for item in downloads
                ),
                "compressed_bytes": sum(int(item["compressed_bytes"]) for item in downloads),
                "downloads": downloads,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"GHCNh overlap failed; see {summary_path}")
    return {"summary": summary_path}


def augment_historical_features_with_ghcnh_peers(
    cfg: AppConfig,
    *,
    year: int = 2026,
    max_peer_age_minutes: int = 120,
) -> dict[str, Path]:

    feature_path = cfg.reports_dir / "historical" / "predictive_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError("Run backfill-history before peer augmentation")
    with feature_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    raw_root = cfg.data_dir / "historical" / "ghcnh_overlap" / "raw"
    output_rows: list[dict[str, Any]] = []
    station_audits: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        ids = [station.station_id, *station.peer_station_ids]
        available_ids = [
            station_id
            for station_id in ids
            if (raw_root / station_id / f"{year}.psv.gz").exists()
        ]
        series = {
            station_id: _load_ghcnh_observations(
                raw_root / station_id / f"{year}.psv.gz", timezone=station.timezone
            )
            for station_id in available_ids
        }
        station_rows = [row for row in rows if row.get("station_id") == station.station_id]
        augmented = 0
        peer_counts: list[float] = []
        zone = ZoneInfo(station.timezone)
        for row in station_rows:
            local_date = str(row["local_date"])
            hour = int(float(row["local_hour"]))
            lag = int(float(row.get("observation_publication_safety_lag_minutes") or 5))
            decision = datetime.combine(date.fromisoformat(local_date), time(hour), tzinfo=zone)
            cutoff_ns = int(
                (decision.astimezone(UTC) - timedelta(minutes=lag)).timestamp() * 1e9
            )
            primary_current = _finite(row.get("temperature_c"))
            running_market = _finite(row.get("resolution_running_max_market_unit"))
            if primary_current is None or running_market is None:
                output_rows.append(row)
                continue
            primary_running_c = (
                running_market
                if station.market_unit == "C"
                else (running_market - 32.0) * 5.0 / 9.0
            )
            peer_series = {
                peer_id: series.get(peer_id, {}) for peer_id in station.peer_station_ids
            }
            features = _peer_features(
                station,
                local_date,
                cutoff_ns,
                primary_current_c=primary_current,
                primary_running_max_c=primary_running_c,
                series=peer_series,
                max_peer_age_minutes=max_peer_age_minutes,
            )
            row.update(features)
            row["peer_features_source"] = "ghcnh"
            row["peer_history_point_in_time"] = True
            peer_counts.append(float(features["peer_available_count"]))
            augmented += 1
            output_rows.append(row)
        station_audits.append(
            {
                "station_id": station.station_id,
                "rows": len(station_rows),
                "augmented_rows": augmented,
                "configured_peers": len(station.peer_station_ids),
                "available_ghcnh_peers": [
                    peer for peer in station.peer_station_ids if peer in series
                ],
                "mean_available_peers_at_decision": _mean(peer_counts),
                "rows_with_all_peers": sum(
                    count == len(station.peer_station_ids) for count in peer_counts
                ),
            }
        )
        bridge_rows.append(_ghcnh_wu_bridge_for_station(cfg, station, series))

    output_rows.sort(
        key=lambda row: (
            str(row.get("station_id")),
            str(row.get("local_date")),
            float(row.get("local_hour") or 0),
            int(float(row.get("received_wall_ns") or 0)),
        )
    )
    _write_csv_plain(feature_path, output_rows)
    report_dir = cfg.reports_dir / "ncei_history"
    summary_path = report_dir / "ghcnh_peer_augmentation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "year": year,
                "feature_file": str(feature_path),
                "max_peer_age_minutes": max_peer_age_minutes,
                "rows": len(output_rows),
                "stations": station_audits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bridge_path = report_dir / "ghcnh_wu_bridge_summary.json"
    bridge_path.write_text(
        json.dumps(
            {
                "purpose": "GHCNh-to-WU overlap on resolved market dates",
                "stations": bridge_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"features": feature_path, "summary": summary_path, "bridge": bridge_path}


def build_ncei_pretraining_dataset(
    cfg: AppConfig,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    decision_hours: tuple[int, ...] = DEFAULT_DECISION_HOURS,
    observation_lag_minutes: int = 10,
    max_peer_age_minutes: int = 120,
    minimum_primary_observations: int = 12,
    maximum_primary_gap_minutes: int = 180,
) -> dict[str, Path]:

    if not decision_hours or any(hour < 0 or hour > 23 for hour in decision_hours):
        raise ValueError("decision_hours must contain values from 0 through 23")
    if observation_lag_minutes < 0 or max_peer_age_minutes <= 0:
        raise ValueError("observation lag must be nonnegative and peer age must be positive")
    raw_root = cfg.data_dir / "historical" / "ncei_global_hourly" / "raw"
    expected = [
        raw_root / station_id / f"{year}.csv.gz"
        for station in cfg.stations.values()
        for station_id in (station.station_id, *station.peer_station_ids)
        for year in range(start_year, end_year + 1)
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} NCEI files; run backfill-ncei-history first: {preview}"
        )

    output_rows: list[dict[str, Any]] = []
    audit_stations: list[dict[str, Any]] = []
    for station in cfg.stations.values():
        ids = [station.station_id, *station.peer_station_ids]
        series = {
            station_id: _load_station_observations(
                raw_root / station_id,
                years=range(start_year, end_year + 1),
                timezone=station.timezone,
            )
            for station_id in ids
        }
        rows, audit = _build_station_pretraining_rows(
            station,
            series,
            start_year=start_year,
            end_year=end_year,
            decision_hours=decision_hours,
            observation_lag_minutes=observation_lag_minutes,
            max_peer_age_minutes=max_peer_age_minutes,
            minimum_primary_observations=minimum_primary_observations,
            maximum_primary_gap_minutes=maximum_primary_gap_minutes,
        )
        output_rows.extend(rows)
        audit_stations.append(audit)

    report_dir = cfg.reports_dir / "ncei_history"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = report_dir / "pretraining_features.csv.gz"
    _write_csv_gzip(dataset_path, output_rows)
    bridge = _wu_bridge_audit(cfg, output_rows)
    bridge_path = report_dir / "ncei_wu_bridge_summary.json"
    bridge_path.write_text(json.dumps(bridge, indent=2), encoding="utf-8")
    summary_path = report_dir / "ncei_pretraining_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": "NOAA NCEI Global Hourly",
                "target_semantics": "NCEI observed local-day maximum; not WU resolution truth",
                "evaluation_eligible": False,
                "start_year": start_year,
                "end_year": end_year,
                "decision_hours_local": list(decision_hours),
                "observation_lag_minutes": observation_lag_minutes,
                "max_peer_age_minutes": max_peer_age_minutes,
                "minimum_primary_observations": minimum_primary_observations,
                "maximum_primary_gap_minutes": maximum_primary_gap_minutes,
                "rows": len(output_rows),
                "stations": audit_stations,
                "bridge_report": str(bridge_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"dataset": dataset_path, "summary": summary_path, "bridge": bridge_path}


def parse_isd_inventory(
    text: str, requested_icaos: Iterable[str]
) -> dict[str, list[NceiStationRecord]]:
    wanted = {item.upper() for item in requested_icaos}
    output = {icao: [] for icao in wanted}
    for row in csv.DictReader(text.splitlines()):
        icao = str(row.get("ICAO") or "").strip().upper()
        if icao not in wanted:
            continue
        try:
            begin = datetime.strptime(str(row["BEGIN"]), "%Y%m%d").date()
            end = datetime.strptime(str(row["END"]), "%Y%m%d").date()
            latitude = float(row["LAT"])
            longitude = float(row["LON"])
        except (KeyError, TypeError, ValueError):
            continue
        elevation_raw = str(row.get("ELEV(M)") or "").strip()
        try:
            elevation = float(elevation_raw) if elevation_raw else None
        except ValueError:
            elevation = None
        output[icao].append(
            NceiStationRecord(
                icao=icao,
                usaf=str(row.get("USAF") or "").strip().zfill(6),
                wban=str(row.get("WBAN") or "").strip().zfill(5),
                name=str(row.get("STATION NAME") or "").strip(),
                country=str(row.get("CTRY") or "").strip(),
                state=str(row.get("STATE") or "").strip(),
                latitude=latitude,
                longitude=longitude,
                elevation_m=elevation,
                begin=begin,
                end=end,
            )
        )
    missing = sorted(icao for icao, records in output.items() if not records)
    if missing:
        raise ValueError(f"ICAO identifiers absent from ISD inventory: {', '.join(missing)}")
    return output


def resolve_station_years(
    records: dict[str, list[NceiStationRecord]],
    requested_icaos: Iterable[str],
    start_year: int,
    end_year: int,
) -> dict[tuple[str, int], NceiStationRecord]:
    output: dict[tuple[str, int], NceiStationRecord] = {}
    missing: list[str] = []
    for icao in sorted({item.upper() for item in requested_icaos}):
        candidates = records.get(icao, [])
        for year in range(start_year, end_year + 1):
            first = date(year, 1, 1)
            last = date(year, 12, 31)
            overlapping = [item for item in candidates if item.begin <= last and item.end >= first]
            if not overlapping:
                missing.append(f"{icao}:{year}")
                continue


            record = max(
                overlapping,
                key=lambda item: (
                    (min(item.end, last) - max(item.begin, first)).days,
                    item.begin,
                    item.station_number,
                ),
            )
            output[(icao, year)] = record
    if missing:
        raise ValueError(f"No ISD station segment for: {', '.join(missing[:20])}")
    return output


def parse_ncei_scaled_field(
    value: Any,
    *,
    missing: int,
    scale: float = 10.0,
    accepted_quality: frozenset[str] = GOOD_QUALITY_CODES,
) -> tuple[float | None, str | None]:
    parts = str(value or "").split(",")
    if len(parts) < 2:
        return None, None
    try:
        raw = int(parts[0])
    except ValueError:
        return None, parts[1]
    quality = parts[1]
    if abs(raw) == missing or quality not in accepted_quality:
        return None, quality
    return raw / scale, quality


def _parse_wind_speed(value: Any) -> float | None:
    parts = str(value or "").split(",")
    if len(parts) < 5:
        return None
    try:
        raw = int(parts[3])
    except ValueError:
        return None
    if raw == 9999 or parts[4] not in GOOD_QUALITY_CODES:
        return None
    return raw / 10.0


async def _load_inventory(
    session: aiohttp.ClientSession,
    root: Path,
    *,
    refresh: bool,
    retries: int,
) -> tuple[str, Path, str]:
    path = root / "isd-history.csv.gz"
    metadata_path = root / "isd-history.meta.json"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
        return text, path, hashlib.sha256(text.encode("utf-8")).hexdigest()
    body, response_headers = await _get_bytes(
        session, NCEI_INVENTORY_URL, retries=retries
    )
    text = body.decode("utf-8-sig")
    if "USAF" not in text[:200] or "ICAO" not in text[:200]:
        raise RuntimeError("NCEI station inventory has an unexpected header")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    with gzip.open(temp_path, "wb", compresslevel=6) as handle:
        handle.write(body)
    temp_path.replace(path)
    digest = hashlib.sha256(body).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "url": NCEI_INVENTORY_URL,
                "sha256": digest,
                "bytes": len(body),
                "etag": response_headers.get("ETag"),
                "last_modified": response_headers.get("Last-Modified"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return text, path, digest


async def _download_year(
    session: aiohttp.ClientSession,
    *,
    url: str,
    destination: Path,
    metadata_path: Path,
    icao: str,
    year: int,
    record: NceiStationRecord,
    refresh: bool,
    retries: int,
) -> dict[str, Any]:
    if destination.exists() and metadata_path.exists() and not refresh:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_cached_csv(destination, record.station_number)
        return {**metadata, "cache_hit": True}
    body, response_headers = await _get_bytes(session, url, retries=retries)
    header = body.splitlines()[0].decode("utf-8-sig", errors="replace") if body else ""
    if '"STATION"' not in header or '"DATE"' not in header or '"TMP"' not in header:
        raise RuntimeError(f"Unexpected NCEI CSV header for {icao} {year}")
    if record.station_number.encode() not in body[: max(4096, len(header) + 512)]:
        raise RuntimeError(f"NCEI CSV station mismatch for {icao} {year}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(temp_path, "wb", compresslevel=6) as handle:
        handle.write(body)
    _validate_cached_csv(temp_path, record.station_number)
    temp_path.replace(destination)
    metadata = {
        "station_id": icao,
        "year": year,
        "station_number": record.station_number,
        "source_url": url,
        "sha256_uncompressed": hashlib.sha256(body).hexdigest(),
        "uncompressed_bytes": len(body),
        "compressed_bytes": destination.stat().st_size,
        "etag": response_headers.get("ETag"),
        "last_modified": response_headers.get("Last-Modified"),
        "cache_hit": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


async def _download_ghcnh_year(
    session: aiohttp.ClientSession,
    *,
    url: str,
    destination: Path,
    metadata_path: Path,
    icao: str,
    ghcnh_id: str,
    year: int,
    refresh: bool,
    retries: int,
) -> dict[str, Any]:
    if destination.exists() and metadata_path.exists() and not refresh:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_cached_psv(destination, ghcnh_id)
        return {**metadata, "cache_hit": True}
    body, response_headers = await _get_bytes(session, url, retries=retries)
    header = body.splitlines()[0].decode("utf-8-sig", errors="replace") if body else ""
    if not header.startswith("STATION|") or "|DATE|" not in header or "|temperature|" not in header:
        raise RuntimeError(f"Unexpected GHCNh PSV header for {icao} {year}")
    if ghcnh_id.encode() not in body[: max(20_000, len(header) + 1000)]:
        raise RuntimeError(f"GHCNh station mismatch for {icao} {year}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(temp_path, "wb", compresslevel=6) as handle:
        handle.write(body)
    _validate_cached_psv(temp_path, ghcnh_id)
    temp_path.replace(destination)
    metadata = {
        "station_id": icao,
        "year": year,
        "ghcnh_id": ghcnh_id,
        "source_url": url,
        "sha256_uncompressed": hashlib.sha256(body).hexdigest(),
        "uncompressed_bytes": len(body),
        "compressed_bytes": destination.stat().st_size,
        "etag": response_headers.get("ETag"),
        "last_modified": response_headers.get("Last-Modified"),
        "cache_hit": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


async def _get_bytes(
    session: aiohttp.ClientSession, url: str, *, retries: int
) -> tuple[bytes, dict[str, str]]:
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read(), dict(response.headers)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                await asyncio.sleep(min(20.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"NCEI request failed after {retries} attempts: {url}: {last_error}")


def _validate_cached_csv(path: Path, station_number: str) -> None:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            first = handle.readline()
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"Invalid gzip cache: {path}") from exc
    if '"STATION"' not in header or station_number not in first:
        raise RuntimeError(f"Invalid or mismatched NCEI cache: {path}")


def _validate_cached_psv(path: Path, ghcnh_id: str) -> None:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            first = handle.readline()
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"Invalid gzip cache: {path}") from exc
    if not header.startswith("STATION|") or ghcnh_id not in first:
        raise RuntimeError(f"Invalid or mismatched GHCNh cache: {path}")


def _load_ghcnh_observations(
    path: Path, *, timezone: str
) -> dict[str, list[Observation]]:
    zone = ZoneInfo(timezone)
    by_date: dict[str, list[Observation]] = {}
    accepted_quality = GOOD_QUALITY_CODES | {""}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            temperature = _finite(row.get("temperature"))
            quality = str(row.get("temperature_Quality_Code") or "").strip()
            if temperature is None or quality not in accepted_quality:
                continue
            try:
                timestamp = datetime.fromisoformat(str(row["DATE"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            timestamp = timestamp.astimezone(UTC)
            report_type = str(row.get("temperature_Report_Type") or "").strip()
            observation = Observation(
                timestamp_ns=int(timestamp.timestamp() * 1e9),
                temperature_c=temperature,
                dewpoint_c=_finite(row.get("dew_point_temperature")),
                sea_level_pressure_hpa=_finite(row.get("sea_level_pressure")),
                wind_speed_m_s=_finite(row.get("wind_speed")),
                report_type=(f"FM-{report_type[2:]}" if report_type.startswith("FM") else report_type),
                temperature_quality=quality or "9",
            )
            local_date = timestamp.astimezone(zone).date().isoformat()
            by_date.setdefault(local_date, []).append(observation)
    for local_date, observations in by_date.items():
        by_timestamp: dict[int, Observation] = {}
        for observation in sorted(observations, key=_observation_preference):
            by_timestamp[observation.timestamp_ns] = observation
        by_date[local_date] = sorted(by_timestamp.values(), key=lambda item: item.timestamp_ns)
    return by_date


def _ghcnh_wu_bridge_for_station(
    cfg: AppConfig,
    station: StationConfig,
    series: dict[str, dict[str, list[Observation]]],
) -> dict[str, Any]:
    primary = series.get(station.station_id)
    if primary is None:
        return {
            "station_id": station.station_id,
            "overlap_days": 0,
            "status": "primary_station_absent_from_ghcnh",
            "mean_wu_minus_ghcnh_c": None,
            "mae_c": None,
            "same_rounded_market_value_rate": None,
        }
    differences: list[float] = []
    exact_market_bucket = 0
    overlap = 0
    wu_root = cfg.data_dir / "historical" / "wunderground" / station.station_id
    for path in sorted(wu_root.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        local_date = str(payload.get("local_date") or "")
        observations = primary.get(local_date, [])
        wu_value = _finite(payload.get("final_wu_max"))
        if not observations or wu_value is None:
            continue
        ghcnh_c = max(item.temperature_c for item in observations)
        wu_c = wu_value if station.market_unit == "C" else (wu_value - 32.0) * 5.0 / 9.0
        differences.append(wu_c - ghcnh_c)
        ghcnh_market = _to_market_unit(ghcnh_c, station.market_unit)
        exact_market_bucket += int(
            _round_half_up(ghcnh_market) == _round_half_up(wu_value)
        )
        overlap += 1
    return {
        "station_id": station.station_id,
        "overlap_days": overlap,
        "status": "evaluated" if overlap else "no_overlap",
        "mean_wu_minus_ghcnh_c": _mean(differences),
        "mae_c": _mean([abs(item) for item in differences]),
        "rmse_c": (
            math.sqrt(_mean([item * item for item in differences]) or 0.0)
            if differences
            else None
        ),
        "p95_absolute_error_c": _quantile(
            sorted(abs(item) for item in differences), 0.95
        ),
        "same_rounded_market_value_rate": (
            exact_market_bucket / overlap if overlap else None
        ),
    }


def _load_station_observations(
    directory: Path, *, years: Iterable[int], timezone: str
) -> dict[str, list[Observation]]:
    zone = ZoneInfo(timezone)
    by_date: dict[str, list[Observation]] = {}
    for year in years:
        path = directory / f"{year}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                temperature, quality = parse_ncei_scaled_field(
                    row.get("TMP"), missing=9999
                )
                if temperature is None or quality is None:
                    continue
                try:
                    timestamp = datetime.fromisoformat(str(row["DATE"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                timestamp = timestamp.astimezone(UTC)
                dewpoint, _ = parse_ncei_scaled_field(row.get("DEW"), missing=9999)
                pressure, _ = parse_ncei_scaled_field(row.get("SLP"), missing=99999)
                observation = Observation(
                    timestamp_ns=int(timestamp.timestamp() * 1_000_000_000),
                    temperature_c=temperature,
                    dewpoint_c=dewpoint,
                    sea_level_pressure_hpa=pressure,
                    wind_speed_m_s=_parse_wind_speed(row.get("WND")),
                    report_type=str(row.get("REPORT_TYPE") or "").strip(),
                    temperature_quality=quality,
                )
                local_date = timestamp.astimezone(zone).date().isoformat()
                by_date.setdefault(local_date, []).append(observation)
    for local_date, observations in by_date.items():
        by_timestamp: dict[int, Observation] = {}
        for observation in sorted(observations, key=_observation_preference):
            by_timestamp[observation.timestamp_ns] = observation
        by_date[local_date] = sorted(by_timestamp.values(), key=lambda item: item.timestamp_ns)
    return by_date


def _observation_preference(observation: Observation) -> tuple[int, int, str]:
    quality_rank = {"9": 0, "0": 1, "4": 2, "1": 3, "5": 4}.get(
        observation.temperature_quality, -1
    )
    report_rank = {"FM-12": 0, "FM-14": 1, "FM-15": 2, "FM-16": 3}.get(
        observation.report_type, 0
    )
    return observation.timestamp_ns, quality_rank, f"{report_rank:02d}"


def _build_station_pretraining_rows(
    station: StationConfig,
    series: dict[str, dict[str, list[Observation]]],
    *,
    start_year: int,
    end_year: int,
    decision_hours: tuple[int, ...],
    observation_lag_minutes: int,
    max_peer_age_minutes: int,
    minimum_primary_observations: int,
    maximum_primary_gap_minutes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timezone = ZoneInfo(station.timezone)
    primary_days = series[station.station_id]
    rows: list[dict[str, Any]] = []
    rejected_coverage = 0
    rejected_no_prior = 0
    complete_days = 0
    first = date(start_year, 1, 1)
    last = date(end_year, 12, 31)
    current = first
    while current <= last:
        local_date = current.isoformat()
        primary = primary_days.get(local_date, [])
        if not _complete_primary_day(
            primary,
            current,
            timezone,
            minimum_observations=minimum_primary_observations,
            maximum_gap_minutes=maximum_primary_gap_minutes,
        ):
            rejected_coverage += 1
            current += timedelta(days=1)
            continue
        complete_days += 1
        final_max_c = max(item.temperature_c for item in primary)
        for decision_hour in sorted(set(decision_hours)):
            decision = datetime.combine(current, time(decision_hour), tzinfo=timezone)
            cutoff = decision.astimezone(UTC) - timedelta(minutes=observation_lag_minutes)
            cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
            available = _available_at(primary, cutoff_ns)
            if not available:
                rejected_no_prior += 1
                continue
            latest = available[-1]
            running_max_c = max(item.temperature_c for item in available)
            running_max_obs = max(available, key=lambda item: item.temperature_c)
            row: dict[str, Any] = {
                "station_id": station.station_id,
                "city": station.city,
                "local_date": local_date,
                "local_hour": decision_hour,
                "decision_mode": "ncei_pretraining",
                "trigger_source": "ncei_scheduled",
                "source": "ncei_global_hourly",
                "target_source": "ncei_global_hourly",
                "evaluation_eligible": False,
                "historical_point_in_time": True,
                "day_history_complete": True,
                "continuity_eligible": True,
                "continuity_segment": f"ncei-{start_year}-{end_year}",
                "observation_publication_safety_lag_minutes": observation_lag_minutes,
                "observation_time": datetime.fromtimestamp(
                    latest.timestamp_ns / 1e9, tz=UTC
                ).isoformat(),
                "observation_key": (
                    f"ncei|{station.station_id}|{local_date}|{decision_hour:02d}"
                ),
                "received_wall_ns": int(decision.astimezone(UTC).timestamp() * 1e9),
                "temperature_c": latest.temperature_c,
                "temperature_market_unit": _to_market_unit(
                    latest.temperature_c, station.market_unit
                ),
                "running_max_market_unit": _to_market_unit(
                    running_max_c, station.market_unit
                ),
                "resolution_running_max_market_unit": _to_market_unit(
                    running_max_c, station.market_unit
                ),
                "final_temperature_target": _to_market_unit(
                    final_max_c, station.market_unit
                ),
                "final_wu_max": None,
                "ncei_final_max_c": final_max_c,
                "ncei_running_max_c": running_max_c,
                "ncei_remaining_increment_c": max(0.0, final_max_c - running_max_c),
                "historical_observation_count": len(available),
                "minutes_since_last_observation": (
                    cutoff_ns - latest.timestamp_ns
                )
                / 60_000_000_000,
                "hours_since_running_max": (
                    cutoff_ns - running_max_obs.timestamp_ns
                )
                / 3_600_000_000_000,
                "primary_dewpoint_c": latest.dewpoint_c,
                "primary_sea_level_pressure_hpa": latest.sea_level_pressure_hpa,
                "primary_wind_speed_m_s": latest.wind_speed_m_s,
            }
            for hours in (1, 3, 6):
                prior = _latest_at_or_before(
                    available, cutoff_ns - hours * 3_600_000_000_000
                )
                row[f"temperature_change_{hours}h"] = (
                    latest.temperature_c - prior.temperature_c if prior else None
                )
            row.update(station_time_features(decision, station))
            row["minutes_since_local_midnight"] = decision_hour * 60
            row.update(
                _peer_features(
                    station,
                    local_date,
                    cutoff_ns,
                    primary_current_c=latest.temperature_c,
                    primary_running_max_c=running_max_c,
                    series=series,
                    max_peer_age_minutes=max_peer_age_minutes,
                )
            )
            rows.append(row)
        current += timedelta(days=1)
    return rows, {
        "station_id": station.station_id,
        "peer_station_ids": station.peer_station_ids,
        "complete_primary_days": complete_days,
        "rows": len(rows),
        "rejected_days_for_primary_coverage": rejected_coverage,
        "rejected_decisions_without_prior_observation": rejected_no_prior,
        "first_date": min((row["local_date"] for row in rows), default=None),
        "last_date": max((row["local_date"] for row in rows), default=None),
        "mean_available_peers": _mean(
            [float(row["peer_available_count"]) for row in rows]
        ),
    }


def _complete_primary_day(
    observations: list[Observation],
    local_date: date,
    timezone: ZoneInfo,
    *,
    minimum_observations: int,
    maximum_gap_minutes: int,
) -> bool:
    if len(observations) < minimum_observations:
        return False
    local_times = [
        datetime.fromtimestamp(item.timestamp_ns / 1e9, tz=UTC).astimezone(timezone)
        for item in observations
    ]
    if local_times[0].date() != local_date or local_times[-1].date() != local_date:
        return False
    first_minute = local_times[0].hour * 60 + local_times[0].minute
    last_minute = local_times[-1].hour * 60 + local_times[-1].minute
    if first_minute > 180 or last_minute < 1260:
        return False
    gaps = [
        (right.timestamp() - left.timestamp()) / 60.0
        for left, right in zip(local_times, local_times[1:], strict=False)
    ]
    return not gaps or max(gaps) <= maximum_gap_minutes


def _peer_features(
    station: StationConfig,
    local_date: str,
    cutoff_ns: int,
    *,
    primary_current_c: float,
    primary_running_max_c: float,
    series: dict[str, dict[str, list[Observation]]],
    max_peer_age_minutes: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    temperatures: list[float] = []
    running_maxima: list[float] = []
    trends: list[float] = []
    ages: list[float] = []
    for peer_id in station.peer_station_ids:
        observations = _available_at(series.get(peer_id, {}).get(local_date, []), cutoff_ns)
        latest = observations[-1] if observations else None
        age = (
            (cutoff_ns - latest.timestamp_ns) / 60_000_000_000 if latest is not None else None
        )
        if latest is None or age is None or age > max_peer_age_minutes:
            output[f"peer_{peer_id}_available"] = 0.0
            continue
        peer_max = max(item.temperature_c for item in observations)
        previous = _latest_at_or_before(observations, cutoff_ns - 3_600_000_000_000)
        trend = latest.temperature_c - previous.temperature_c if previous else None
        output.update(
            {
                f"peer_{peer_id}_available": 1.0,
                f"peer_{peer_id}_temperature_c": latest.temperature_c,
                f"peer_{peer_id}_running_max_c": peer_max,
                f"peer_{peer_id}_age_minutes": age,
                f"peer_{peer_id}_temperature_delta_primary_c": (
                    latest.temperature_c - primary_current_c
                ),
                f"peer_{peer_id}_running_max_delta_primary_c": (
                    peer_max - primary_running_max_c
                ),
                f"peer_{peer_id}_temperature_change_1h": trend,
            }
        )
        temperatures.append(latest.temperature_c)
        running_maxima.append(peer_max)
        ages.append(age)
        if trend is not None:
            trends.append(trend)
    output.update(
        {
            "peer_available_count": float(len(temperatures)),
            "peer_temperature_mean_c": _mean(temperatures),
            "peer_temperature_max_c": max(temperatures, default=None),
            "peer_temperature_min_c": min(temperatures, default=None),
            "peer_temperature_spread_c": (
                max(temperatures) - min(temperatures) if temperatures else None
            ),
            "peer_running_max_mean_c": _mean(running_maxima),
            "peer_running_max_max_c": max(running_maxima, default=None),
            "peer_temperature_change_1h_mean": _mean(trends),
            "peer_age_minutes_mean": _mean(ages),
        }
    )
    return output


def _wu_bridge_audit(cfg: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ncei_daily: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["station_id"]), str(row["local_date"]))
        ncei_daily[key] = float(row["ncei_final_max_c"])
    by_station: list[dict[str, Any]] = []
    wu_root = cfg.data_dir / "historical" / "wunderground"
    for station in cfg.stations.values():
        differences: list[float] = []
        exact_market_bucket = 0
        overlap = 0
        for path in sorted((wu_root / station.station_id).glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            local_date = str(payload.get("local_date") or "")
            ncei_c = ncei_daily.get((station.station_id, local_date))
            wu_value = _finite(payload.get("final_wu_max"))
            if ncei_c is None or wu_value is None:
                continue
            wu_c = wu_value if station.market_unit == "C" else (wu_value - 32.0) * 5.0 / 9.0
            differences.append(wu_c - ncei_c)
            ncei_market = _to_market_unit(ncei_c, station.market_unit)
            exact_market_bucket += int(_round_half_up(ncei_market) == _round_half_up(wu_value))
            overlap += 1
        by_station.append(
            {
                "station_id": station.station_id,
                "overlap_days": overlap,
                "mean_wu_minus_ncei_c": _mean(differences),
                "mae_c": _mean([abs(item) for item in differences]),
                "rmse_c": (
                    math.sqrt(_mean([item * item for item in differences]) or 0.0)
                    if differences
                    else None
                ),
                "p95_absolute_error_c": _quantile(
                    sorted(abs(item) for item in differences), 0.95
                ),
                "same_rounded_market_value_rate": (
                    exact_market_bucket / overlap if overlap else None
                ),
            }
        )
    return {
        "purpose": "Measure domain shift; NCEI maxima are not silently relabelled as WU truth",
        "stations": by_station,
    }


def _available_at(observations: list[Observation], cutoff_ns: int) -> list[Observation]:
    timestamps = [item.timestamp_ns for item in observations]
    return observations[: bisect_right(timestamps, cutoff_ns)]


def _latest_at_or_before(
    observations: list[Observation], cutoff_ns: int
) -> Observation | None:
    timestamps = [item.timestamp_ns for item in observations]
    index = bisect_right(timestamps, cutoff_ns) - 1
    return observations[index] if index >= 0 else None


def _write_csv_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    temp_path = path.with_suffix(path.suffix + ".part")
    with gzip.open(temp_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _write_csv_plain(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    temp_path = path.with_suffix(path.suffix + ".part")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _record_dict(record: NceiStationRecord) -> dict[str, Any]:
    return {
        "station_id": record.icao,
        "usaf": record.usaf,
        "wban": record.wban,
        "station_number": record.station_number,
        "name": record.name,
        "country": record.country,
        "state": record.state,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "elevation_m": record.elevation_m,
        "begin": record.begin.isoformat(),
        "end": record.end.isoformat(),
    }


def _to_market_unit(value_c: float, unit: str) -> float:
    return value_c if unit == "C" else c_to_f(value_c)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(ordered: list[float], probability: float) -> float | None:
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)
