from __future__ import annotations

import csv
import gzip
import json
import asyncio
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import weather_alpha.analysis.historical as historical
from weather_alpha.analysis.historical import (
    _build_historical_rows,
    _deduplicate_event_dates,
    _event_cache_covers,
    _event_date,
    _hourly_index,
    _latest_price_at,
    _merge_hourly,
)
from weather_alpha.analysis.historical_alpha import evaluate_historical_price_screen
from weather_alpha.analysis.live_costs import _ask_vwap
from weather_alpha.analysis.model import _bootstrap_mean_ci
from weather_alpha.analysis.ncei_history import (
    Observation,
    _build_station_pretraining_rows,
    _load_ghcnh_observations,
    parse_isd_inventory,
    parse_ncei_scaled_field,
    resolve_station_years,
)
from weather_alpha.analysis.resolution_ledger import (
    _canonical_bucket,
    _prediction_metrics,
)
from weather_alpha.config import load_config


def test_latest_price_is_strictly_point_in_time() -> None:
    history = [{"t": 100, "p": 0.2}, {"t": 200, "p": 0.8}]

    assert _latest_price_at(history, 150) == {"t": 100, "p": 0.2}
    assert _latest_price_at(history, 99) is None


def test_hourly_index_keeps_variables_aligned() -> None:
    indexed = _hourly_index(
        {"time": ["2026-07-20T09:00", "2026-07-20T10:00"], "temperature": [20, 21]}
    )

    assert indexed["2026-07-20T10:00"]["temperature"] == 21


def test_merge_hourly_extends_cache_and_replaces_duplicate_timestamp() -> None:
    merged = _merge_hourly(
        {
            "time": ["2026-07-20T09:00", "2026-07-20T10:00"],
            "temperature": [20, 21],
            "cloud": [10, 20],
        },
        {
            "time": ["2026-07-20T10:00", "2026-07-20T11:00"],
            "temperature": [21.5, 22],
        },
    )

    assert merged["time"] == [
        "2026-07-20T09:00",
        "2026-07-20T10:00",
        "2026-07-20T11:00",
    ]
    assert merged["temperature"] == [20, 21.5, 22]
    assert merged["cloud"] == [10, 20, None]


def test_event_cache_must_cover_requested_end_date() -> None:
    cached = [
        {"eventDate": "2026-04-01"},
        {"eventDate": "2026-07-22"},
    ]

    assert _event_cache_covers(cached, date(2026, 4, 14), date(2026, 7, 22))
    assert not _event_cache_covers(cached, date(2026, 4, 14), date(2026, 7, 26))


def test_event_date_falls_back_to_legacy_market_end_date() -> None:
    event = {
        "eventDate": None,
        "endDate": "2025-01-22T12:00:00Z",
        "closedTime": "2025-01-23T09:46:25Z",
    }

    assert _event_date(event) == date(2025, 1, 22)


def test_gamma_keyset_follows_opaque_cursor_without_offset(monkeypatch) -> None:
    station = load_config(project_root=".").stations["nyc"]
    calls: list[dict[str, object]] = []

    async def fake_request(*_args, **kwargs):
        calls.append(kwargs["params"])
        if len(calls) == 1:
            return {
                "events": [{"id": "1"}],
                "next_cursor": "opaque-cursor",
            }
        return {"events": [{"id": "2"}], "next_cursor": None}

    monkeypatch.setattr(historical, "_request_json", fake_request)
    events = asyncio.run(
        historical._download_station_events(
            object(),
            asyncio.Semaphore(1),
            station,
        )
    )

    assert [event["id"] for event in events] == ["1", "2"]
    assert "offset" not in calls[0]
    assert "after_cursor" not in calls[0]
    assert calls[1]["after_cursor"] == "opaque-cursor"


def test_duplicate_event_date_prefers_non_archived_market() -> None:
    events = [
        {
            "event_id": "493684",
            "event_slug": "arch-highest-temperature-in-los-angeles-on-may-19-2026",
            "event_date": "2026-05-19",
            "contracts": [{"label": "74°F or higher"}],
        },
        {
            "event_id": "503659",
            "event_slug": "highest-temperature-in-los-angeles-on-may-19-2026",
            "event_date": "2026-05-19",
            "contracts": [{"label": "74-75°F"}],
        },
    ]

    selected = _deduplicate_event_dates(events)

    assert len(selected) == 1
    assert selected[0]["event_id"] == "503659"


def test_resolution_metrics_measure_model_lead_without_calling_it_pnl() -> None:
    station = load_config(project_root=".").stations["nyc"]
    local_day = date(2026, 7, 20)
    zone = ZoneInfo(station.timezone)

    def snapshot(
        hour: int,
        model: dict[str, float],
        market: dict[str, float],
    ) -> dict[str, object]:
        at = datetime.combine(local_day, datetime.min.time(), tzinfo=zone)
        at += timedelta(hours=hour)
        return {
            "timestamp_ns": int(at.astimezone(UTC).timestamp() * 1e9),
            "timestamp_utc": at.astimezone(UTC),
            "local_hour": float(hour),
            "source_winner": "80Â°F",
            "model_distribution": model,
            "market_distribution": market,
            "market_price_history_only": True,
            "market_price_max_age_seconds": 30.0,
        }

    metrics = _prediction_metrics(
        station,
        {
            "event_date": local_day.isoformat(),
            "winner": "80°F",
            "end_time": "2026-07-21T10:00:00Z",
        },
        [
            snapshot(9, {"79°F": 0.6, "80°F": 0.4}, {"79°F": 0.7, "80°F": 0.3}),
            snapshot(12, {"79°F": 0.4, "80°F": 0.6}, {"79°F": 0.6, "80°F": 0.4}),
            snapshot(15, {"79°F": 0.2, "80°F": 0.8}, {"79°F": 0.4, "80°F": 0.6}),
        ],
    )

    assert _canonical_bucket("80Â°F") == "80°F"
    assert metrics["first_model_top_correct"] is False
    assert metrics["model_ever_top_correct"] is True
    assert metrics["model_lead_vs_market_first_correct_hours"] == 3.0
    assert metrics["first_model_two_confirmation_correct_time_utc"]
    assert metrics["first_nonoracle_disagreement_signal_bucket"] == "80°F"
    assert metrics["first_nonoracle_disagreement_signal_correct"] is True


def test_live_cost_vwap_walks_real_ask_depth() -> None:
    levels = [{"price": "0.20", "size": "4"}, {"price": "0.25", "size": "10"}]

    assert _ask_vwap(levels, 10) == 0.23
    assert _ask_vwap(levels, 20) is None


def test_bootstrap_interval_is_invariant_to_row_order() -> None:
    values = [-0.8, 0.2, 0.0, 0.7, -0.1]

    assert _bootstrap_mean_ci(values) == _bootstrap_mean_ci(list(reversed(values)))


def test_ncei_inventory_resolves_exact_icao_and_year() -> None:
    inventory = (
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON",'
        '"ELEV(M)","BEGIN","END"\n'
        '"062400","99999","SCHIPHOL","NL","","EHAM","+52.309","+004.764",'
        '"-0003.4","19320503","20251231"\n'
    )

    records = parse_isd_inventory(inventory, ["EHAM"])
    mapping = resolve_station_years(records, ["EHAM"], 2010, 2025)

    assert mapping[("EHAM", 2010)].station_number == "06240099999"
    assert mapping[("EHAM", 2025)].name == "SCHIPHOL"


def test_ncei_temperature_parser_enforces_quality_and_missing_value() -> None:
    assert parse_ncei_scaled_field("+0123,1", missing=9999) == (12.3, "1")
    assert parse_ncei_scaled_field("+0123,3", missing=9999) == (None, "3")
    assert parse_ncei_scaled_field("+9999,1", missing=9999) == (None, "1")


def test_ghcnh_parser_keeps_only_acceptable_temperature_quality(tmp_path) -> None:
    path = tmp_path / "station.psv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "STATION|DATE|temperature|temperature_Quality_Code|temperature_Report_Type\n"
            "USW00014732|2026-07-01T12:00:00|23.4|1|FM15\n"
            "USW00014732|2026-07-01T13:00:00|99.9|3|FM15\n"
        )

    parsed = _load_ghcnh_observations(path, timezone="America/New_York")
    observations = parsed["2026-07-01"]

    assert len(observations) == 1
    assert observations[0].temperature_c == 23.4


def test_ncei_pretraining_row_cannot_see_future_daily_max() -> None:
    station = load_config(project_root=".").stations["nyc"]
    local_day = date(2020, 7, 1)
    zone = ZoneInfo(station.timezone)

    def observation(hour: int, temperature: float) -> Observation:
        at = datetime.combine(local_day, datetime.min.time(), tzinfo=zone) + timedelta(
            hours=hour
        )
        return Observation(
            timestamp_ns=int(at.astimezone(UTC).timestamp() * 1e9),
            temperature_c=temperature,
            dewpoint_c=temperature - 4,
            sea_level_pressure_hpa=1012,
            wind_speed_m_s=3,
            report_type="FM-15",
            temperature_quality="1",
        )

    primary = [observation(hour, 15 + min(hour, 15)) for hour in range(24)]
    peers = [observation(hour, 14 + min(hour, 15)) for hour in range(24)]
    series = {
        station.station_id: {local_day.isoformat(): primary},
        **{
            peer: {local_day.isoformat(): peers}
            for peer in station.peer_station_ids
        },
    }

    rows, audit = _build_station_pretraining_rows(
        station,
        series,
        start_year=2020,
        end_year=2020,
        decision_hours=(9,),
        observation_lag_minutes=10,
        max_peer_age_minutes=120,
        minimum_primary_observations=12,
        maximum_primary_gap_minutes=180,
    )

    assert audit["complete_primary_days"] == 1
    assert len(rows) == 1
    assert rows[0]["temperature_c"] == 23
    assert rows[0]["ncei_running_max_c"] == 23
    assert rows[0]["ncei_final_max_c"] == 30
    assert rows[0]["evaluation_eligible"] is False


def test_historical_rows_use_wu_floor_and_resolved_winner(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    station = cfg.stations["nyc"]
    local_day = "2026-07-20"
    decision = datetime(2026, 7, 20, 9, tzinfo=ZoneInfo(station.timezone))
    before = int((decision.astimezone(UTC).timestamp() - 600) * 1e9)
    event = {
        "event_id": "1",
        "event_slug": "weather",
        "event_date": local_day,
        "winner": "80°F",
        "contracts": [
            {
                "label": "79°F or below",
                "yes_token_id": "low",
                "fee_rate": 0.05,
                "fee_exponent": 1,
                "fees_enabled": True,
            },
            {
                "label": "80°F",
                "yes_token_id": "winner",
                "fee_rate": 0.05,
                "fee_exponent": 1,
                "fees_enabled": True,
            },
        ],
    }
    rows, audit = _build_historical_rows(
        cfg,
        {station.station_id: [event]},
        {},
        {
            (station.station_id, "1"): {
                "history": {
                    "low": [{"t": int(decision.timestamp()) - 10, "p": 0.4}],
                    "winner": [{"t": int(decision.timestamp()) - 10, "p": 0.6}],
                }
            }
        },
        {
            (station.station_id, local_day): {
                "final_wu_max": 80,
                "observations": [
                    {
                        "timestamp_ns": before,
                        "timestamp": datetime.fromtimestamp(before / 1e9, tz=UTC).isoformat(),
                        "temperature_market_unit": 78,
                    }
                ],
            }
        },
        decision_hours=(9,),
        observation_lag_minutes=5,
    )

    assert audit["winner_wu_mismatches"] == 0
    assert rows[0]["resolution_running_max_market_unit"] == 78
    assert json.loads(rows[0]["market_ask_json"]) == {}
    assert rows[0]["market_price_history_only"] is True


def test_historical_screen_excludes_cheap_tail_and_uses_first_signal(tmp_path) -> None:
    cfg = load_config(project_root=".").model_copy(
        update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
    )
    validation = tmp_path / "validation.csv"
    fields = [
        "station_id",
        "test_date",
        "decision_mode",
        "received_wall_ns",
        "local_hour",
        "probability_calibrated",
        "market_blend_calibrated",
        "winning_bucket",
        "model_distribution_json",
        "historical_market_price_json",
        "market_price_history_only",
        "market_price_max_age_seconds",
        "market_quote_coverage",
        "historical_market_fee_rate_json",
        "historical_market_fee_exponent_json",
        "historical_market_fees_enabled_json",
    ]
    rows = [
        {
            "station_id": "KLGA",
            "test_date": "2026-07-20",
            "decision_mode": "historical_scheduled",
            "received_wall_ns": 1,
            "local_hour": 9,
            "probability_calibrated": True,
            "market_blend_calibrated": True,
            "winning_bucket": "80°F",
            "model_distribution_json": json.dumps({"tail": 0.9, "80°F": 0.7}),
            "historical_market_price_json": json.dumps({"tail": 0.009, "80°F": 0.4}),
            "market_price_history_only": True,
            "market_price_max_age_seconds": 60,
            "market_quote_coverage": 1.0,
            "historical_market_fee_rate_json": json.dumps({"80°F": 0.05}),
            "historical_market_fee_exponent_json": json.dumps({"80°F": 1}),
            "historical_market_fees_enabled_json": json.dumps({"80°F": True}),
        },
        {
            "station_id": "KLGA",
            "test_date": "2026-07-20",
            "decision_mode": "historical_scheduled",
            "received_wall_ns": 2,
            "local_hour": 12,
            "probability_calibrated": True,
            "market_blend_calibrated": True,
            "winning_bucket": "80°F",
            "model_distribution_json": json.dumps({"80°F": 0.99}),
            "historical_market_price_json": json.dumps({"80°F": 0.1}),
            "market_price_history_only": True,
            "market_price_max_age_seconds": 60,
            "market_quote_coverage": 1.0,
            "historical_market_fee_rate_json": "{}",
            "historical_market_fee_exponent_json": "{}",
            "historical_market_fees_enabled_json": "{}",
        },
    ]
    with validation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paths = evaluate_historical_price_screen(
        cfg, validation_path=validation, edge_threshold=0.06
    )
    with paths["trades"].open(newline="", encoding="utf-8") as handle:
        trades = list(csv.DictReader(handle))

    assert len(trades) == 1
    assert trades[0]["chosen_bucket"] == "80°F"
    assert trades[0]["received_wall_ns"] == "1"
