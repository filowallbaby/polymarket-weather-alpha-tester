from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import UTC, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import _canonical_bucket
from .station_maximum_v5 import _finite
from .station_specific_alpha import _load_csv, _read_json, _truth
from .station_tail_v4 import _sha256_file


EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "v6_modal_residual_to_market_preregistered"
MINIMUM_MODEL_PROBABILITY = 0.40
MINIMUM_WEATHER_MARKET_GAP = 0.15
WEATHER_TAKER_FEE_RATE = 0.05
FLAT_EXECUTION_BUFFER = 0.02
TARGET_CALENDAR_DAYS_PER_STATION = 20
MINIMUM_SETTLED_TRADES_FOR_STATION_GATE = 10
MINIMUM_PREDICTION_ROWS_PER_STATION_DAY = 80
MINIMUM_ACCEPTED_ROWS_PER_STATION_DAY = 40
MINIMUM_LOCAL_HOUR_SPAN = 20.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_081_000
EXPECTED_STATION_IDS = ("EGLC", "EHAM", "KLAX", "KLGA")


def deploy_v6_residual_experiment(
    cfg: AppConfig, *, deployed_at_ns: int | None = None
) -> dict[str, Path]:

    root = cfg.reports_dir / "paper_v6_residual_experiment"
    active_path = root / "active_experiment.json"
    if active_path.exists():
        active, protocol, deployment_dir = _load_active_experiment(cfg)
        return _experiment_paths(active_path, deployment_dir, protocol)

    v6_active_path = (
        cfg.reports_dir
        / "paper_shadow_station_maximum_v6"
        / "active_deployment.json"
    )
    if not v6_active_path.exists():
        raise FileNotFoundError(
            "Missing active V6 shadow deployment; deploy V6 before the experiment"
        )
    v6_active = json.loads(v6_active_path.read_text(encoding="utf-8"))
    v6_deployment_dir = Path(v6_active["deployment_directory"])
    v6_model_path = v6_deployment_dir / "model_bundle.pkl"
    v6_deployment_path = v6_deployment_dir / "deployment.json"
    if _sha256_file(v6_model_path) != v6_active["model_sha256"]:
        raise RuntimeError("Active V6 model hash mismatch")
    if _sha256_file(v6_deployment_path) != v6_active["deployment_sha256"]:
        raise RuntimeError("Active V6 deployment hash mismatch")

    deployed_at_ns = int(deployed_at_ns or time.time_ns())
    deployed_at = datetime.fromtimestamp(deployed_at_ns / 1e9, tz=UTC)
    station_windows: dict[str, dict[str, str]] = {}
    collection_end_candidates: list[datetime] = []
    stations_by_id = {
        station.station_id: station for station in cfg.stations.values()
    }
    missing = sorted(set(EXPECTED_STATION_IDS) - set(stations_by_id))
    if missing:
        raise RuntimeError(f"Missing experiment stations: {', '.join(missing)}")
    for station_id in EXPECTED_STATION_IDS:
        station = stations_by_id[station_id]
        zone = ZoneInfo(station.timezone)
        first_date = deployed_at.astimezone(zone).date() + timedelta(days=1)
        last_date = first_date + timedelta(
            days=TARGET_CALENDAR_DAYS_PER_STATION - 1
        )
        local_end = datetime.combine(
            last_date + timedelta(days=1), datetime_time.min, tzinfo=zone
        )
        collection_end_candidates.append(local_end.astimezone(UTC))
        station_windows[station_id] = {
            "timezone": station.timezone,
            "first_eligible_local_date": first_date.isoformat(),
            "last_eligible_local_date": last_date.isoformat(),
        }
    collection_end = max(collection_end_candidates) + timedelta(minutes=10)

    identity = hashlib.sha256(
        (
            f"{EXPERIMENT_NAME}|{deployed_at_ns}|"
            f"{v6_active['deployment_id']}|{v6_active['model_sha256']}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    experiment_id = f"{deployed_at:%Y%m%dT%H%M%SZ}-{identity}"
    deployment_dir = root / "deployments" / experiment_id
    protocol_path = deployment_dir / "protocol.json"
    protocol = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_name": EXPERIMENT_NAME,
        "deployed_at_ns": deployed_at_ns,
        "deployed_at_utc": deployed_at.isoformat(),
        "prospective_boundary": (
            "Only episodes received after deployment and on a complete future "
            "local date inside the frozen station window are eligible."
        ),
        "v6_deployment_id": v6_active["deployment_id"],
        "v6_deployment_directory": str(v6_deployment_dir.resolve()),
        "v6_deployment_sha256": v6_active["deployment_sha256"],
        "v6_model_sha256": v6_active["model_sha256"],
        "config_sha256": cfg.config_hash(),
        "code_sha256": _code_sha256(),
        "code_hash_scope": "all Python files below src/weather_alpha",
        "station_ids": list(EXPECTED_STATION_IDS),
        "station_windows": station_windows,
        "collection_end_utc": collection_end.isoformat(),
        "signal_rule": {
            "source": "first eligible V6 modal episode per station-day",
            "requires_new_external_information": True,
            "minimum_model_probability": MINIMUM_MODEL_PROBABILITY,
            "minimum_v6_minus_normalized_market_mid": (
                MINIMUM_WEATHER_MARKET_GAP
            ),
            "minimum_confirmations": 1,
            "minimum_persistence_seconds": 0,
            "maximum_entries_per_station_day": 1,
            "shares_per_entry": 1,
            "all_airports_remain_enabled": True,
        },
        "cost_rule": {
            "entry_price": "normalized decision-time market midpoint",
            "weather_taker_fee_rate": WEATHER_TAKER_FEE_RATE,
            "weather_taker_fee_formula": "price * (1 - price) * fee_rate",
            "flat_execution_buffer_per_share": FLAT_EXECUTION_BUFFER,
            "pnl_formula": (
                "winner_payout - midpoint - taker_fee - flat_execution_buffer"
            ),
        },
        "sample_rule": {
            "calendar_days_per_station": TARGET_CALENDAR_DAYS_PER_STATION,
            "expected_station_days": (
                TARGET_CALENDAR_DAYS_PER_STATION * len(EXPECTED_STATION_IDS)
            ),
            "no_early_stopping": True,
            "no_threshold_changes": True,
            "no_model_retraining": True,
            "no_station_removal": True,
            "no_extension_for_missing_signals": True,
        },
        "data_quality_rule": {
            "minimum_v6_prediction_rows_per_station_day": (
                MINIMUM_PREDICTION_ROWS_PER_STATION_DAY
            ),
            "minimum_accepted_v6_rows_per_station_day": (
                MINIMUM_ACCEPTED_ROWS_PER_STATION_DAY
            ),
            "minimum_local_hour_span": MINIMUM_LOCAL_HOUR_SPAN,
            "all_80_station_days_must_pass": True,
            "failure_is_technical_not_economic": True,
            "no_window_extension": True,
        },
        "decision_rule": {
            "global_gate": (
                "completed 80-station-day holdout and two-sided 95% cluster-"
                "bootstrap lower bound of mean daily net PnL > 0"
            ),
            "station_gate": (
                "20 resolved opportunity days, at least 10 settled trades, and "
                "familywise one-sided 98.75% bootstrap lower bounds > 0 for "
                "both opportunity-day net PnL and win-minus-midpoint value"
            ),
            "alpha_demonstrated_when": (
                "the global gate or at least one multiplicity-adjusted station "
                "gate passes"
            ),
            "failure_action": (
                "abandon the directional weather strategy if no gate passes"
            ),
            "technical_invalid_action": (
                "do not infer economic failure when any station-day lacks the "
                "preregistered data coverage"
            ),
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "global_unit": "local_date cluster across stations",
            "station_unit": "station local_date including zero-PnL no-signal days",
        },
        "live_orders_enabled": False,
    }
    _write_json_atomic(protocol_path, protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    existing_episode_ids = {
        str(row.get("episode_id") or "")
        for row in _load_csv(v6_deployment_dir / "update_alpha_episodes.csv")
        if row.get("episode_id")
    }
    _write_json_atomic(
        deployment_dir / "state.json",
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "processed_episode_ids": sorted(existing_episode_ids),
            "last_cycle_at_utc": None,
        },
    )
    _write_csv_atomic(deployment_dir / "episode_audit.csv", [])
    _write_csv_atomic(deployment_dir / "paper_trades.csv", [])
    _write_csv_atomic(deployment_dir / "opportunity_days.csv", [])
    (deployment_dir / "PROTOCOL.md").write_text(
        _protocol_markdown(protocol, protocol_sha256), encoding="utf-8"
    )
    _write_json_atomic(
        active_path,
        {
            "experiment_id": experiment_id,
            "deployment_directory": str(deployment_dir.resolve()),
            "protocol_sha256": protocol_sha256,
            "v6_deployment_id": v6_active["deployment_id"],
            "v6_model_sha256": v6_active["model_sha256"],
            "live_orders_enabled": False,
        },
    )
    return run_v6_residual_experiment_cycle(cfg)


def run_v6_residual_experiment_cycle(cfg: AppConfig) -> dict[str, Path]:

    root = cfg.reports_dir / "paper_v6_residual_experiment"
    active_path = root / "active_experiment.json"
    active, protocol, deployment_dir = _load_active_experiment(cfg)
    episodes_path = (
        Path(protocol["v6_deployment_directory"])
        / "update_alpha_episodes.csv"
    )
    audit_path = deployment_dir / "episode_audit.csv"
    trades_path = deployment_dir / "paper_trades.csv"
    opportunities_path = deployment_dir / "opportunity_days.csv"
    state_path = deployment_dir / "state.json"
    summary_path = deployment_dir / "summary.json"

    episodes = _load_csv(episodes_path)
    audit_rows = _load_csv(audit_path)
    trade_rows = _load_csv(trades_path)
    state = _read_json(state_path, {})
    processed = set(state.get("processed_episode_ids") or [])
    traded_days = {
        (str(row.get("station_id") or ""), str(row.get("local_date") or ""))
        for row in trade_rows
    }
    new_audit_rows = 0
    new_trades = 0
    ordered = sorted(
        episodes,
        key=lambda row: (
            _integer(row.get("episode_start_wall_ns")) or 0,
            str(row.get("episode_id") or ""),
        ),
    )
    for episode in ordered:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id or episode_id in processed:
            continue
        processed.add(episode_id)
        audit, trade = _evaluate_episode(
            episode, protocol, traded_days=traded_days
        )
        audit_rows.append(audit)
        new_audit_rows += 1
        if trade is not None:
            trade_rows.append(trade)
            traded_days.add((trade["station_id"], trade["local_date"]))
            new_trades += 1

    winners = _official_winners(cfg, protocol)
    _settle_trades(trade_rows, winners)
    coverage = _coverage_by_station_day(protocol)
    opportunity_rows = _opportunity_rows(winners, coverage)
    summary = _experiment_summary(protocol, trade_rows, opportunity_rows)
    summary.update(
        {
            "last_cycle_at_utc": datetime.now(UTC).isoformat(),
            "new_episode_audit_rows": new_audit_rows,
            "new_paper_trades": new_trades,
        }
    )

    _write_csv_atomic(audit_path, audit_rows)
    _write_csv_atomic(trades_path, trade_rows)
    _write_csv_atomic(opportunities_path, opportunity_rows)
    state.update(
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": protocol["experiment_id"],
            "processed_episode_ids": sorted(processed),
            "last_cycle_at_utc": summary["last_cycle_at_utc"],
            "last_new_episode_audit_rows": new_audit_rows,
            "last_new_paper_trades": new_trades,
        }
    )
    _write_json_atomic(state_path, state)
    _write_json_atomic(summary_path, summary)
    (deployment_dir / "STATUS.md").write_text(
        _status_markdown(summary), encoding="utf-8"
    )
    return _experiment_paths(active_path, deployment_dir, protocol)


def _evaluate_episode(
    episode: dict[str, Any],
    protocol: dict[str, Any],
    *,
    traded_days: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    station_id = str(episode.get("station_id") or "")
    local_date = str(episode.get("local_date") or "")
    model_probability = _finite(episode.get("initial_model_probability"))
    gap = _finite(episode.get("initial_weather_market_gap"))
    episode_ns = _integer(episode.get("episode_start_wall_ns"))
    window = protocol.get("station_windows", {}).get(station_id)
    reasons: list[str] = []
    if station_id not in set(protocol.get("station_ids") or []):
        reasons.append("station_not_preregistered")
    if episode_ns is None or episode_ns <= int(protocol["deployed_at_ns"]):
        reasons.append("before_prospective_boundary")
    if str(episode.get("evaluation_phase") or "") != "prospective":
        reasons.append("not_prospective")
    if str(episode.get("signal_variant") or "") != "model_modal_bucket":
        reasons.append("not_v6_modal_bucket")
    if int(_finite(episode.get("active_confirmation_count")) or 0) < 1:
        reasons.append("not_new_external_information")
    if window is None:
        reasons.append("missing_station_window")
    elif not (
        window["first_eligible_local_date"]
        <= local_date
        <= window["last_eligible_local_date"]
    ):
        reasons.append("outside_frozen_date_window")
    if model_probability is None or model_probability < float(
        protocol["signal_rule"]["minimum_model_probability"]
    ):
        reasons.append("model_probability_below_40pct")
    if gap is None or gap < float(
        protocol["signal_rule"][
            "minimum_v6_minus_normalized_market_mid"
        ]
    ):
        reasons.append("weather_market_gap_below_15pct")
    market_mid = (
        model_probability - gap
        if model_probability is not None and gap is not None
        else None
    )
    if market_mid is None or not 0.0 <= market_mid <= 1.0:
        reasons.append("invalid_normalized_market_mid")
    if not str(episode.get("target_bucket") or ""):
        reasons.append("missing_target_bucket")
    if (station_id, local_date) in traded_days:
        reasons.append("station_day_entry_already_taken")

    fee = (
        _weather_taker_fee(market_mid)
        if market_mid is not None and 0.0 <= market_mid <= 1.0
        else None
    )
    cost = (
        market_mid
        + float(protocol["cost_rule"]["flat_execution_buffer_per_share"])
        + float(fee)
        if market_mid is not None and fee is not None
        else None
    )
    if cost is not None and cost >= 1.0:
        reasons.append("assumed_cost_not_below_one")
    eligible = not reasons
    audit = {
        "experiment_id": protocol["experiment_id"],
        "episode_id": episode.get("episode_id"),
        "start_decision_id": episode.get("start_decision_id"),
        "station_id": station_id,
        "local_date": local_date,
        "episode_start_wall_ns": episode_ns,
        "trigger_source": episode.get("trigger_source"),
        "information_class": episode.get("information_class"),
        "target_bucket": episode.get("target_bucket"),
        "initial_model_probability": model_probability,
        "initial_weather_market_gap": gap,
        "normalized_market_mid": market_mid,
        "eligible": eligible,
        "rejection_reasons": "|".join(reasons),
        "paper_trade_opened": eligible,
        "outcome_used_for_selection": False,
        "live_orders_enabled": False,
    }
    if not eligible:
        return audit, None
    trade_id = hashlib.sha256(
        (
            f"{protocol['experiment_id']}|{station_id}|{local_date}|"
            f"{episode.get('episode_id')}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    trade = {
        "paper_trade_id": trade_id,
        "experiment_id": protocol["experiment_id"],
        "episode_id": episode.get("episode_id"),
        "start_decision_id": episode.get("start_decision_id"),
        "station_id": station_id,
        "local_date": local_date,
        "entry_received_wall_ns": episode_ns,
        "entry_received_utc": _iso_from_ns(episode_ns),
        "trigger_source": episode.get("trigger_source"),
        "information_class": episode.get("information_class"),
        "bucket": episode.get("target_bucket"),
        "model_probability": model_probability,
        "weather_market_gap": gap,
        "normalized_market_mid": market_mid,
        "fee_per_share": fee,
        "flat_execution_buffer_per_share": protocol["cost_rule"][
            "flat_execution_buffer_per_share"
        ],
        "assumed_cost_per_share": cost,
        "estimated_model_edge_after_assumed_cost": (
            model_probability - cost
        ),
        "requested_shares": 1,
        "status": "paper_open",
        "official_winning_bucket": "",
        "won": "",
        "realized_pnl_per_share": "",
        "realized_pnl_total": "",
        "outcome_used_for_selection": False,
        "live_order_sent": False,
    }
    return audit, trade


def _weather_taker_fee(price: float) -> float:
    return float(price) * (1.0 - float(price)) * WEATHER_TAKER_FEE_RATE


def _official_winners(
    cfg: AppConfig, protocol: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_csv(
        cfg.reports_dir / "resolutions" / "resolution_ledger.csv"
    ):
        station_id = str(row.get("station_id") or "")
        local_date = str(row.get("event_date") or "")
        window = protocol.get("station_windows", {}).get(station_id)
        if window is None or not (
            window["first_eligible_local_date"]
            <= local_date
            <= window["last_eligible_local_date"]
        ):
            continue
        winner = _canonical_bucket(row.get("official_winning_bucket"))
        if not winner:
            continue
        output[(station_id, local_date)] = {
            "station_id": station_id,
            "local_date": local_date,
            "official_winning_bucket": winner,
            "event_id": row.get("event_id"),
            "polymarket_url": row.get("polymarket_url"),
        }
    return output


def _settle_trades(
    trades: list[dict[str, Any]],
    winners: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for trade in trades:
        if str(trade.get("status") or "") == "paper_settled":
            continue
        result = winners.get(
            (
                str(trade.get("station_id") or ""),
                str(trade.get("local_date") or ""),
            )
        )
        if result is None:
            continue
        winner = _canonical_bucket(result["official_winning_bucket"])
        won = _canonical_bucket(trade.get("bucket")) == winner
        cost = float(trade["assumed_cost_per_share"])
        pnl = (1.0 if won else 0.0) - cost
        trade.update(
            {
                "status": "paper_settled",
                "official_winning_bucket": winner,
                "won": won,
                "realized_pnl_per_share": pnl,
                "realized_pnl_total": pnl,
            }
        )


def _coverage_by_station_day(
    protocol: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    prediction_path = (
        Path(protocol["v6_deployment_directory"])
        / "shadow_predictions.csv"
    )
    for row in _load_csv(prediction_path):
        station_id = str(row.get("station_id") or "")
        local_date = str(row.get("local_date") or "")
        window = protocol.get("station_windows", {}).get(station_id)
        if window is None or not (
            window["first_eligible_local_date"]
            <= local_date
            <= window["last_eligible_local_date"]
        ):
            continue
        if str(row.get("evaluation_phase") or "") != "prospective":
            continue
        grouped[(station_id, local_date)].append(row)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    quality = protocol["data_quality_rule"]
    for key, rows in grouped.items():
        hours = [
            value
            for row in rows
            if (value := _finite(row.get("local_hour"))) is not None
        ]
        accepted = sum(
            not str(row.get("rejection_reason") or "").strip() for row in rows
        )
        hour_span = max(hours) - min(hours) if hours else 0.0
        valid = bool(
            len(rows)
            >= int(quality["minimum_v6_prediction_rows_per_station_day"])
            and accepted
            >= int(quality["minimum_accepted_v6_rows_per_station_day"])
            and hour_span >= float(quality["minimum_local_hour_span"])
        )
        output[key] = {
            "v6_prediction_rows": len(rows),
            "accepted_v6_rows": accepted,
            "minimum_local_hour": min(hours) if hours else None,
            "maximum_local_hour": max(hours) if hours else None,
            "local_hour_span": hour_span,
            "decision_modes": len(
                {str(row.get("decision_mode") or "") for row in rows}
            ),
            "coverage_valid": valid,
        }
    return output


def _opportunity_rows(
    winners: dict[tuple[str, str], dict[str, Any]],
    coverage: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            **coverage.get(
                key,
                {
                    "v6_prediction_rows": 0,
                    "accepted_v6_rows": 0,
                    "minimum_local_hour": None,
                    "maximum_local_hour": None,
                    "local_hour_span": 0.0,
                    "decision_modes": 0,
                    "coverage_valid": False,
                },
            ),
            "resolved": True,
            "unit_of_independence": "station_id + local_date",
        }
        for key, row in sorted(winners.items())
    ]


def _experiment_summary(
    protocol: dict[str, Any],
    trades: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
) -> dict[str, Any]:
    station_ids = list(protocol["station_ids"])
    trades_by_day = {
        (str(row.get("station_id")), str(row.get("local_date"))): row
        for row in trades
        if str(row.get("status") or "") == "paper_settled"
    }
    opportunities_by_station: defaultdict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in opportunities:
        opportunities_by_station[str(row.get("station_id"))].append(row)

    by_station: list[dict[str, Any]] = []
    all_opportunity_values: list[dict[str, Any]] = []
    station_gates: list[str] = []
    for index, station_id in enumerate(station_ids):
        station_opportunities = sorted(
            opportunities_by_station.get(station_id, []),
            key=lambda row: str(row.get("local_date")),
        )
        valid_opportunities = [
            row
            for row in station_opportunities
            if _truth(row.get("coverage_valid"))
        ]
        station_trades = [
            row for row in trades if str(row.get("station_id")) == station_id
        ]
        settled_trades = [
            row
            for row in station_trades
            if str(row.get("status") or "") == "paper_settled"
        ]
        opportunity_pnl: list[float] = []
        for opportunity in valid_opportunities:
            key = (station_id, str(opportunity["local_date"]))
            trade = trades_by_day.get(key)
            pnl = (
                float(trade["realized_pnl_per_share"])
                if trade is not None
                else 0.0
            )
            opportunity_pnl.append(pnl)
            all_opportunity_values.append(
                {
                    "station_id": station_id,
                    "local_date": opportunity["local_date"],
                    "pnl": pnl,
                }
            )
        trade_value = [
            (1.0 if _truth(row.get("won")) else 0.0)
            - float(row["normalized_market_mid"])
            for row in settled_trades
        ]
        opportunity_interval = _bootstrap_interval(
            opportunity_pnl,
            confidence=0.975,
            seed=BOOTSTRAP_SEED + index,
        )
        station_value_interval = _bootstrap_interval(
            trade_value,
            confidence=0.975,
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        station_complete = (
            len(station_opportunities) == TARGET_CALENDAR_DAYS_PER_STATION
        )
        station_quality_valid = bool(
            station_complete
            and len(valid_opportunities) == TARGET_CALENDAR_DAYS_PER_STATION
        )
        station_gate = bool(
            station_quality_valid
            and len(settled_trades) >= MINIMUM_SETTLED_TRADES_FOR_STATION_GATE
            and opportunity_interval is not None
            and opportunity_interval["low"] > 0.0
            and station_value_interval is not None
            and station_value_interval["low"] > 0.0
        )
        if station_gate:
            station_gates.append(station_id)
        by_station.append(
            {
                "station_id": station_id,
                "expected_opportunity_days": TARGET_CALENDAR_DAYS_PER_STATION,
                "resolved_opportunity_days": len(station_opportunities),
                "technically_valid_opportunity_days": len(valid_opportunities),
                "window_complete": station_complete,
                "data_quality_gate_passed": station_quality_valid,
                "paper_trades": len(station_trades),
                "settled_trades": len(settled_trades),
                "wins": sum(_truth(row.get("won")) for row in settled_trades),
                "hit_rate": _mean(
                    [1.0 if _truth(row.get("won")) else 0.0 for row in settled_trades]
                ),
                "mean_model_probability": _mean(
                    [_finite(row.get("model_probability")) for row in settled_trades]
                ),
                "mean_market_mid": _mean(
                    [_finite(row.get("normalized_market_mid")) for row in settled_trades]
                ),
                "total_net_pnl": sum(opportunity_pnl),
                "mean_net_pnl_per_opportunity_day": _mean(opportunity_pnl),
                "opportunity_day_pnl_bootstrap_interval_97_5pct": (
                    opportunity_interval
                ),
                "mean_win_minus_midpoint": _mean(trade_value),
                "win_minus_midpoint_bootstrap_interval_97_5pct": (
                    station_value_interval
                ),
                "minimum_settled_trades_for_gate": (
                    MINIMUM_SETTLED_TRADES_FOR_STATION_GATE
                ),
                "multiplicity_adjusted_station_gate_passed": station_gate,
            }
        )

    daily_totals: defaultdict[str, float] = defaultdict(float)
    for row in all_opportunity_values:
        daily_totals[str(row["local_date"])] += float(row["pnl"])
    global_daily_interval = _bootstrap_interval(
        list(daily_totals.values()),
        confidence=0.95,
        seed=BOOTSTRAP_SEED + 1_000,
    )
    expected_station_days = TARGET_CALENDAR_DAYS_PER_STATION * len(station_ids)
    completed = len(opportunities) == expected_station_days and all(
        row["window_complete"] for row in by_station
    )
    technical_valid = bool(
        completed and all(row["data_quality_gate_passed"] for row in by_station)
    )
    global_gate = bool(
        technical_valid
        and global_daily_interval is not None
        and global_daily_interval["low"] > 0.0
    )
    alpha = bool(technical_valid and (global_gate or station_gates))
    if not completed:
        status = "collecting_frozen_holdout"
        decision = "locked_until_all_80_station_days_resolve"
    elif not technical_valid:
        status = "technical_data_quality_gate_failed"
        decision = "economic_verdict_invalid_due_to_missing_data"
    elif alpha:
        status = "alpha_gate_passed"
        decision = "eligible_for_separate_micro_stake_design"
    else:
        status = "alpha_gate_failed"
        decision = "abandon_directional_weather_strategy"
    return {
        "experiment_id": protocol["experiment_id"],
        "experiment_name": protocol["experiment_name"],
        "status": status,
        "decision": decision,
        "decision_locked_until_complete": not completed,
        "evaluation_complete": completed,
        "technical_data_quality_gate_passed": technical_valid,
        "expected_station_days": expected_station_days,
        "resolved_station_days": len(opportunities),
        "technically_valid_station_days": sum(
            int(row["technically_valid_opportunity_days"]) for row in by_station
        ),
        "paper_trades": len(trades),
        "settled_paper_trades": sum(
            str(row.get("status") or "") == "paper_settled" for row in trades
        ),
        "total_net_pnl": sum(
            float(row["pnl"]) for row in all_opportunity_values
        ),
        "global_mean_daily_net_pnl": _mean(list(daily_totals.values())),
        "global_daily_cluster_bootstrap_interval_95pct": global_daily_interval,
        "global_gate_passed": global_gate,
        "station_gates_passed": station_gates,
        "alpha_demonstrated": alpha,
        "by_station": by_station,
        "frozen_thresholds": protocol["signal_rule"],
        "frozen_costs": protocol["cost_rule"],
        "frozen_data_quality_rule": protocol["data_quality_rule"],
        "collection_end_utc": protocol["collection_end_utc"],
        "live_orders_enabled": False,
        "warnings": [
            "Interim results cannot change the frozen rule or stop the test early.",
            "Midpoint plus a fixed two-cent buffer is a research cost assumption, not a fill.",
            "No real order is ever sent by this experiment.",
            "A coverage failure makes the economic verdict invalid; it is not scored as no signal.",
        ],
    }


def _bootstrap_interval(
    values: list[float], *, confidence: float, seed: int
) -> dict[str, Any] | None:
    usable = [float(value) for value in values]
    if not usable:
        return None
    rng = random.Random(seed)
    size = len(usable)
    means = sorted(
        sum(usable[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = min(len(means) - 1, max(0, int(tail * len(means))))
    high_index = min(
        len(means) - 1, max(0, int((1.0 - tail) * len(means)) - 1)
    )
    return {
        "mean": sum(usable) / size,
        "low": means[low_index],
        "high": means[high_index],
        "confidence": confidence,
        "samples": BOOTSTRAP_SAMPLES,
        "n": size,
    }


def _load_active_experiment(
    cfg: AppConfig,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    active_path = (
        cfg.reports_dir
        / "paper_v6_residual_experiment"
        / "active_experiment.json"
    )
    if not active_path.exists():
        raise FileNotFoundError(
            "Missing V6 residual experiment; deploy it before running the cycle"
        )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    deployment_dir = Path(active["deployment_directory"])
    protocol_path = deployment_dir / "protocol.json"
    if _sha256_file(protocol_path) != active["protocol_sha256"]:
        raise RuntimeError("Frozen experiment protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise RuntimeError("Experiment protocol schema mismatch")
    if protocol.get("config_sha256") != cfg.config_hash():
        raise RuntimeError("Configuration changed after experiment preregistration")
    if protocol.get("code_sha256") != _code_sha256():
        raise RuntimeError("Source code changed after experiment preregistration")
    v6_dir = Path(protocol["v6_deployment_directory"])
    if _sha256_file(v6_dir / "model_bundle.pkl") != protocol["v6_model_sha256"]:
        raise RuntimeError("Frozen experiment V6 model hash mismatch")
    if (
        _sha256_file(v6_dir / "deployment.json")
        != protocol["v6_deployment_sha256"]
    ):
        raise RuntimeError("Frozen experiment V6 deployment hash mismatch")
    return active, protocol, deployment_dir


def _experiment_paths(
    active_path: Path, deployment_dir: Path, protocol: dict[str, Any]
) -> dict[str, Path]:
    return {
        "active": active_path,
        "protocol": deployment_dir / "protocol.json",
        "protocol_report": deployment_dir / "PROTOCOL.md",
        "audit": deployment_dir / "episode_audit.csv",
        "trades": deployment_dir / "paper_trades.csv",
        "opportunities": deployment_dir / "opportunity_days.csv",
        "state": deployment_dir / "state.json",
        "summary": deployment_dir / "summary.json",
        "status_report": deployment_dir / "STATUS.md",
        "v6_deployment": Path(protocol["v6_deployment_directory"]),
    }


def _protocol_markdown(protocol: dict[str, Any], sha256: str) -> str:
    windows = "\n".join(
        f"- {station_id}: {window['first_eligible_local_date']} through "
        f"{window['last_eligible_local_date']} ({window['timezone']})"
        for station_id, window in protocol["station_windows"].items()
    )
    return f"""# Frozen V6 residual-to-market experiment

Protocol SHA-256: `{sha256}`

This is a prospective paper-only holdout. No outcome was used to create an
entry, and no real order can be sent.

## Frozen signal

- First eligible V6 modal episode per station-day.
- New external information is required.
- V6 modal probability >= 40%.
- V6 minus normalized market midpoint >= 15 percentage points.
- One confirmation, zero persistence delay, one share, one entry per station-day.
- All four airports remain enabled.

## Frozen window

{windows}

Collection ends automatically after `{protocol['collection_end_utc']}`.

## Frozen economics and decision

PnL uses midpoint + the official 5% weather taker-fee formula + a flat $0.02
execution buffer. The decision stays locked until all 80 station-days resolve.
No threshold, model, airport, cost, sample window, or stopping rule may change.
If neither the global gate nor a multiplicity-adjusted airport gate passes, the
directional weather project is abandoned.

Every station-day must also contain at least 80 V6 rows, 40 accepted V6 rows,
and 20 hours of local coverage. A coverage failure invalidates the economic
verdict instead of being counted as a zero-signal day; the window is not extended.
"""


def _status_markdown(summary: dict[str, Any]) -> str:
    station_lines = "\n".join(
        f"- {row['station_id']}: {row['resolved_opportunity_days']}/20 days, "
        f"{row['settled_trades']} settled trades, PnL {row['total_net_pnl']:.4f}"
        for row in summary["by_station"]
    )
    return f"""# V6 residual experiment status

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Resolved station-days: {summary['resolved_station_days']}/80
- Technically valid station-days: {summary['technically_valid_station_days']}/80
- Paper trades: {summary['paper_trades']}
- Net paper PnL: {summary['total_net_pnl']:.4f}
- Alpha demonstrated: {summary['alpha_demonstrated']}
- Live orders enabled: {summary['live_orders_enabled']}

{station_lines}

Interim results are descriptive only and cannot alter the preregistered test.
"""


def _integer(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _iso_from_ns(value: int | None) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value / 1e9, tz=UTC).isoformat()


def _mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()
