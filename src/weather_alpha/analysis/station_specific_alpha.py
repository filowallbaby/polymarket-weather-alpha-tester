from __future__ import annotations

import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig
from .model import _bootstrap_mean_ci
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import (
    _canonical_bucket,
    _load_validation_rows,
    _probabilities,
    _row_key,
    _timestamp_ns,
)

STATION_POLICIES: dict[str, dict[str, Any]] = {
    "EHAM": {
        "policy_id": "eham_late_weather_disagreement_v1",
        "candidate_source": "weather",
        "relationship": "disagree",
        "target_local_hour": 15,
        "latest_local_minute": 45,
        "minimum_weather_top_probability": 0.40,
        "minimum_market_top_probability": 0.0,
        "minimum_source_advantage": 0.04,
        "minimum_price_proxy": 0.02,
        "maximum_price_proxy": 0.95,
        "fixed_cost_per_share": 0.015,
        "rationale": (
            "At Amsterdam the late weather-top disagreement retained positive "
            "synthetic value in both the early and recent diagnostic periods."
        ),
    },
    "EGLC": {
        "policy_id": "eglc_late_market_follow_v1",
        "candidate_source": "market",
        "relationship": "disagree",
        "target_local_hour": 15,
        "latest_local_minute": 45,
        "minimum_weather_top_probability": 0.0,
        "minimum_market_top_probability": 0.40,
        "minimum_source_advantage": 0.0,
        "minimum_price_proxy": 0.40,
        "maximum_price_proxy": 0.70,
        "fixed_cost_per_share": 0.020,
        "rationale": (
            "London changes bucket rapidly after noon. The market top is tested "
            "as a late candidate when the weather model still disagrees."
        ),
    },
    "KLAX": {
        "policy_id": "klax_late_high_confidence_convergence_v1",
        "candidate_source": "market",
        "relationship": "agree",
        "target_local_hour": 15,
        "latest_local_minute": 45,
        "minimum_weather_top_probability": 0.80,
        "minimum_market_top_probability": 0.0,
        "minimum_source_advantage": 0.0,
        "minimum_price_proxy": 0.02,
        "maximum_price_proxy": 0.95,
        "fixed_cost_per_share": 0.045,
        "rationale": (
            "LAX is usually near-final by 15:00. The policy accepts fewer, "
            "high-confidence model-market convergence cases."
        ),
    },
    "KLGA": {
        "policy_id": "klga_late_convergence_v1",
        "candidate_source": "market",
        "relationship": "agree",
        "target_local_hour": 15,
        "latest_local_minute": 45,
        "minimum_weather_top_probability": 0.60,
        "minimum_market_top_probability": 0.0,
        "minimum_source_advantage": 0.0,
        "minimum_price_proxy": 0.02,
        "maximum_price_proxy": 0.95,
        "fixed_cost_per_share": 0.020,
        "rationale": (
            "LaGuardia revises heavily from noon to 15:00. The policy waits "
            "for late model-market convergence instead of buying the early model."
        ),
    },
}


def evaluate_station_specific_policies(cfg: AppConfig) -> dict[str, Path]:

    snapshots = _load_historical_snapshots(cfg)
    trades: list[dict[str, Any]] = []
    summary_by_station: dict[str, Any] = {}
    for station_id, policy in STATION_POLICIES.items():
        station_snapshots = [
            row
            for row in snapshots
            if row["station_id"] == station_id
            and row["local_hour"] == policy["target_local_hour"]
        ]
        dates = sorted({str(row["local_date"]) for row in station_snapshots})
        split_index = int(len(dates) * 0.70)
        recent_dates = set(dates[split_index:])
        policy_trades = []
        for snapshot in station_snapshots:
            evaluation = evaluate_station_policy_snapshot(
                policy,
                snapshot["weather_distribution"],
                snapshot["market_distribution"],
            )
            if not evaluation["eligible"]:
                continue
            won = (
                _canonical_bucket(evaluation["candidate_bucket"])
                == snapshot["winning_bucket"]
            )
            synthetic_value = (
                (1.0 if won else 0.0)
                - float(evaluation["candidate_price_proxy"])
                - float(policy["fixed_cost_per_share"])
            )
            trade = {
                "station_id": station_id,
                "policy_id": policy["policy_id"],
                "local_date": snapshot["local_date"],
                "split": (
                    "recent_diagnostic"
                    if snapshot["local_date"] in recent_dates
                    else "development_diagnostic"
                ),
                "local_hour": snapshot["local_hour"],
                "candidate_bucket": evaluation["candidate_bucket"],
                "winning_bucket": snapshot["winning_bucket"],
                "weather_top_bucket": evaluation["weather_top_bucket"],
                "market_top_bucket": evaluation["market_top_bucket"],
                "candidate_weather_probability": evaluation[
                    "candidate_weather_probability"
                ],
                "candidate_price_proxy": evaluation[
                    "candidate_price_proxy"
                ],
                "source_advantage": evaluation["source_advantage"],
                "fixed_cost_per_share": policy["fixed_cost_per_share"],
                "won": won,
                "synthetic_value_per_share": synthetic_value,
                "economic_execution_evaluable": False,
            }
            trades.append(trade)
            policy_trades.append(trade)
        summary_by_station[station_id] = {
            "policy": policy,
            "development_diagnostic": _summarize_policy_trades(
                [
                    row
                    for row in policy_trades
                    if row["split"] == "development_diagnostic"
                ]
            ),
            "recent_diagnostic": _summarize_policy_trades(
                [
                    row
                    for row in policy_trades
                    if row["split"] == "recent_diagnostic"
                ]
            ),
            "all_available": _summarize_policy_trades(policy_trades),
        }

    output_dir = cfg.reports_dir / "station_specific_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "historical_policy_signals.csv"
    summary_path = output_dir / "historical_policy_summary.json"
    _write_csv_atomic(trades_path, trades)
    _write_json_atomic(
        summary_path,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "strategy": "station_specific_timing_and_candidate_source_v1",
            "station_policies": summary_by_station,
            "unit_of_independence": "station_id + local_date",
            "selection_warning": (
                "Policy shapes were chosen after inspecting the available "
                "history. Both historical segments are diagnostic; only rows "
                "after a new deployment timestamp are prospective."
            ),
            "price_assumption": (
                "Matched one-minute price proxy plus a fixed station cost; "
                "no historical ask, spread, depth or fill is used."
            ),
            "live_orders_enabled": False,
        },
    )
    return {"signals": trades_path, "summary": summary_path}


def deploy_station_specific_policies(cfg: AppConfig) -> dict[str, Path]:

    diagnostics = evaluate_station_specific_policies(cfg)
    summary = json.loads(diagnostics["summary"].read_text(encoding="utf-8"))
    deployed_at_ns = time.time_ns()
    locked: dict[str, dict[str, Any]] = {}
    for station_id, values in summary["station_policies"].items():
        policy = dict(values["policy"])
        all_available = values["all_available"]
        wins = int(all_available["wins"])
        signals = int(all_available["signals"])


        policy["locked_calibrated_probability"] = (
            (wins + 2.0) / (signals + 4.0)
        )
        policy["diagnostic_station_days"] = signals
        policy["diagnostic_wins"] = wins
        policy["enabled"] = True
        locked[station_id] = policy

    output_dir = cfg.reports_dir / "paper_trading_station_specific"
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_path = output_dir / "deployment.json"
    _write_json_atomic(
        deployment_path,
        {
            "strategy": "station_specific_synthetic_cost_paper_v1",
            "deployed_at_ns": deployed_at_ns,
            "deployed_at_utc": datetime.fromtimestamp(
                deployed_at_ns / 1e9,
                tz=UTC,
            ).isoformat(),
            "prospective_boundary": (
                "Only challenger audit rows received strictly after "
                "deployed_at_ns may open a station-specific paper position."
            ),
            "policies": locked,
            "maximum_entries_per_station_day": 1,
            "probability_lock": "Beta(2,2) shrinkage of the complete diagnostic sample",
            "source_diagnostics": str(diagnostics["summary"]),
            "synthetic_cost_only": True,
            "live_orders_enabled": False,
        },
    )
    state_path = output_dir / "state.json"
    _write_json_atomic(
        state_path,
        {
            "deployed_at_ns": deployed_at_ns,
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    _write_csv_atomic(output_dir / "paper_audit.csv", [])
    _write_csv_atomic(output_dir / "paper_trades.csv", [])
    return {
        "deployment": deployment_path,
        "diagnostics": diagnostics["summary"],
        "state": state_path,
    }


def run_station_specific_paper_cycle(cfg: AppConfig) -> dict[str, Path]:

    output_dir = cfg.reports_dir / "paper_trading_station_specific"
    deployment_path = output_dir / "deployment.json"
    state_path = output_dir / "state.json"
    source_path = (
        cfg.reports_dir
        / "paper_trading_challenger"
        / "paper_signal_audit.csv"
    )
    if not deployment_path.exists():
        raise FileNotFoundError(
            f"Missing {deployment_path}; run deploy-station-strategies"
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}; run paper-challenger-cycle"
        )
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    state = _read_json(
        state_path,
        {
            "deployed_at_ns": deployment["deployed_at_ns"],
            "processed_decision_ids": [],
            "traded_station_days": [],
        },
    )
    audit_path = output_dir / "paper_audit.csv"
    trades_path = output_dir / "paper_trades.csv"
    audit_rows = _load_csv(audit_path)
    trade_rows = _load_csv(trades_path)
    processed = set(state.get("processed_decision_ids") or [])
    traded_days = set(state.get("traded_station_days") or [])
    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    deployed_at_ns = int(deployment["deployed_at_ns"])

    for row in _load_csv(source_path):
        decision_id = str(row.get("decision_id") or "")
        timestamp_ns = _timestamp_ns(row.get("received_wall_ns"))
        station_id = str(row.get("station_id") or "")
        if (
            not decision_id
            or decision_id in processed
            or timestamp_ns is None
            or timestamp_ns <= deployed_at_ns
            or station_id not in deployment["policies"]
            or str(row.get("decision_mode") or "") != "scheduled"
        ):
            continue
        processed.add(decision_id)
        station = stations[station_id]
        policy = deployment["policies"][station_id]
        received = datetime.fromisoformat(
            str(row["received_utc"]).replace("Z", "+00:00")
        ).astimezone(ZoneInfo(station.timezone))
        local_date = received.date().isoformat()
        day_key = f"{station_id}|{local_date}"
        reasons: list[str] = []
        if received.hour != int(policy["target_local_hour"]):
            reasons.append("outside_station_entry_hour")
        if received.minute > int(policy["latest_local_minute"]):
            reasons.append("after_station_entry_window")

        weather = _probabilities(row.get("weather_distribution_json"))
        market = _probabilities(row.get("market_distribution_json"))
        evaluation = evaluate_station_policy_snapshot(policy, weather, market)
        reasons.extend(evaluation["rejection_reasons"])
        calibrated_probability = float(
            policy["locked_calibrated_probability"]
        )
        price_proxy = evaluation.get("candidate_price_proxy")
        synthetic_edge = (
            calibrated_probability
            - float(price_proxy)
            - float(policy["fixed_cost_per_share"])
            if price_proxy is not None
            else None
        )
        if synthetic_edge is None or synthetic_edge <= 0:
            reasons.append("nonpositive_locked_synthetic_edge")
        if day_key in traded_days:
            reasons.append("station_day_already_traded")
        reasons = sorted(set(reasons))
        can_open = not reasons
        audit_rows.append(
            {
                "decision_id": decision_id,
                "station_id": station_id,
                "local_date": local_date,
                "received_utc": row.get("received_utc"),
                "received_wall_ns": timestamp_ns,
                "policy_id": policy["policy_id"],
                "candidate_bucket": evaluation.get("candidate_bucket"),
                "candidate_price_proxy": price_proxy,
                "locked_calibrated_probability": calibrated_probability,
                "fixed_cost_per_share": policy["fixed_cost_per_share"],
                "estimated_synthetic_edge": synthetic_edge,
                "weather_top_bucket": evaluation.get("weather_top_bucket"),
                "market_top_bucket": evaluation.get("market_top_bucket"),
                "paper_trade_opened": can_open,
                "rejection_reasons": "|".join(reasons),
                "live_orders_enabled": False,
            }
        )
        if not can_open:
            continue
        synthetic_cost = float(price_proxy) + float(
            policy["fixed_cost_per_share"]
        )
        trade_rows.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "policy_id": policy["policy_id"],
                "entry_received_utc": row.get("received_utc"),
                "entry_received_wall_ns": timestamp_ns,
                "bucket": evaluation["candidate_bucket"],
                "price_proxy": price_proxy,
                "fixed_cost_per_share": policy["fixed_cost_per_share"],
                "synthetic_cost_per_share": synthetic_cost,
                "locked_calibrated_probability": calibrated_probability,
                "estimated_synthetic_edge": synthetic_edge,
                "winning_bucket": "",
                "won": "",
                "synthetic_value_per_share": "",
                "synthetic_value_10_shares": "",
                "settled": False,
                "live_orders_enabled": False,
            }
        )
        traded_days.add(day_key)

    _settle_trades(cfg, trade_rows)
    _write_csv_atomic(audit_path, audit_rows)
    _write_csv_atomic(trades_path, trade_rows)
    state["processed_decision_ids"] = sorted(processed)[-50_000:]
    state["traded_station_days"] = sorted(traded_days)
    _write_json_atomic(state_path, state)
    summary_path = output_dir / "paper_summary.json"
    settled = [row for row in trade_rows if _truth(row.get("settled"))]
    _write_json_atomic(
        summary_path,
        {
            "strategy": deployment["strategy"],
            "deployed_at_ns": deployed_at_ns,
            "policies": deployment["policies"],
            "processed_decisions": len(processed),
            "audit_rows": len(audit_rows),
            "paper_trades": len(trade_rows),
            "settled_trades": len(settled),
            "wins": sum(_truth(row.get("won")) for row in settled),
            "synthetic_value_per_share_total": sum(
                float(row["synthetic_value_per_share"])
                for row in settled
            ),
            "synthetic_cost_only": True,
            "live_orders_enabled": False,
        },
    )
    return {
        "audit": audit_path,
        "trades": trades_path,
        "summary": summary_path,
        "state": state_path,
    }


def evaluate_station_policy_snapshot(
    policy: dict[str, Any],
    weather_distribution: dict[str, float],
    market_distribution: dict[str, float],
) -> dict[str, Any]:

    reasons: list[str] = []
    if not weather_distribution:
        reasons.append("missing_weather_distribution")
    if not market_distribution:
        reasons.append("missing_market_distribution")
    if reasons:
        return {
            "eligible": False,
            "rejection_reasons": reasons,
            "candidate_bucket": "",
        }
    weather_top = max(weather_distribution, key=weather_distribution.get)
    market_top = max(market_distribution, key=market_distribution.get)
    relationship = str(policy["relationship"])
    if relationship == "agree" and weather_top != market_top:
        reasons.append("model_market_not_converged")
    if relationship == "disagree" and weather_top == market_top:
        reasons.append("model_market_not_disagreeing")

    source = str(policy["candidate_source"])
    candidate = weather_top if source == "weather" else market_top
    weather_probability = float(weather_distribution.get(candidate, 0.0))
    market_probability = float(market_distribution.get(candidate, 0.0))
    source_advantage = (
        weather_probability - market_probability
        if source == "weather"
        else market_probability - weather_probability
    )
    if (
        float(weather_distribution[weather_top])
        < float(policy["minimum_weather_top_probability"])
    ):
        reasons.append("weather_top_probability_below_station_rule")
    if (
        float(market_distribution[market_top])
        < float(policy["minimum_market_top_probability"])
    ):
        reasons.append("market_top_probability_below_station_rule")
    if (
        relationship == "disagree"
        and source_advantage
        < float(policy["minimum_source_advantage"])
    ):
        reasons.append("source_advantage_below_station_rule")
    if not (
        float(policy["minimum_price_proxy"])
        <= market_probability
        <= float(policy["maximum_price_proxy"])
    ):
        reasons.append("price_proxy_outside_station_rule")
    return {
        "eligible": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "candidate_bucket": candidate,
        "candidate_weather_probability": weather_probability,
        "candidate_price_proxy": market_probability,
        "source_advantage": source_advantage,
        "weather_top_bucket": weather_top,
        "weather_top_probability": weather_distribution[weather_top],
        "market_top_bucket": market_top,
        "market_top_probability": market_distribution[market_top],
    }


def _load_historical_snapshots(cfg: AppConfig) -> list[dict[str, Any]]:
    comparison_path = (
        cfg.reports_dir
        / "calibration_challenger"
        / "chronological_comparison.csv"
    )
    market_rows, _paths = _load_validation_rows(cfg)
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in _load_csv(comparison_path):
        if str(row.get("decision_mode") or "") != "historical_scheduled":
            continue
        try:
            local_hour = int(float(row["local_hour"]))
        except (KeyError, TypeError, ValueError):
            continue
        if local_hour not in {9, 12, 15}:
            continue
        weather = _probabilities(
            row.get("challenger_distribution_json")
        )
        validation = market_rows.get(_row_key(row), {})
        market = _probabilities(
            validation.get("historical_market_price_json")
        )
        timestamp_ns = _timestamp_ns(row.get("received_wall_ns"))
        station_id = str(row.get("station_id") or "")
        local_date = str(row.get("test_date") or "")
        if (
            not weather
            or not market
            or timestamp_ns is None
            or not station_id
            or not local_date
        ):
            continue
        key = (station_id, local_date, local_hour)
        candidate = {
            "station_id": station_id,
            "local_date": local_date,
            "local_hour": local_hour,
            "received_wall_ns": timestamp_ns,
            "winning_bucket": _canonical_bucket(
                row.get("winning_bucket")
            ),
            "weather_distribution": weather,
            "market_distribution": market,
        }
        previous = selected.get(key)
        if (
            previous is None
            or timestamp_ns < int(previous["received_wall_ns"])
        ):
            selected[key] = candidate
    return sorted(
        selected.values(),
        key=lambda row: (
            row["local_date"],
            row["station_id"],
            row["local_hour"],
        ),
    )


def _summarize_policy_trades(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [
        float(row["synthetic_value_per_share"]) for row in trades
    ]
    wins = sum(_truth(row.get("won")) for row in trades)
    signals = len(trades)
    return {
        "signals": signals,
        "wins": wins,
        "hit_rate": wins / signals if signals else None,
        "mean_price_proxy": (
            sum(float(row["candidate_price_proxy"]) for row in trades)
            / signals
            if signals
            else None
        ),
        "mean_synthetic_value_per_share": (
            sum(values) / signals if signals else None
        ),
        "mean_synthetic_value_ci95": (
            _bootstrap_mean_ci(values) if values else None
        ),
    }


def _settle_trades(
    cfg: AppConfig,
    trades: list[dict[str, Any]],
) -> None:
    ledger_path = (
        cfg.reports_dir / "resolutions" / "resolution_ledger.csv"
    )
    winners = {
        (
            str(row.get("station_id") or ""),
            str(row.get("event_date") or ""),
        ): _canonical_bucket(row.get("official_winning_bucket"))
        for row in _load_csv(ledger_path)
    }
    for trade in trades:
        if _truth(trade.get("settled")):
            continue
        winner = winners.get(
            (
                str(trade.get("station_id") or ""),
                str(trade.get("local_date") or ""),
            )
        )
        if not winner:
            continue
        won = _canonical_bucket(trade.get("bucket")) == winner
        value = (
            (1.0 if won else 0.0)
            - float(trade["synthetic_cost_per_share"])
        )
        trade.update(
            {
                "winning_bucket": winner,
                "won": won,
                "synthetic_value_per_share": value,
                "synthetic_value_10_shares": value * 10.0,
                "settled": True,
            }
        )


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []


    csv.field_size_limit(max(csv.field_size_limit(), 10 * 1024 * 1024))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}
