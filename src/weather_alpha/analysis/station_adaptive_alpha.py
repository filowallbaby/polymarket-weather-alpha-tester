from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import AppConfig
from .model import _bootstrap_mean_ci
from .paper_trading import _write_csv_atomic, _write_json_atomic
from .resolution_ledger import _probabilities, _timestamp_ns
from .station_specific_alpha import (
    _load_csv,
    _load_historical_snapshots,
    _read_json,
    _settle_trades,
    _truth,
    evaluate_station_policy_snapshot,
)


STATION_ADAPTIVE_LANES: dict[str, list[dict[str, Any]]] = {
    "EHAM": [
        {
            "policy_id": "eham_late_convergence_v2",
            "candidate_source": "weather",
            "relationship": "agree",
            "target_local_hour": 15,
            "latest_local_minute": 45,
            "minimum_weather_top_probability": 0.60,
            "minimum_market_top_probability": 0.0,
            "minimum_source_advantage": 0.0,
            "minimum_price_proxy": 0.02,
            "maximum_price_proxy": 0.90,
            "fixed_cost_per_share": 0.015,
            "rationale": (
                "Amsterdam agreement is a distinct, relatively reliable "
                "late regime; it must not share calibration with disagreement."
            ),
        },
        {
            "policy_id": "eham_late_disagreement_v2",
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
                "Amsterdam disagreement keeps the weather-selected bucket but "
                "replaces its overconfident raw probability with a separate "
                "empirical probability."
            ),
        },
    ],
    "EGLC": [
        {
            "policy_id": "eglc_late_market_follow_v2",
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
                "London remains enabled as a separate paper hypothesis while "
                "its recent instability is measured prospectively."
            ),
        }
    ],
    "KLAX": [
        {
            "policy_id": "klax_midday_convergence_v2",
            "candidate_source": "weather",
            "relationship": "agree",
            "target_local_hour": 12,
            "latest_local_minute": 45,
            "minimum_weather_top_probability": 0.60,
            "minimum_market_top_probability": 0.0,
            "minimum_source_advantage": 0.0,
            "minimum_price_proxy": 0.02,
            "maximum_price_proxy": 0.80,
            "fixed_cost_per_share": 0.045,
            "rationale": (
                "At LAX the 12:00 convergence lane tests the interval before "
                "the nearly final 15:00 maximum is fully priced."
            ),
        },
        {
            "policy_id": "klax_late_certainty_fallback_v2",
            "candidate_source": "weather",
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
                "The late LAX lane is a fallback only when no midday position "
                "was opened and a positive calibrated edge still remains."
            ),
        },
    ],
    "KLGA": [
        {
            "policy_id": "klga_late_convergence_v2",
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
                "LaGuardia remains a late convergence lane because its "
                "morning remaining-maximum uncertainty is materially larger."
            ),
        }
    ],
}


def evaluate_station_adaptive_policies(cfg: AppConfig) -> dict[str, Path]:

    snapshots = _load_historical_snapshots(cfg)
    split_dates = _recent_dates_by_station(snapshots)
    signals: list[dict[str, Any]] = []
    lane_summaries: dict[str, Any] = {}
    locked_lanes: dict[str, dict[str, Any]] = {}

    for station_id, lanes in STATION_ADAPTIVE_LANES.items():
        for lane in lanes:
            lane_rows: list[dict[str, Any]] = []
            for snapshot in snapshots:
                if (
                    snapshot["station_id"] != station_id
                    or snapshot["local_hour"] != lane["target_local_hour"]
                ):
                    continue
                evaluation = evaluate_station_policy_snapshot(
                    lane,
                    snapshot["weather_distribution"],
                    snapshot["market_distribution"],
                )
                if not evaluation["eligible"]:
                    continue
                won = (
                    evaluation["candidate_bucket"]
                    == snapshot["winning_bucket"]
                )
                row = {
                    "station_id": station_id,
                    "policy_id": lane["policy_id"],
                    "local_date": snapshot["local_date"],
                    "split": (
                        "recent_diagnostic"
                        if snapshot["local_date"]
                        in split_dates[station_id]
                        else "development_diagnostic"
                    ),
                    "local_hour": snapshot["local_hour"],
                    "candidate_bucket": evaluation["candidate_bucket"],
                    "winning_bucket": snapshot["winning_bucket"],
                    "relationship": lane["relationship"],
                    "candidate_weather_probability": evaluation[
                        "candidate_weather_probability"
                    ],
                    "candidate_price_proxy": evaluation[
                        "candidate_price_proxy"
                    ],
                    "won": won,
                    "fixed_cost_per_share": lane[
                        "fixed_cost_per_share"
                    ],
                    "synthetic_value_per_share": (
                        (1.0 if won else 0.0)
                        - float(evaluation["candidate_price_proxy"])
                        - float(lane["fixed_cost_per_share"])
                    ),
                }
                signals.append(row)
                lane_rows.append(row)

            wins = sum(_truth(row["won"]) for row in lane_rows)
            count = len(lane_rows)
            locked_probability = _weather_centered_locked_probability(
                lane_rows
            )
            locked_lane = dict(lane)
            locked_lane.update(
                {
                    "station_id": station_id,
                    "locked_calibrated_probability": locked_probability,
                    "diagnostic_station_days": count,
                    "diagnostic_wins": wins,
                    "enabled": True,
                }
            )
            locked_lanes[lane["policy_id"]] = locked_lane
            lane_summaries[lane["policy_id"]] = {
                "station_id": station_id,
                "policy": lane,
                "locked_calibrated_probability": locked_probability,
                "development_diagnostic": _summarize_lane(
                    [
                        row
                        for row in lane_rows
                        if row["split"] == "development_diagnostic"
                    ],
                    locked_probability,
                ),
                "recent_diagnostic": _summarize_lane(
                    [
                        row
                        for row in lane_rows
                        if row["split"] == "recent_diagnostic"
                    ],
                    locked_probability,
                ),
                "all_available": _summarize_lane(
                    lane_rows,
                    locked_probability,
                ),
            }

    portfolio = _replay_locked_portfolio(
        snapshots,
        locked_lanes,
        split_dates,
    )
    output_dir = cfg.reports_dir / "station_adaptive_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / "historical_lane_signals.csv"
    portfolio_path = output_dir / "historical_portfolio_trades.csv"
    summary_path = output_dir / "historical_adaptive_summary.json"
    _write_csv_atomic(signals_path, signals)
    _write_csv_atomic(portfolio_path, portfolio)
    _write_json_atomic(
        summary_path,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "strategy": "station_hour_relationship_adaptive_v2",
            "station_lanes": lane_summaries,
            "calibration_diagnostics": _calibration_diagnostics(
                snapshots
            ),
            "locked_portfolio_replay": _summarize_portfolio(
                portfolio
            ),
            "selection_warning": (
                "Lane shapes and locked probabilities use the available "
                "history and are diagnostic, not untouched validation. The "
                "new deployment timestamp is the prospective boundary."
            ),
            "locked_replay_warning": (
                "The locked-edge replay uses a probability estimated on the "
                "same sample. It is included to mirror execution logic, not "
                "as proof of out-of-sample profit."
            ),
            "unit_of_independence": "station_id + local_date",
            "maximum_entries_per_station_day": 1,
            "live_orders_enabled": False,
        },
    )
    return {
        "signals": signals_path,
        "portfolio": portfolio_path,
        "summary": summary_path,
    }


def deploy_station_adaptive_policies(
    cfg: AppConfig,
    *,
    output_namespace: str = "paper_trading_station_adaptive",
    strategy: str = "station_hour_relationship_adaptive_v2",
    minimum_edge: float | None = None,
    maximum_edge: float | None = None,
) -> dict[str, Path]:

    diagnostics = evaluate_station_adaptive_policies(cfg)
    summary = json.loads(diagnostics["summary"].read_text(encoding="utf-8"))
    lanes: dict[str, dict[str, Any]] = {}
    for lane_id, values in summary["station_lanes"].items():
        lane = dict(values["policy"])
        lane.update(
            {
                "station_id": values["station_id"],
                "locked_calibrated_probability": values[
                    "locked_calibrated_probability"
                ],
                "diagnostic_station_days": values["all_available"][
                    "signals"
                ],
                "diagnostic_wins": values["all_available"]["wins"],
                "enabled": True,
            }
        )
        lanes[lane_id] = lane

    deployed_at_ns = time.time_ns()
    output_dir = cfg.reports_dir / output_namespace
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_path = output_dir / "deployment.json"
    state_path = output_dir / "state.json"
    _write_json_atomic(
        deployment_path,
        {
            "strategy": strategy,
            "deployed_at_ns": deployed_at_ns,
            "deployed_at_utc": datetime.fromtimestamp(
                deployed_at_ns / 1e9,
                tz=UTC,
            ).isoformat(),
            "prospective_boundary": (
                "Only scheduled challenger rows strictly after deployment "
                "may open an adaptive paper position."
            ),
            "lanes": lanes,
            "maximum_entries_per_station_day": 1,
            "minimum_edge_inclusive": minimum_edge,
            "maximum_edge_exclusive": maximum_edge,
            "probability_lock": (
                "Separate weak weather-centered Beta probability for each "
                "station/hour/relationship lane (two prior observations)"
            ),
            "source_diagnostics": str(diagnostics["summary"]),
            "all_airports_enabled": True,
            "synthetic_cost_only": True,
            "live_orders_enabled": False,
        },
    )
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


def deploy_station_edge_band_policies(
    cfg: AppConfig,
) -> dict[str, Path]:

    return deploy_station_adaptive_policies(
        cfg,
        output_namespace="paper_trading_station_edge_band",
        strategy="station_edge_band_15_20_v1",
        minimum_edge=0.15,
        maximum_edge=0.20,
    )


def run_station_adaptive_paper_cycle(
    cfg: AppConfig,
    *,
    output_namespace: str = "paper_trading_station_adaptive",
) -> dict[str, Path]:

    output_dir = cfg.reports_dir / output_namespace
    deployment_path = output_dir / "deployment.json"
    state_path = output_dir / "state.json"
    source_path = (
        cfg.reports_dir
        / "paper_trading_challenger"
        / "paper_signal_audit.csv"
    )
    if not deployment_path.exists():
        raise FileNotFoundError(
            f"Missing {deployment_path}; run deploy-station-adaptive"
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

    lanes_by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane in deployment["lanes"].values():
        if lane.get("enabled", True):
            lanes_by_station[str(lane["station_id"])].append(lane)

    for row in _load_csv(source_path):
        decision_id = str(row.get("decision_id") or "")
        timestamp_ns = _timestamp_ns(row.get("received_wall_ns"))
        station_id = str(row.get("station_id") or "")
        if (
            not decision_id
            or decision_id in processed
            or timestamp_ns is None
            or timestamp_ns <= deployed_at_ns
            or station_id not in lanes_by_station
            or str(row.get("decision_mode") or "") != "scheduled"
        ):
            continue
        processed.add(decision_id)
        station = stations[station_id]
        received = datetime.fromisoformat(
            str(row["received_utc"]).replace("Z", "+00:00")
        ).astimezone(ZoneInfo(station.timezone))
        local_date = received.date().isoformat()
        day_key = f"{station_id}|{local_date}"
        weather = _probabilities(row.get("weather_distribution_json"))
        market = _probabilities(row.get("market_distribution_json"))
        matching = [
            lane
            for lane in lanes_by_station[station_id]
            if received.hour == int(lane["target_local_hour"])
        ]
        evaluations: list[dict[str, Any]] = []
        for lane in matching:
            evaluation = evaluate_station_policy_snapshot(
                lane,
                weather,
                market,
            )
            reasons = list(evaluation["rejection_reasons"])
            if received.minute > int(lane["latest_local_minute"]):
                reasons.append("after_station_entry_window")
            price = evaluation.get("candidate_price_proxy")
            probability = float(
                lane["locked_calibrated_probability"]
            )
            edge = (
                probability
                - float(price)
                - float(lane["fixed_cost_per_share"])
                if price is not None
                else None
            )
            if edge is None or edge <= 0:
                reasons.append("nonpositive_locked_synthetic_edge")
            reasons.extend(_edge_band_rejections(edge, deployment))
            if day_key in traded_days:
                reasons.append("station_day_already_traded")
            evaluations.append(
                {
                    "lane": lane,
                    "evaluation": evaluation,
                    "price": price,
                    "probability": probability,
                    "edge": edge,
                    "reasons": sorted(set(reasons)),
                }
            )

        eligible = [
            item for item in evaluations if not item["reasons"]
        ]
        chosen = (
            max(eligible, key=lambda item: float(item["edge"]))
            if eligible
            else None
        )
        if not evaluations:
            audit_rows.append(
                {
                    "decision_id": decision_id,
                    "station_id": station_id,
                    "local_date": local_date,
                    "received_utc": row.get("received_utc"),
                    "received_wall_ns": timestamp_ns,
                    "policy_id": "",
                    "paper_trade_opened": False,
                    "rejection_reasons": "no_adaptive_lane_at_this_hour",
                    "live_orders_enabled": False,
                }
            )
            continue

        for item in evaluations:
            lane = item["lane"]
            evaluation = item["evaluation"]
            opened = item is chosen
            reasons = list(item["reasons"])
            if not opened and not reasons and chosen is not None:
                reasons.append("lower_edge_than_selected_lane")
            audit_rows.append(
                {
                    "decision_id": decision_id,
                    "station_id": station_id,
                    "local_date": local_date,
                    "received_utc": row.get("received_utc"),
                    "received_wall_ns": timestamp_ns,
                    "policy_id": lane["policy_id"],
                    "candidate_bucket": evaluation.get(
                        "candidate_bucket"
                    ),
                    "candidate_price_proxy": item["price"],
                    "locked_calibrated_probability": item[
                        "probability"
                    ],
                    "fixed_cost_per_share": lane[
                        "fixed_cost_per_share"
                    ],
                    "estimated_synthetic_edge": item["edge"],
                    "weather_top_bucket": evaluation.get(
                        "weather_top_bucket"
                    ),
                    "market_top_bucket": evaluation.get(
                        "market_top_bucket"
                    ),
                    "paper_trade_opened": opened,
                    "rejection_reasons": "|".join(sorted(set(reasons))),
                    "live_orders_enabled": False,
                }
            )
        if chosen is None:
            continue

        lane = chosen["lane"]
        evaluation = chosen["evaluation"]
        synthetic_cost = float(chosen["price"]) + float(
            lane["fixed_cost_per_share"]
        )
        trade_rows.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "policy_id": lane["policy_id"],
                "entry_received_utc": row.get("received_utc"),
                "entry_received_wall_ns": timestamp_ns,
                "bucket": evaluation["candidate_bucket"],
                "price_proxy": chosen["price"],
                "fixed_cost_per_share": lane[
                    "fixed_cost_per_share"
                ],
                "synthetic_cost_per_share": synthetic_cost,
                "locked_calibrated_probability": chosen[
                    "probability"
                ],
                "estimated_synthetic_edge": chosen["edge"],
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
            "minimum_edge_inclusive": deployment.get(
                "minimum_edge_inclusive"
            ),
            "maximum_edge_exclusive": deployment.get(
                "maximum_edge_exclusive"
            ),
            "lanes": deployment["lanes"],
            "processed_decisions": len(processed),
            "audit_rows": len(audit_rows),
            "paper_trades": len(trade_rows),
            "settled_trades": len(settled),
            "wins": sum(_truth(row.get("won")) for row in settled),
            "synthetic_value_per_share_total": sum(
                float(row["synthetic_value_per_share"])
                for row in settled
            ),
            "all_airports_enabled": True,
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


def run_station_edge_band_paper_cycle(
    cfg: AppConfig,
) -> dict[str, Path]:

    return run_station_adaptive_paper_cycle(
        cfg,
        output_namespace="paper_trading_station_edge_band",
    )


def _edge_band_rejections(
    edge: float | None,
    deployment: dict[str, Any],
) -> list[str]:
    if edge is None:
        return []
    reasons: list[str] = []
    minimum = deployment.get("minimum_edge_inclusive")
    maximum = deployment.get("maximum_edge_exclusive")
    if minimum is not None and edge < float(minimum):
        reasons.append("edge_below_deployed_band")
    if maximum is not None and edge >= float(maximum):
        reasons.append("edge_above_deployed_band")
    return reasons


def _recent_dates_by_station(
    snapshots: list[dict[str, Any]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for station_id in STATION_ADAPTIVE_LANES:
        dates = sorted(
            {
                str(row["local_date"])
                for row in snapshots
                if row["station_id"] == station_id
            }
        )
        result[station_id] = set(dates[int(len(dates) * 0.70) :])
    return result


def _weather_centered_locked_probability(
    rows: list[dict[str, Any]],
    *,
    prior_strength: float = 2.0,
) -> float:

    if not rows:
        return 0.5
    wins = sum(_truth(row.get("won")) for row in rows)
    raw_center = sum(
        float(row["candidate_weather_probability"]) for row in rows
    ) / len(rows)
    return (
        wins + prior_strength * raw_center
    ) / (len(rows) + prior_strength)


def _summarize_lane(
    rows: list[dict[str, Any]],
    locked_probability: float,
) -> dict[str, Any]:
    count = len(rows)
    wins = sum(_truth(row.get("won")) for row in rows)
    mean_price = (
        sum(float(row["candidate_price_proxy"]) for row in rows) / count
        if count
        else None
    )
    mean_weather = (
        sum(
            float(row["candidate_weather_probability"])
            for row in rows
        )
        / count
        if count
        else None
    )
    locked_rows = [
        row
        for row in rows
        if locked_probability
        > float(row["candidate_price_proxy"])
        + float(row["fixed_cost_per_share"])
    ]
    locked_value = sum(
        float(row["synthetic_value_per_share"])
        for row in locked_rows
    )
    return {
        "signals": count,
        "wins": wins,
        "hit_rate": wins / count if count else None,
        "mean_raw_weather_probability": mean_weather,
        "raw_weather_calibration_error": (
            mean_weather - (wins / count)
            if count and mean_weather is not None
            else None
        ),
        "mean_price_proxy": mean_price,
        "mean_synthetic_value_per_share_all_eligible": (
            sum(
                float(row["synthetic_value_per_share"])
                for row in rows
            )
            / count
            if count
            else None
        ),
        "locked_edge_replay_signals": len(locked_rows),
        "locked_edge_replay_wins": sum(
            _truth(row.get("won")) for row in locked_rows
        ),
        "locked_edge_replay_total_value": locked_value,
        "locked_edge_replay_mean_value": (
            locked_value / len(locked_rows) if locked_rows else None
        ),
    }


def _replay_locked_portfolio(
    snapshots: list[dict[str, Any]],
    locked_lanes: dict[str, dict[str, Any]],
    split_dates: dict[str, set[str]],
) -> list[dict[str, Any]]:
    by_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        by_day[
            (snapshot["station_id"], snapshot["local_date"])
        ].append(snapshot)

    trades: list[dict[str, Any]] = []
    for (station_id, local_date), day_rows in sorted(by_day.items()):
        candidates: list[dict[str, Any]] = []
        for snapshot in sorted(
            day_rows,
            key=lambda row: int(row["local_hour"]),
        ):
            hour_candidates: list[dict[str, Any]] = []
            for lane in locked_lanes.values():
                if (
                    lane["station_id"] != station_id
                    or lane["target_local_hour"]
                    != snapshot["local_hour"]
                ):
                    continue
                evaluation = evaluate_station_policy_snapshot(
                    lane,
                    snapshot["weather_distribution"],
                    snapshot["market_distribution"],
                )
                if not evaluation["eligible"]:
                    continue
                edge = (
                    float(lane["locked_calibrated_probability"])
                    - float(evaluation["candidate_price_proxy"])
                    - float(lane["fixed_cost_per_share"])
                )
                if edge <= 0:
                    continue
                hour_candidates.append(
                    {
                        "lane": lane,
                        "evaluation": evaluation,
                        "edge": edge,
                        "snapshot": snapshot,
                    }
                )
            if hour_candidates:
                candidates = hour_candidates
                break
        if not candidates:
            continue
        chosen = max(candidates, key=lambda row: row["edge"])
        lane = chosen["lane"]
        evaluation = chosen["evaluation"]
        won = (
            evaluation["candidate_bucket"]
            == chosen["snapshot"]["winning_bucket"]
        )
        trades.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "split": (
                    "recent_diagnostic"
                    if local_date in split_dates[station_id]
                    else "development_diagnostic"
                ),
                "policy_id": lane["policy_id"],
                "local_hour": lane["target_local_hour"],
                "bucket": evaluation["candidate_bucket"],
                "winning_bucket": chosen["snapshot"]["winning_bucket"],
                "locked_calibrated_probability": lane[
                    "locked_calibrated_probability"
                ],
                "price_proxy": evaluation["candidate_price_proxy"],
                "fixed_cost_per_share": lane[
                    "fixed_cost_per_share"
                ],
                "estimated_synthetic_edge": chosen["edge"],
                "won": won,
                "synthetic_value_per_share": (
                    (1.0 if won else 0.0)
                    - float(evaluation["candidate_price_proxy"])
                    - float(lane["fixed_cost_per_share"])
                ),
            }
        )
    return trades


def _summarize_portfolio(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        wins = sum(_truth(row.get("won")) for row in rows)
        total = sum(
            float(row["synthetic_value_per_share"]) for row in rows
        )
        return {
            "trades": count,
            "wins": wins,
            "hit_rate": wins / count if count else None,
            "total_value": total,
            "mean_value": total / count if count else None,
            "mean_value_ci95": (
                _bootstrap_mean_ci(
                    [
                        float(row["synthetic_value_per_share"])
                        for row in rows
                    ]
                )
                if rows
                else None
            ),
        }

    by_station: dict[str, Any] = {}
    for station_id in STATION_ADAPTIVE_LANES:
        station_rows = [
            row for row in trades if row["station_id"] == station_id
        ]
        by_station[station_id] = {
            "development_diagnostic": summarize(
                [
                    row
                    for row in station_rows
                    if row["split"] == "development_diagnostic"
                ]
            ),
            "recent_diagnostic": summarize(
                [
                    row
                    for row in station_rows
                    if row["split"] == "recent_diagnostic"
                ]
            ),
            "all_available": summarize(station_rows),
        }
    return {
        "by_station": by_station,
        "all_stations": summarize(trades),
    }


def _calibration_diagnostics(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[tuple[float, float, bool]]] = (
        defaultdict(list)
    )
    for snapshot in snapshots:
        weather = snapshot["weather_distribution"]
        market = snapshot["market_distribution"]
        weather_top = max(weather, key=weather.get)
        market_top = max(market, key=market.get)
        relationship = (
            "agree" if weather_top == market_top else "disagree"
        )
        item = (
            float(weather[weather_top]),
            float(market.get(weather_top, 0.0)),
            weather_top == snapshot["winning_bucket"],
        )
        key = (
            snapshot["station_id"],
            int(snapshot["local_hour"]),
            relationship,
        )
        groups[key].append(item)
        groups[
            (
                snapshot["station_id"],
                int(snapshot["local_hour"]),
                "all",
            )
        ].append(item)

    diagnostics: list[dict[str, Any]] = []
    for (station_id, hour, relationship), rows in sorted(groups.items()):
        count = len(rows)
        wins = sum(item[2] for item in rows)
        hit_rate = wins / count
        mean_weather = sum(item[0] for item in rows) / count
        mean_market = sum(item[1] for item in rows) / count
        diagnostics.append(
            {
                "station_id": station_id,
                "local_hour": hour,
                "relationship": relationship,
                "signals": count,
                "wins": wins,
                "hit_rate": hit_rate,
                "mean_raw_weather_top_probability": mean_weather,
                "raw_weather_probability_minus_hit_rate": (
                    mean_weather - hit_rate
                ),
                "mean_market_probability_for_weather_top": mean_market,
                "hit_rate_minus_market_probability": (
                    hit_rate - mean_market
                ),
            }
        )
    return diagnostics
