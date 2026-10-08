from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .model import _brier, _safe_float
from .paper_trading import (
    PERSISTENT_DISAGREEMENT_STRATEGY,
    _add_canonical_features,
    _bundle_persistence_rules,
    _bundle_station_strategy_parameters,
    _evaluate_record,
    _json_floats,
    _load_bundle,
    _normalize,
    _station_execution_buffers,
    _write_csv_atomic,
    _write_json_atomic,
    advance_persistence,
)


def replay_persistent_disagreement(cfg: AppConfig) -> dict[str, Path]:

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Install analysis dependencies: pip install -e '.[analysis]'"
        ) from exc

    comparison_path = (
        cfg.reports_dir
        / "calibration_challenger"
        / "chronological_comparison.csv"
    )
    bundle_path = (
        cfg.reports_dir
        / "paper_trading_persistent_disagreement"
        / "paper_model_bundle.pkl"
    )
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing {comparison_path}; run evaluate-calibration-challenger"
        )
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing {bundle_path}; run fit-persistent-disagreement"
        )
    bundle = _load_bundle(bundle_path)
    comparison = pd.read_csv(comparison_path)
    comparison["received_wall_ns"] = pd.to_numeric(
        comparison["received_wall_ns"], errors="coerce"
    )
    comparison = comparison.dropna(
        subset=["station_id", "test_date", "received_wall_ns"]
    )
    comparison["received_wall_ns"] = comparison["received_wall_ns"].astype(
        "int64"
    )
    priority = {
        "forecast_update": 0,
        "scheduled": 1,
        "historical_scheduled": 1,
        "event_driven": 2,
    }
    comparison["_mode_priority"] = (
        comparison["decision_mode"].astype(str).map(priority).fillna(3)
    )
    comparison = (
        comparison.sort_values(["received_wall_ns", "_mode_priority"])
        .drop_duplicates(
            ["station_id", "test_date", "received_wall_ns"], keep="first"
        )
        .drop(columns=["_mode_priority"])
    )

    feature_frames = []
    for path in sorted(cfg.reports_dir.glob("*/predictive_features.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["_feature_file"] = path.parent.name
        feature_frames.append(frame)
    if not feature_frames:
        raise FileNotFoundError("No predictive feature files were found")
    features = pd.concat(feature_frames, ignore_index=True)
    features["received_wall_ns"] = pd.to_numeric(
        features["received_wall_ns"], errors="coerce"
    )
    features = features.dropna(subset=["station_id", "received_wall_ns"])
    features["received_wall_ns"] = features["received_wall_ns"].astype("int64")
    features["_has_exact_l2"] = (
        features["market_ask_json"]
        .fillna("{}")
        .astype(str)
        .ne("{}")
        & features["market_ask_size_json"]
        .fillna("{}")
        .astype(str)
        .ne("{}")
    )
    features = features.sort_values(
        ["_has_exact_l2", "received_wall_ns"], ascending=[False, True]
    ).drop_duplicates(["station_id", "received_wall_ns"], keep="first")
    feature_map = {
        (str(row["station_id"]), int(row["received_wall_ns"])): row
        for row in features.to_dict(orient="records")
    }
    exact_dates = {
        str(row["test_date"])
        for row in comparison.to_dict(orient="records")
        if (
            feature := feature_map.get(
                (str(row["station_id"]), int(row["received_wall_ns"]))
            )
        )
        and bool(feature.get("_has_exact_l2"))
    }

    stations = {
        station.station_id: station for station in cfg.stations.values()
    }
    enabled = set(
        bundle.parameters.get(
            "calibration_challenger_enabled_stations", bundle.stations
        )
    )
    station_buffers = _station_execution_buffers(cfg)
    persistence_minutes, minimum_confirmations = _bundle_persistence_rules(
        bundle, cfg.paper_trading
    )
    persistence_state: dict[str, Any] = {}
    traded_days: set[tuple[str, str]] = set()
    audit_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for comparison_row in comparison.to_dict(orient="records"):
        local_date = str(comparison_row["test_date"])
        if local_date not in exact_dates:
            continue
        station_id = str(comparison_row["station_id"])
        timestamp_ns = int(comparison_row["received_wall_ns"])
        station = stations[station_id]
        feature = feature_map.get((station_id, timestamp_ns))
        winner = str(comparison_row.get("winning_bucket") or "")
        day_key = (station_id, local_date)
        if not feature or not bool(feature.get("_has_exact_l2")):
            persistence = advance_persistence(
                persistence_state,
                station_id=station_id,
                local_date=local_date,
                timestamp_ns=timestamp_ns,
                bucket=None,
                eligible=False,
                minimum_confirmations=minimum_confirmations,
                persistence_minutes=persistence_minutes,
            )
            audit_rows.append(
                {
                    "station_id": station_id,
                    "local_date": local_date,
                    "received_wall_ns": timestamp_ns,
                    "decision_mode": comparison_row.get("decision_mode"),
                    "rejection_reasons": "missing_exact_l2_state",
                    "paper_trade_opened": False,
                    "winning_bucket": winner,
                    **persistence,
                }
            )
            continue

        feature_frame = _add_canonical_features(
            pd.DataFrame([feature]), station
        )
        record = feature_frame.iloc[0].to_dict()
        distribution_field = (
            "challenger_distribution_json"
            if station_id in enabled
            else "incumbent_distribution_json"
        )
        weather_distribution = _json_floats(
            comparison_row.get(distribution_field)
        )
        strategy_parameters = _bundle_station_strategy_parameters(
            bundle, station_id
        )
        evaluation = _evaluate_record(
            record,
            station,
            bundle.stations[station_id],
            cfg.paper_trading,
            station_buffers.get(
                station_id, cfg.paper_trading.default_execution_buffer
            ),
            calibration_strategy=PERSISTENT_DISAGREEMENT_STRATEGY,
            strategy_parameters=strategy_parameters,
            weather_distribution_override=weather_distribution,
            calibration_days_override=int(
                _safe_float(comparison_row.get("calibration_days")) or 0
            ),
        )
        persistence = advance_persistence(
            persistence_state,
            station_id=station_id,
            local_date=local_date,
            timestamp_ns=timestamp_ns,
            bucket=evaluation.get("candidate_bucket"),
            eligible=not evaluation["rejection_reasons"],
            minimum_confirmations=minimum_confirmations,
            persistence_minutes=persistence_minutes,
        )
        market_distribution = _normalize(
            _json_floats(record.get("market_distribution_json"))
        )
        weather_brier = (
            _brier(weather_distribution, winner)
            if weather_distribution and winner
            else None
        )
        market_brier = (
            _brier(market_distribution, winner)
            if market_distribution and winner
            else None
        )
        can_open = bool(
            persistence["persistent"] and day_key not in traded_days
        )
        audit = {
            "station_id": station_id,
            "local_date": local_date,
            "received_wall_ns": timestamp_ns,
            "decision_mode": comparison_row.get("decision_mode"),
            **evaluation,
            **persistence,
            "paper_trade_opened": can_open,
            "winning_bucket": winner,
            "weather_brier": weather_brier,
            "market_brier": market_brier,
            "brier_improvement": (
                market_brier - weather_brier
                if market_brier is not None and weather_brier is not None
                else None
            ),
        }
        audit_rows.append(audit)
        if not can_open:
            continue
        cost = float(evaluation["candidate_conservative_cost"])
        won = str(evaluation["candidate_bucket"]) == winner
        pnl_per_share = 1.0 - cost if won else -cost
        trade_rows.append(
            {
                "station_id": station_id,
                "local_date": local_date,
                "entry_received_wall_ns": timestamp_ns,
                "bucket": evaluation["candidate_bucket"],
                "winning_bucket": winner,
                "model_probability": evaluation[
                    "candidate_model_probability"
                ],
                "market_probability": evaluation[
                    "candidate_market_probability"
                ],
                "weather_market_gap": evaluation[
                    "candidate_weather_market_gap"
                ],
                "executable_ask": evaluation["candidate_ask"],
                "best_ask_size": evaluation["candidate_ask_size"],
                "conservative_cost_per_share": cost,
                "estimated_edge_after_costs": evaluation[
                    "candidate_edge_after_costs"
                ],
                "supporting_provider_votes": evaluation[
                    "candidate_provider_votes"
                ],
                "confirmations": persistence["confirmations"],
                "persistence_seconds": persistence["persistence_seconds"],
                "won": won,
                "realized_pnl_per_share": pnl_per_share,
                "realized_pnl_total": (
                    pnl_per_share * cfg.paper_trading.requested_shares
                ),
            }
        )
        traded_days.add(day_key)

    rejection_counts: Counter[str] = Counter()
    for row in audit_rows:
        for reason in str(row.get("rejection_reasons") or "").split("|"):
            if reason:
                rejection_counts[reason] += 1
    output_dir = cfg.reports_dir / "persistent_disagreement_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "replay_audit.csv"
    trades_path = output_dir / "replay_trades.csv"
    summary_path = output_dir / "summary.json"
    _write_csv_atomic(audit_path, audit_rows)
    _write_csv_atomic(trades_path, trade_rows)
    _write_json_atomic(
        summary_path,
        {
            "strategy": PERSISTENT_DISAGREEMENT_STRATEGY,
            "outcome_blind_signal": True,
            "probability_source": (
                "chronological OOS distributions; final winner used only after "
                "the locked decision"
            ),
            "exact_l2_dates": sorted(exact_dates),
            "audit_rows": len(audit_rows),
            "eligible_rows": sum(
                not str(row.get("rejection_reasons") or "")
                for row in audit_rows
            ),
            "persistent_rows": sum(
                bool(row.get("persistent")) for row in audit_rows
            ),
            "trade_station_days": len(trade_rows),
            "wins": sum(bool(row.get("won")) for row in trade_rows),
            "total_pnl_10_shares": sum(
                float(row["realized_pnl_total"]) for row in trade_rows
            ),
            "rejection_counts": dict(rejection_counts.most_common()),
            "warnings": [
                "Only dates with exact OOS probability timestamps and exact L2 are included.",
                "The strategy thresholds were defined after inspecting earlier data.",
                "This replay generates hypotheses and is not prospective alpha proof.",
            ],
        },
    )
    return {
        "audit": audit_path,
        "trades": trades_path,
        "summary": summary_path,
    }
