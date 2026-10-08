from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import AppConfig
from ..parsing.buckets import c_to_f, parse_temperature_bucket, round_half_up
MARKET_THRESHOLDS = (0.50, 0.75, 0.90)
REVISION_DEDUPLICATION_SECONDS = 10
MIN_ALPHA_STATION_DAYS = 20
MIN_CALIBRATED_VALIDATION_DAYS = 20
MIN_TRADE_DAYS = 20


def discover_alpha(cfg: AppConfig, run_id: str | None = None) -> dict[str, Path]:

    report_dirs = _report_dirs(cfg.reports_dir, run_id)
    if not report_dirs:
        raise FileNotFoundError(
            "No predictive reports found. Run build-dataset for at least one run first."
        )

    feature_rows = _load_csvs(report_dirs, "predictive_features.csv")
    live_revision_rows = _load_csvs(report_dirs, "forecast_revision_reactions.csv")
    if run_id:
        output_dir = cfg.reports_dir / run_id / "alpha_discovery"
    else:
        output_dir = cfg.reports_dir / "alpha_discovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    active_report_runs = [
        path.name
        for path in report_dirs
        if _run_manifest_status(cfg.data_dir, path.name) == "running"
    ]

    forecast_rows, opportunities = _forecast_evidence(feature_rows, cfg)
    forecast_summary = _forecast_summary(forecast_rows, opportunities)
    actionable_revisions = _actionable_revision_rows(live_revision_rows)
    revision_summary = _live_revision_summary(
        actionable_revisions,
        raw_rows=live_revision_rows,
    )
    master_validation = _load_master_validation(cfg.reports_dir)
    verdict = _alpha_verdict(forecast_summary, revision_summary, master_validation)

    summary = {
        "scope": "single_run" if run_id else "all_built_runs",
        "run_id": run_id,
        "included_report_runs": [path.name for path in report_dirs],
        "included_active_partial_runs": active_report_runs,
        "method": {
            "order": [
                "discard revisions that do not change the actual Polymarket contract",
                "deduplicate identical revisions from equivalent providers",
                "measure executable ask-to-future-bid markouts",
                "score station-specific forecasts only after the final WU label exists",
            ],
            "model_fit_executed_by_this_command": False,
            "master_validation_loaded": master_validation.get("status") == "available",
            "warning": (
                "Final winning buckets are used only to assess recorded paper decisions. "
                "Unfinalized revisions and repeated intraday rows cannot establish alpha."
            ),
        },
        "alpha_verdict": verdict,
        "forecast_hypotheses": forecast_summary,
        "live_forecast_revisions": revision_summary,
        "master_model_validation": master_validation,
    }

    paths = {
        "summary": output_dir / "alpha_discovery.json",
        "forecast_evidence": output_dir / "forecast_evidence.csv",
        "opportunities": output_dir / "ex_post_opportunities.csv",
        "revision_reactions": output_dir / "actionable_forecast_revisions.csv",
        "report": output_dir / "alpha_discovery.md",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(paths["forecast_evidence"], forecast_rows)
    _write_csv(paths["opportunities"], opportunities)
    _write_csv(paths["revision_reactions"], actionable_revisions)
    paths["report"].write_text(_markdown_report(summary, opportunities), encoding="utf-8")
    return paths


def _report_dirs(reports_dir: Path, run_id: str | None) -> list[Path]:
    if run_id:
        candidate = reports_dir / run_id
        return [candidate] if (candidate / "predictive_features.csv").exists() else []
    return sorted(
        path
        for path in reports_dir.iterdir()
        if path.is_dir() and (path / "predictive_features.csv").exists()
    )


def _load_csvs(report_dirs: Iterable[Path], filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for report_dir in report_dirs:
        path = report_dir / filename
        if not path.exists() or not path.stat().st_size:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["report_run_id"] = report_dir.name
                rows.append(row)
    return rows


def _load_master_validation(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "model_validation" / "master_remaining_max_summary.json"
    if not path.exists():
        return {
            "status": "not_run",
            "summary_path": str(path),
            "station_modes": [],
            "model_readiness": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "summary_path": str(path),
            "error": str(exc),
            "station_modes": [],
            "model_readiness": [],
        }
    payload["status"] = "available"
    payload["summary_path"] = str(path)
    return payload


def _run_manifest_status(data_dir: Path, run_id: str) -> str | None:
    path = data_dir / "runs" / run_id / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def _forecast_evidence(
    feature_rows: list[dict[str, str]], cfg: AppConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stations = {station.station_id: station for station in cfg.stations.values()}
    prepared: list[dict[str, Any]] = []
    by_day: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        station = stations.get(str(row.get("station_id") or ""))
        winner = str(row.get("winning_bucket") or "")
        timestamp_ns = _integer(row.get("received_wall_ns"))
        mids = _json_prices(row.get("market_distribution_json"))
        asks = _json_prices(row.get("market_ask_json"))
        mid_sum = sum(mids.values())
        if (
            station is None
            or not winner
            or timestamp_ns is None
            or not mids
            or winner not in asks
            or not _truth(row.get("market_ws_healthy"))
            or not _truth(row.get("continuity_eligible"))


            or not 0.90 <= mid_sum <= 1.25
        ):
            continue
        probabilities = _normalize(mids)
        if winner not in probabilities:
            continue
        prepared_row = {
            "raw": row,
            "station": station,
            "timestamp_ns": timestamp_ns,
            "winner": winner,
            "probabilities": probabilities,
            "asks": asks,
            "observed_mid_sum": mid_sum,
        }
        prepared.append(prepared_row)
        by_day[(station.station_id, str(row.get("local_date")), str(row.get("decision_mode")))].append(
            prepared_row
        )

    crossings: dict[tuple[str, str, str], dict[float, int | None]] = {}
    for key, rows in by_day.items():
        ordered = sorted(rows, key=lambda item: item["timestamp_ns"])
        crossings[key] = {
            threshold: next(
                (
                    item["timestamp_ns"]
                    for item in ordered
                    if item["probabilities"].get(item["winner"], 0.0) >= threshold
                ),
                None,
            )
            for threshold in MARKET_THRESHOLDS
        }

    evidence: list[dict[str, Any]] = []
    for item in prepared:
        row = item["raw"]
        station = item["station"]
        labels = set(item["probabilities"])
        predictions: dict[str, tuple[float, str | None]] = {}
        consensus = _num(row.get("model_consensus_today_max_c"))
        if consensus is not None:
            predictions["consensus"] = (
                consensus,
                _bucket_for_temperature(labels, consensus, station.market_unit),
            )
        for provider in station.forecast_providers:
            value = _num(row.get(f"model_{provider}_today_max_c"))
            if value is not None:
                predictions[provider] = (
                    value,
                    _bucket_for_temperature(labels, value, station.market_unit),
                )
        predicted_labels = [label for _, label in predictions.values() if label]
        vote_counts = Counter(predicted_labels)
        day_key = (station.station_id, str(row.get("local_date")), str(row.get("decision_mode")))
        for provider, (forecast_c, predicted_bucket) in predictions.items():
            if not predicted_bucket:
                continue
            forecast_age_minutes = (
                _num(row.get(f"model_{provider}_age_minutes"))
                if provider != "consensus"
                else None
            )
            information_ns = item["timestamp_ns"]
            information_time_basis = "decision_timestamp_conservative"
            if forecast_age_minutes is not None:
                information_ns -= round(forecast_age_minutes * 60e9)
                information_time_basis = "forecast_first_seen"
            predicted_probability = item["probabilities"].get(predicted_bucket)
            winner_probability = item["probabilities"].get(item["winner"])
            evidence.append(
                {
                    "run_id": row.get("report_run_id") or row.get("run_id"),
                    "station_id": station.station_id,
                    "local_date": row.get("local_date"),
                    "decision_mode": row.get("decision_mode"),
                    "received_wall_ns": item["timestamp_ns"],
                    "received_utc": _iso(item["timestamp_ns"]),
                    "information_available_ns": information_ns,
                    "information_available_utc": _iso(information_ns),
                    "information_time_basis": information_time_basis,
                    "local_hour": _num(row.get("local_hour")),
                    "trigger_source": row.get("trigger_source"),
                    "provider": provider,
                    "forecast_age_minutes": forecast_age_minutes,
                    "forecast_max_c": forecast_c,
                    "predicted_bucket": predicted_bucket,
                    "winning_bucket": item["winner"],
                    "prediction_correct": predicted_bucket == item["winner"],
                    "provider_votes_for_prediction": vote_counts[predicted_bucket],
                    "available_provider_predictions": len(predicted_labels),
                    "ensemble_spread_c": _num(row.get("model_spread_today_max_c")),
                    "market_probability_predicted_bucket": predicted_probability,
                    "market_probability_winner": winner_probability,
                    "observed_market_mid_sum": item["observed_mid_sum"],
                    "market_ask_predicted_bucket": item["asks"].get(predicted_bucket),
                    "market_ask_winner": item["asks"].get(item["winner"]),
                    "market_quote_coverage": _num(row.get("market_quote_coverage")),
                    "final_wu_max": _num(row.get("final_wu_max")),
                    **{
                        f"market_{int(threshold * 100)}_crossing_delta_ms": (
                            (crossing_ns - item["timestamp_ns"]) / 1e6
                            if (crossing_ns := crossings[day_key][threshold]) is not None
                            else None
                        )
                        for threshold in MARKET_THRESHOLDS
                    },
                    **{
                        f"information_to_market_{int(threshold * 100)}_ms": (
                            (crossing_ns - information_ns) / 1e6
                            if (crossing_ns := crossings[day_key][threshold]) is not None
                            else None
                        )
                        for threshold in MARKET_THRESHOLDS
                    },
                }
            )

    evidence.sort(key=lambda row: (row["received_wall_ns"], row["station_id"], row["provider"]))
    previous_predictions: dict[tuple[str, str, str, str], str] = {}
    for row in evidence:
        key = (row["station_id"], row["local_date"], row["decision_mode"], row["provider"])
        previous = previous_predictions.get(key)
        row["previous_predicted_bucket"] = previous
        row["prediction_changed"] = previous is not None and previous != row["predicted_bucket"]
        row["revision_to_winner"] = bool(
            row["prediction_changed"] and row["prediction_correct"]
        )
        previous_predictions[key] = row["predicted_bucket"]
    opportunities: list[dict[str, Any]] = []
    grouped_evidence: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        grouped_evidence[
            (row["station_id"], row["local_date"], row["decision_mode"], row["provider"])
        ].append(row)
    for rows in grouped_evidence.values():
        ordered = sorted(rows, key=lambda row: row["received_wall_ns"])
        first_correct = next((row for row in ordered if row["prediction_correct"]), None)
        if first_correct is None:
            continue
        later = [row for row in ordered if row["received_wall_ns"] >= first_correct["received_wall_ns"]]
        persistence = sum(row["prediction_correct"] for row in later) / len(later)
        ask = _num(first_correct.get("market_ask_winner"))
        probability = _num(first_correct.get("market_probability_winner"))
        opportunities.append(
            {
                **first_correct,
                "subsequent_prediction_persistence": persistence,
                "subsequent_rows": len(later),
                "oracle_gross_payout_minus_ask": 1.0 - ask if ask is not None else None,
                "market_below_50_at_discovery": probability is not None and probability < 0.50,
                "hypothesis_type": (
                    "forecast_revision_to_winner"
                    if first_correct.get("revision_to_winner")
                    else "persistent_level_disagreement"
                ),
                "classification": _opportunity_classification(
                    probability, ask, persistence, first_correct.get("market_50_crossing_delta_ms")
                ),
            }
        )
    opportunities.sort(
        key=lambda row: (
            row["classification"] != "candidate_information_lead",
            _num(row.get("market_probability_winner")) or math.inf,
            row["received_wall_ns"],
        )
    )
    return evidence, opportunities


def _opportunity_classification(
    probability: float | None,
    ask: float | None,
    persistence: float,
    crossing_delta_ms: Any,
) -> str:
    crossing = _num(crossing_delta_ms)
    if probability is None or ask is None:
        return "missing_executable_quote"
    if probability >= 0.50 or ask >= 0.95:
        return "market_already_high"
    if persistence < 0.50:
        return "unstable_forecast"
    if crossing is not None and crossing <= 0:
        return "market_crossed_50_earlier"
    return "candidate_information_lead"


def _bucket_for_temperature(labels: set[str], forecast_c: float, market_unit: str) -> str | None:
    represented = c_to_f(forecast_c) if market_unit == "F" else forecast_c
    represented = round_half_up(represented)
    for label in sorted(labels):
        try:
            if parse_temperature_bucket(label, market_unit).contains(represented):
                return label
        except ValueError:
            continue
    return None


def _forecast_summary(
    evidence: list[dict[str, Any]], opportunities: list[dict[str, Any]]
) -> dict[str, Any]:
    independent_days = {
        (row["station_id"], row["local_date"]) for row in evidence if row.get("local_date")
    }
    independent_days_by_station = {
        station_id: len(
            {local_date for candidate, local_date in independent_days if candidate == station_id}
        )
        for station_id in sorted({station_id for station_id, _ in independent_days})
    }
    by_station_provider: dict[str, dict[str, Any]] = {}
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        grouped[(row["station_id"], row["provider"])].append(row)
    for (station, provider), rows in sorted(grouped.items()):
        by_station_provider[f"{station}:{provider}"] = {
            "rows": len(rows),
            "station_days": len({row["local_date"] for row in rows}),
            "row_hit_rate_non_independent": sum(row["prediction_correct"] for row in rows)
            / len(rows),
        }
    candidates = [
        row for row in opportunities if row["classification"] == "candidate_information_lead"
    ]
    return {
        "eligible_forecast_rows": len(evidence),
        "independent_station_days": len(independent_days),
        "independent_station_days_by_station": independent_days_by_station,
        "ex_post_earliest_correct_rows": len(opportunities),
        "candidate_information_lead_rows": len(candidates),
        "candidate_information_leads": len(
            {
                (row["station_id"], row["local_date"], row["provider"])
                for row in candidates
            }
        ),
        "by_station_provider": by_station_provider,
        "warning": (
            "Rows within a station-day share the same outcome and are not independent. Hit rates "
            "are descriptive diagnostics only; one realized day cannot establish forecast alpha."
        ),
    }


def _economic_revision(row: dict[str, Any]) -> bool:
    previous = str(row.get("previous_bucket_label") or "").strip()
    current = str(row.get("current_bucket_label") or "").strip()
    return bool(previous and current and previous != current)


def _revision_event_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("report_run_id") or row.get("run_id") or ""),
        str(row.get("source") or row.get("provider") or ""),
        str(row.get("revision_received_wall_ns") or ""),
    )


def _revision_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    def stable_number(name: str) -> float | None:
        value = _num(row.get(name))
        return round(value, 6) if value is not None else None

    return (
        str(row.get("report_run_id") or row.get("run_id") or ""),
        str(row.get("station_id") or ""),
        str(row.get("local_date") or ""),
        str(row.get("previous_bucket_label") or ""),
        str(row.get("current_bucket_label") or ""),
        stable_number("previous_today_max_c"),
        stable_number("current_today_max_c"),
        stable_number("previous_remaining_max_c"),
        stable_number("current_remaining_max_c"),
    )


def _actionable_revision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:

    t0_rows = sorted(
        (
            row
            for row in rows
            if _num(row.get("horizon_seconds")) == 0 and _economic_revision(row)
        ),
        key=lambda row: _integer(row.get("revision_received_wall_ns")) or 0,
    )
    canonical: dict[tuple[str, str, str], dict[str, Any]] = {}
    latest_cluster: dict[tuple[Any, ...], dict[str, Any]] = {}
    deduplication_ns = REVISION_DEDUPLICATION_SECONDS * 1_000_000_000
    for row in t0_rows:
        event_key = _revision_event_key(row)
        timestamp_ns = _integer(row.get("revision_received_wall_ns"))
        if timestamp_ns is None:
            continue
        signature = _revision_signature(row)
        cluster = latest_cluster.get(signature)
        if cluster and timestamp_ns - cluster["last_timestamp_ns"] <= deduplication_ns:
            cluster["last_timestamp_ns"] = timestamp_ns
            cluster["providers"].add(str(row.get("provider") or "unknown"))
            continue
        cluster = {
            "canonical_key": event_key,
            "last_timestamp_ns": timestamp_ns,
            "providers": {str(row.get("provider") or "unknown")},
        }
        latest_cluster[signature] = cluster
        canonical[event_key] = cluster

    output: list[dict[str, Any]] = []
    for row in rows:
        cluster = canonical.get(_revision_event_key(row))
        if cluster is None or not _economic_revision(row):
            continue
        prepared = dict(row)
        providers = sorted(cluster["providers"])
        prepared["collector_bucket_changed"] = row.get(
            "rounded_degree_changed", row.get("bucket_changed")
        )
        prepared["economic_bucket_changed"] = True
        prepared["bucket_changed"] = True
        prepared["confirming_providers"] = json.dumps(providers)
        prepared["provider_confirmation_count"] = len(providers)
        output.append(prepared)
    return sorted(
        output,
        key=lambda row: (
            _integer(row.get("revision_received_wall_ns")) or 0,
            _num(row.get("horizon_seconds")) or 0,
        ),
    )


def _optional_truth(value: Any) -> bool | None:
    raw = str(value or "").strip().lower()
    if raw in {"", "none", "null", "nan"}:
        return None
    return _truth(value)


def _live_revision_summary(
    rows: list[dict[str, Any]],
    *,
    raw_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_rows = raw_rows if raw_rows is not None else rows
    raw_t0 = [row for row in raw_rows if _num(row.get("horizon_seconds")) == 0]
    raw_economic = [row for row in raw_t0 if _economic_revision(row)]
    bucket_changes = [row for row in rows if _num(row.get("horizon_seconds")) == 0]
    finalized = [
        row for row in bucket_changes if _optional_truth(row.get("revision_correct")) is not None
    ]
    by_provider: dict[str, dict[str, Any]] = {}
    provider_names = {
        provider
        for row in bucket_changes
        for provider in _provider_confirmations(row)
    }
    for provider in sorted(provider_names):
        provider_rows = [row for row in bucket_changes if provider in _provider_confirmations(row)]
        provider_finalized = [
            row
            for row in provider_rows
            if _optional_truth(row.get("revision_correct")) is not None
        ]
        by_provider[provider] = {
            "confirmed_actionable_revisions": len(provider_rows),
            "t0_executable_quotes": sum(
                _num(row.get("market_ask")) is not None
                and _num(row.get("market_ask_size")) is not None
                and _truth(row.get("market_ws_healthy"))
                for row in provider_rows
            ),
            "finalized": len(provider_finalized),
            "correct": sum(
                _optional_truth(row.get("revision_correct")) is True
                for row in provider_finalized
            ),
        }
    markouts_by_horizon: dict[str, dict[str, float | int | None]] = {}
    for horizon in sorted(
        {
            int(value)
            for row in rows
            if (value := _num(row.get("horizon_seconds"))) is not None and value > 0
        }
    ):
        values = [
            value
            for row in rows
            if _num(row.get("horizon_seconds")) == horizon
            and _truth(row.get("market_ws_healthy"))
            and _truth(row.get("t0_market_ws_healthy", row.get("market_ws_healthy")))
            and (value := _num(row.get("executable_bid_markout_from_t0_ask"))) is not None
        ]
        markouts_by_horizon[str(horizon)] = _outcome_distribution(values)
    resolution_pnl = [
        (1.0 if _optional_truth(row.get("revision_correct")) is True else 0.0) - ask
        for row in finalized
        if (ask := _num(row.get("market_ask"))) is not None
        and _num(row.get("market_ask_size")) is not None
        and _truth(row.get("market_ws_healthy"))
    ]
    return {
        "raw_revision_events": len(raw_t0),
        "rounded_degree_change_events": sum(
            _truth(row.get("rounded_degree_changed", row.get("bucket_changed")))
            for row in raw_t0
        ),
        "economic_bucket_change_events_before_provider_deduplication": len(raw_economic),
        "duplicate_equivalent_provider_events_removed": len(raw_economic) - len(bucket_changes),
        "actionable_revision_events": len(bucket_changes),
        "t0_executable_quotes": sum(
            _num(row.get("market_ask")) is not None
            and _num(row.get("market_ask_size")) is not None
            and _truth(row.get("market_ws_healthy"))
            for row in bucket_changes
        ),
        "finalized_actionable_revisions": len(finalized),
        "correct_actionable_revisions": sum(
            _optional_truth(row.get("revision_correct")) is True for row in finalized
        ),
        "by_provider": by_provider,
        "executable_bid_markout_by_horizon_seconds": markouts_by_horizon,
        "gross_resolution_pnl_per_share_before_fees": _outcome_distribution(resolution_pnl),
    }


def _provider_confirmations(row: dict[str, Any]) -> list[str]:
    raw = row.get("confirming_providers")
    try:
        providers = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        providers = None
    if isinstance(providers, list):
        cleaned = [str(provider) for provider in providers if str(provider)]
        if cleaned:
            return cleaned
    return [str(row.get("provider") or "unknown")]


def _outcome_distribution(values: list[float]) -> dict[str, float | int | None]:
    result = _distribution(values)
    result.update(
        {
            "mean": sum(values) / len(values) if values else None,
            "positive": sum(value > 0 for value in values),
            "positive_share": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
            "non_negative": sum(value >= 0 for value in values),
        }
    )
    return result


def _alpha_verdict(
    forecast_summary: dict[str, Any],
    revision_summary: dict[str, Any],
    master_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    master_validation = master_validation or {
        "status": "not_run",
        "station_modes": [],
        "model_readiness": [],
    }
    five_second = revision_summary["executable_bid_markout_by_horizon_seconds"].get(
        "5", {}
    )
    short_count = int(five_second.get("count") or 0)
    short_positive_share = _num(five_second.get("positive_share"))
    short_median = _num(five_second.get("p50"))
    if (
        short_count >= 10
        and short_positive_share is not None
        and short_positive_share <= 0.20
        and short_median is not None
        and short_median <= 0
    ):
        trigger_status = "rejected"
        trigger_reason = "Blind forecast-revision buying does not overcome the spread."
    else:
        trigger_status = "insufficient_evidence"
        trigger_reason = "Too few executable short-horizon revision outcomes."

    forecast_days = {
        str(station): int(days or 0)
        for station, days in (
            forecast_summary.get("independent_station_days_by_station") or {}
        ).items()
    }
    readiness = {
        str(row.get("station_id")): row
        for row in master_validation.get("model_readiness", [])
        if row.get("station_id")
    }
    modes_by_station: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in master_validation.get("station_modes", []):
        station_id = str(row.get("station_id") or "")
        if station_id:
            modes_by_station[station_id].append(row)

    station_ids = sorted(set(forecast_days) | set(readiness) | set(modes_by_station))
    station_verdicts: dict[str, dict[str, Any]] = {}
    for station_id in station_ids:
        station_readiness = readiness.get(station_id, {})
        finalized_days = int(
            station_readiness.get("finalized_station_days")
            or forecast_days.get(station_id)
            or 0
        )
        modes = [row for row in modes_by_station.get(station_id, []) if row.get("status") == "evaluated"]
        supported_modes: list[str] = []
        conclusive_modes: list[str] = []
        for row in modes:
            calibrated_days = int(row.get("calibrated_days") or 0)
            trade_days = int(row.get("trade_days") or 0)
            brier_ci = row.get("brier_improvement_ci95") or {}
            pnl_ci = row.get("mean_pnl_per_station_day_ci95") or {}
            brier_low = _num(brier_ci.get("low"))
            brier_high = _num(brier_ci.get("high"))
            pnl_low = _num(pnl_ci.get("low"))
            pnl_high = _num(pnl_ci.get("high"))
            evidence_ready = (
                calibrated_days >= MIN_CALIBRATED_VALIDATION_DAYS
                and trade_days >= MIN_TRADE_DAYS
            )
            if evidence_ready:
                conclusive_modes.append(str(row.get("decision_mode") or "unknown"))
            if (
                evidence_ready
                and brier_low is not None
                and pnl_low is not None
                and brier_low > 0
                and pnl_low > 0
            ):
                supported_modes.append(str(row.get("decision_mode") or "unknown"))
            row["alpha_evidence_ready"] = evidence_ready
            row["brier_ci_entirely_non_positive"] = (
                brier_high is not None and brier_high <= 0
            )
            row["pnl_ci_entirely_non_positive"] = pnl_high is not None and pnl_high <= 0

        if supported_modes:
            status = "supported_candidate"
            reason = (
                "Chronological out-of-sample Brier improvement and fee-aware paper PnL have "
                "strictly positive 95% intervals."
            )
        elif finalized_days < MIN_ALPHA_STATION_DAYS or not conclusive_modes:
            status = "insufficient_finalized_evidence"
            reason = (
                "Too few independent finalized days, calibrated evaluation days or trade days."
            )
        else:
            status = "not_supported"
            reason = "The available chronological validation does not support positive alpha."
        station_verdicts[station_id] = {
            "status": status,
            "reason": reason,
            "independent_finalized_days": finalized_days,
            "required_finalized_days": MIN_ALPHA_STATION_DAYS,
            "required_calibrated_days": MIN_CALIBRATED_VALIDATION_DAYS,
            "required_trade_days": MIN_TRADE_DAYS,
            "supported_decision_modes": supported_modes,
        }

    supported_stations = [
        station_id
        for station_id, row in station_verdicts.items()
        if row["status"] == "supported_candidate"
    ]
    all_robust = bool(readiness) and all(
        row.get("robust_validation_ready") is True for row in readiness.values()
    )
    if supported_stations:
        predictive_status = "supported_candidate"
        alpha_present: bool | None = True
        predictive_reason = (
            "At least one station passes both probability and fee-aware PnL criteria."
        )
    elif all_robust:
        predictive_status = "not_supported"
        alpha_present = False
        predictive_reason = (
            "Every configured station reached robust validation without supporting alpha."
        )
    else:
        predictive_status = "insufficient_finalized_evidence"
        alpha_present = None
        predictive_reason = (
            "No station yet has enough independent chronological evidence to prove or reject "
            "the master model."
        )

    return {
        "alpha_present": alpha_present,
        "overall_status": (
            "candidate_alpha_detected" if alpha_present is True else
            "no_alpha_detected" if alpha_present is False else
            "not_yet_proven"
        ),
        "blind_revision_trigger": {
            "status": trigger_status,
            "reason": trigger_reason,
            "five_second_executable_outcomes": short_count,
        },
        "station_specific_predictive_model": {
            "status": predictive_status,
            "reason": predictive_reason,
            "required_station_days": MIN_ALPHA_STATION_DAYS,
            "supported_stations": supported_stations,
            "by_station": station_verdicts,
        },
    }


def _markdown_report(summary: dict[str, Any], opportunities: list[dict[str, Any]]) -> str:
    verdict = summary["alpha_verdict"]
    forecast = summary["forecast_hypotheses"]
    live_revisions = summary["live_forecast_revisions"]
    lines = [
        "# Predictive alpha status",
        "",
        f"Included report runs: {', '.join(summary['included_report_runs'])}.",
        "No model fitting was executed by this command; the latest saved master validation was "
        "loaded when available.",
        (
            "Active partial runs included for markouts only: "
            + ", ".join(summary["included_active_partial_runs"])
            + "."
            if summary["included_active_partial_runs"]
            else "No active partial run is included."
        ),
        "",
        "## Verdict",
        "",
        f"Overall status: **{verdict['overall_status']}**.",
        f"Blind revision trigger: **{verdict['blind_revision_trigger']['status']}**.",
        verdict["blind_revision_trigger"]["reason"],
        "Station-specific predictive model: "
        f"**{verdict['station_specific_predictive_model']['status']}**.",
        verdict["station_specific_predictive_model"]["reason"],
        "",
        "| Station | Finalized days | Model status |",
        "|---|---:|---|",
        *[
            f"| {station_id} | {row['independent_finalized_days']} | {row['status']} |"
            for station_id, row in verdict["station_specific_predictive_model"][
                "by_station"
            ].items()
        ],
        "",
        "## Economically actionable forecast revisions",
        "",
        f"Raw revision events: {live_revisions['raw_revision_events']}.",
        "True market-contract changes before provider deduplication: "
        f"{live_revisions['economic_bucket_change_events_before_provider_deduplication']}.",
        f"Equivalent provider duplicates removed: "
        f"{live_revisions['duplicate_equivalent_provider_events_removed']}.",
        f"Actionable revisions: {live_revisions['actionable_revision_events']}.",
        f"Executable t0 quotes: {live_revisions['t0_executable_quotes']}.",
        f"Finalized actionable revisions: {live_revisions['finalized_actionable_revisions']}.",
        f"Correct finalized actionable revisions: "
        f"{live_revisions['correct_actionable_revisions']}.",
        "",
        "| Horizon (s) | Outcomes | Mean bid−t0 ask | Median | Positive share |",
        "|---:|---:|---:|---:|---:|",
        *[
            "| {horizon} | {count} | {mean} | {median} | {positive_share} |".format(
                horizon=horizon,
                count=row["count"],
                mean=_format_number(row.get("mean")),
                median=_format_number(row.get("p50")),
                positive_share=_format_number(row.get("positive_share")),
            )
            for horizon, row in live_revisions[
                "executable_bid_markout_by_horizon_seconds"
            ].items()
        ],
        "",
        "## Finalized forecast hypotheses",
        "",
        f"Independent station-days with final labels: {forecast['independent_station_days']}.",
        "By station: "
        + ", ".join(
            f"{station}={days}"
            for station, days in forecast["independent_station_days_by_station"].items()
        )
        + ".",
        f"Ex-post candidate information leads: {forecast['candidate_information_leads']}.",
        "These candidates are hypothesis generators, not trading signals.",
        "",
        "| Station | Date | Mode | Provider | Forecast | Winner | Winner p | Ask | Persistence | Type | Class |",
        "|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    unique_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in opportunities:
        if row.get("classification") != "candidate_information_lead":
            continue
        key = (row["station_id"], row["local_date"], row["provider"])
        previous = unique_candidates.get(key)
        if previous is None or row["received_wall_ns"] < previous["received_wall_ns"]:
            unique_candidates[key] = row
    for row in sorted(unique_candidates.values(), key=lambda item: item["received_wall_ns"]):
        lines.append(
            "| {station_id} | {local_date} | {decision_mode} | {provider} | "
            "{predicted_bucket} | {winning_bucket} | {probability} | {ask} | "
            "{persistence} | {hypothesis_type} | {classification} |".format(
                **row,
                probability=_format_number(row.get("market_probability_winner")),
                ask=_format_number(row.get("market_ask_winner")),
                persistence=_format_number(row.get("subsequent_prediction_persistence")),
            )
        )
    lines.extend(
        [
            "",
            "## Correct interpretation",
            "",
            "A revision is actionable only when it changes the actual Polymarket bucket, is not "
            "a duplicate provider delivery, has a healthy executable quote and later receives a "
            "final WU label. The surviving hypothesis is a separate probability model per "
            "station. Publication races and already-impossible buckets are excluded because the "
            "recorded evidence did not show an executable strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _normalize(prices: dict[str, float]) -> dict[str, float]:
    clipped = {label: max(value, 0.0) for label, value in prices.items()}
    total = sum(clipped.values())
    return {label: value / total for label, value in clipped.items()} if total else {}


def _json_prices(raw: Any) -> dict[str, float]:
    try:
        payload = json.loads(str(raw or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(label): value
        for label, raw_value in payload.items()
        if (value := _num(raw_value)) is not None
    }


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        if isinstance(value, int):
            return value
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _iso(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC).isoformat()


def _format_number(value: Any) -> str:
    parsed = _num(value)
    return f"{parsed:.3f}" if parsed is not None else "n/a"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
