from __future__ import annotations

from weather_alpha.analysis.alpha_discovery import (
    _actionable_revision_rows,
    _alpha_verdict,
    _bucket_for_temperature,
    _live_revision_summary,
    _opportunity_classification,
)


def test_bucket_mapping_converts_celsius_forecast_for_fahrenheit_market() -> None:
    labels = {"79°F or below", "80°F", "81°F or higher"}

    assert _bucket_for_temperature(labels, 26.7, "F") == "80°F"
    assert _bucket_for_temperature(labels, 25.0, "F") == "79°F or below"


def test_candidate_requires_price_headroom_and_stable_prediction() -> None:
    assert _opportunity_classification(0.30, 0.34, 0.80, 5_000) == (
        "candidate_information_lead"
    )
    assert _opportunity_classification(0.60, 0.62, 1.0, -1_000) == "market_already_high"
    assert _opportunity_classification(0.30, 0.34, 0.20, 5_000) == "unstable_forecast"


def test_actionable_revisions_use_market_labels_and_deduplicate_providers() -> None:
    rows = [
        _revision_row(
            source="forecast_best_match",
            provider="best_match",
            timestamp=100_000_000_000,
            previous_label="82-83°F",
            current_label="84-85°F",
            horizon=0,
        ),
        _revision_row(
            source="forecast_best_match",
            provider="best_match",
            timestamp=100_000_000_000,
            previous_label="82-83°F",
            current_label="84-85°F",
            horizon=5,
        ),
        _revision_row(
            source="forecast_gfs_hrrr",
            provider="gfs_hrrr",
            timestamp=102_000_000_000,
            previous_label="82-83°F",
            current_label="84-85°F",
            horizon=0,
        ),
        _revision_row(
            source="forecast_gfs_hrrr",
            provider="gfs_hrrr",
            timestamp=102_000_000_000,
            previous_label="82-83°F",
            current_label="84-85°F",
            horizon=5,
        ),
        _revision_row(
            source="forecast_best_match",
            provider="best_match",
            timestamp=200_000_000_000,
            previous_label="88-89°F",
            current_label="88-89°F",
            horizon=0,
        ),
    ]

    actionable = _actionable_revision_rows(rows)

    assert len(actionable) == 2
    assert {row["horizon_seconds"] for row in actionable} == {0, 5}
    assert actionable[0]["provider_confirmation_count"] == 2
    assert "gfs_hrrr" in actionable[0]["confirming_providers"]
    summary = _live_revision_summary(actionable, raw_rows=rows)
    assert summary["raw_revision_events"] == 3
    assert summary["economic_bucket_change_events_before_provider_deduplication"] == 2
    assert summary["duplicate_equivalent_provider_events_removed"] == 1
    assert summary["actionable_revision_events"] == 1
    assert summary["executable_bid_markout_by_horizon_seconds"]["5"]["count"] == 1


def test_alpha_verdict_does_not_claim_alpha_without_finalized_days() -> None:
    verdict = _alpha_verdict(
        {"independent_station_days": 3},
        {
            "finalized_actionable_revisions": 2,
            "executable_bid_markout_by_horizon_seconds": {
                "5": {"count": 20, "positive_share": 0.0, "p50": -0.01}
            },
            "gross_resolution_pnl_per_share_before_fees": {
                "count": 2,
                "mean": 0.1,
                "positive_share": 1.0,
            },
        },
    )

    assert verdict["alpha_present"] is None
    assert verdict["overall_status"] == "not_yet_proven"
    assert verdict["blind_revision_trigger"]["status"] == "rejected"


def test_alpha_verdict_keeps_station_days_separate() -> None:
    verdict = _alpha_verdict(
        {
            "independent_station_days": 20,
            "independent_station_days_by_station": {"EHAM": 10, "EGLC": 10},
        },
        {
            "executable_bid_markout_by_horizon_seconds": {},
            "gross_resolution_pnl_per_share_before_fees": {},
        },
    )

    assert verdict["alpha_present"] is None
    assert verdict["station_specific_predictive_model"]["by_station"]["EHAM"][
        "independent_finalized_days"
    ] == 10
    assert verdict["station_specific_predictive_model"]["by_station"]["EGLC"][
        "independent_finalized_days"
    ] == 10


def test_alpha_verdict_requires_positive_probability_and_pnl_intervals() -> None:
    verdict = _alpha_verdict(
        {
            "independent_station_days": 60,
            "independent_station_days_by_station": {"EHAM": 60},
        },
        {
            "executable_bid_markout_by_horizon_seconds": {},
            "gross_resolution_pnl_per_share_before_fees": {},
        },
        {
            "status": "available",
            "model_readiness": [
                {
                    "station_id": "EHAM",
                    "finalized_station_days": 60,
                    "robust_validation_ready": False,
                }
            ],
            "station_modes": [
                {
                    "station_id": "EHAM",
                    "decision_mode": "event_driven",
                    "status": "evaluated",
                    "calibrated_days": 25,
                    "trade_days": 22,
                    "brier_improvement_ci95": {"low": 0.01, "high": 0.04},
                    "mean_pnl_per_station_day_ci95": {"low": 0.005, "high": 0.03},
                }
            ],
        },
    )

    assert verdict["alpha_present"] is True
    assert verdict["station_specific_predictive_model"]["supported_stations"] == [
        "EHAM"
    ]


def _revision_row(
    *,
    source: str,
    provider: str,
    timestamp: int,
    previous_label: str,
    current_label: str,
    horizon: int,
) -> dict:
    return {
        "run_id": "run",
        "report_run_id": "run",
        "source": source,
        "provider": provider,
        "station_id": "KLGA",
        "local_date": "2026-07-22",
        "revision_received_wall_ns": timestamp,
        "previous_bucket_label": previous_label,
        "current_bucket_label": current_label,
        "previous_today_max_c": 28.0,
        "current_today_max_c": 28.9,
        "previous_remaining_max_c": 28.0,
        "current_remaining_max_c": 28.9,
        "bucket_changed": True,
        "horizon_seconds": horizon,
        "market_ask": 0.25,
        "market_ask_size": 100.0,
        "t0_market_ws_healthy": True,
        "market_ws_healthy": True,
        "executable_bid_markout_from_t0_ask": -0.01 if horizon else None,
        "revision_correct": "",
    }
