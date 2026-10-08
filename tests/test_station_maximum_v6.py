from __future__ import annotations

import json

import pytest

from weather_alpha.analysis.station_maximum_v6 import (
    _expert_mixture_weight,
    _fit_provider_stack,
    _historical_provider_values,
    _live_provider_values,
    _predict_provider_stack,
    _timing_analysis,
)


def test_provider_stack_learns_provider_correction_from_prior_days() -> None:
    rows = [
        {
            "local_date": f"2026-06-{day:02d}",
            "base_point_c": 20.0,
            "providers": [21.0, 22.0, 23.0],
            "target_c": 22.0,
        }
        for day in range(1, 21)
    ]

    model = _fit_provider_stack(rows)
    prediction = _predict_provider_stack(
        model,
        {
            "base_point_c": 20.0,
            "running_max_c": 19.0,
            "providers": [21.0, 22.0, 23.0],
        },
    )

    assert model is not None
    assert prediction == pytest.approx(22.0, abs=0.1)


def test_expert_weight_accepts_helpful_and_rejects_harmful_provider() -> None:
    helpful = [
        {
            "local_date": f"2026-07-{day:02d}",
            "v5_point_c": 20.0,
            "provider_point_c": 22.0,
            "target_c": 22.0,
        }
        for day in range(1, 13)
    ]
    harmful = [
        {
            **row,
            "target_c": 19.0,
        }
        for row in helpful
    ]

    assert 0.0 < _expert_mixture_weight(helpful) <= 1.0
    assert _expert_mixture_weight(harmful) == 0.0


def test_historical_provider_mapping_is_station_specific() -> None:
    row = {
        "historical_forecast_best_match_day1_remaining_max_c": 20.0,
        "historical_forecast_dwd_icon_day1_remaining_max_c": 21.0,
        "historical_forecast_ecmwf_day1_remaining_max_c": 22.0,
        "historical_forecast_gfs_hrrr_day1_remaining_max_c": 23.0,
    }

    assert _historical_provider_values(row, "EHAM") == [
        20.0,
        21.0,
        22.0,
    ]
    assert _historical_provider_values(row, "KLAX") == [
        20.0,
        22.0,
        23.0,
    ]


def test_live_provider_mapping_prefers_remaining_max() -> None:
    row = {
        "model_best_match_remaining_max_c": 20.0,
        "model_best_match_today_max_c": 30.0,
        "model_ecmwf_remaining_max_c": 21.0,
        "model_gfs_hrrr_remaining_max_c": 22.0,
    }

    assert _live_provider_values(row, "KLAX") == [20.0, 21.0, 22.0]


def test_timing_analysis_reports_model_lead_without_oracle_selection() -> None:
    rows = [
        {
            "station_id": "KLAX",
            "local_date": "2026-07-01",
            "local_hour": hour,
            "received_wall_ns": timestamp,
            "winning_bucket": "80-81°F",
            "v6_distribution_json": json.dumps(model),
            "market_distribution_json": json.dumps(market),
        }
        for hour, timestamp, model, market in [
            (
                9,
                1_000_000_000,
                {"80-81°F": 0.6, "82-83°F": 0.4},
                {"80-81°F": 0.3, "82-83°F": 0.7},
            ),
            (
                12,
                10_801_000_000_000,
                {"80-81°F": 0.7, "82-83°F": 0.3},
                {"80-81°F": 0.7, "82-83°F": 0.3},
            ),
        ]
    ]

    events, summary = _timing_analysis(rows)

    assert events[0]["v6_correct_before_market"] is True
    assert events[0]["disagreement_signal_correct"] is True
    assert summary["v6_correct_before_market_events"] == 1
    assert summary["median_v6_lead_vs_market_hours"] == pytest.approx(3.0)
