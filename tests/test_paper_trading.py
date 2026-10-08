from __future__ import annotations

import json

import pytest

from weather_alpha.analysis.paper_trading import (
    PERSISTENT_DISAGREEMENT_RULES,
    PERSISTENT_DISAGREEMENT_STRATEGY,
    StationPaperModel,
    _evaluate_record,
    advance_persistence,
)
from weather_alpha.config import load_config


def test_persistence_requires_time_and_confirmations_and_resets() -> None:
    state = {}
    first = advance_persistence(
        state,
        station_id="KLGA",
        local_date="2026-07-28",
        timestamp_ns=1_000_000_000,
        bucket="84\u00b0F",
        eligible=True,
        minimum_confirmations=2,
        persistence_minutes=15,
    )
    second = advance_persistence(
        state,
        station_id="KLGA",
        local_date="2026-07-28",
        timestamp_ns=901_000_000_000,
        bucket="84\u00b0F",
        eligible=True,
        minimum_confirmations=2,
        persistence_minutes=15,
    )
    reset = advance_persistence(
        state,
        station_id="KLGA",
        local_date="2026-07-28",
        timestamp_ns=902_000_000_000,
        bucket=None,
        eligible=False,
        minimum_confirmations=2,
        persistence_minutes=15,
    )

    assert first["persistent"] is False
    assert second["persistent"] is True
    assert second["confirmations"] == 2
    assert reset["persistent"] is False
    assert state == {}


def test_live_candidate_charges_fee_depth_and_incremental_p90_buffer(
    monkeypatch,
) -> None:
    cfg = load_config(project_root=".")
    station = cfg.stations["nyc"]
    calibration = [
        {
            "test_date": f"2026-06-{day:02d}",
            "decision_mode": "scheduled",
            "local_hour": 12.0,
            "rise_residual": 0.0,
            "increment_residual": 0.0,
            "final_residual": 0.0,
        }
        for day in range(1, 21)
    ]
    model = StationPaperModel(
        station_id="KLGA",
        market_unit="F",
        feature_columns=[],
        imputer=None,
        rise_classifier=None,
        constant_rise_probability=1.0,
        increment_regressor=None,
        fallback_positive_increments=[1.0],
        calibration_history=calibration,
        training_rows=100,
        training_days=100,
        trained_through_local_date="2026-07-27",
    )
    monkeypatch.setattr(
        "weather_alpha.analysis.paper_trading._predict_remaining_maximum",
        lambda row, fitted: (1.0, 0.0, [0.0]),
    )
    label = "84\u00b0F"
    row = {
        "day_history_complete": True,
        "continuity_eligible": True,
        "market_ws_healthy": True,
        "market_quote_coverage": 1.0,
        "minutes_since_last_observation": 5.0,
        "model_provider_count": 3,
        "model_best_match_age_minutes": 5.0,
        "model_ecmwf_age_minutes": 5.0,
        "model_gfs_hrrr_age_minutes": 5.0,
        "model_spread_remaining_max_c": 0.5,
        "peer_station_coverage": 0.5,
        "market_distribution_json": json.dumps({label: 0.39}),
        "market_ask_json": json.dumps({label: 0.40}),
        "market_ask_size_json": json.dumps({label: 20.0}),
        "market_fee_rate_json": json.dumps({label: 0.0}),
        "market_fee_exponent_json": json.dumps({label: 1.0}),
        "market_fees_enabled_json": json.dumps({label: False}),
        "resolution_running_max_market_unit": 84.0,
        "local_hour": 12.0,
        "decision_mode": "scheduled",
    }

    result = _evaluate_record(row, station, model, cfg.paper_trading, 0.02)

    assert result["rejection_reasons"] == ""
    assert result["candidate_bucket"] == label
    assert result["candidate_additional_slippage_buffer"] == pytest.approx(0.01)
    assert result["candidate_conservative_cost"] == pytest.approx(0.41)

    row["market_ask_size_json"] = json.dumps({label: 2.0})
    shallow = _evaluate_record(row, station, model, cfg.paper_trading, 0.02)
    assert "insufficient_best_ask_depth" in shallow["rejection_reasons"]


def test_persistent_disagreement_is_midprice_consensus_supported_and_outcome_blind(
    monkeypatch,
) -> None:
    cfg = load_config(project_root=".")
    station = cfg.stations["nyc"]
    calibration = [
        {
            "test_date": f"2026-06-{day:02d}",
            "decision_mode": "scheduled",
            "local_hour": 12.0,
            "rise_residual": 0.0,
            "increment_residual": 0.0,
            "final_residual": 0.0,
        }
        for day in range(1, 21)
    ]
    model = StationPaperModel(
        station_id="KLGA",
        market_unit="F",
        feature_columns=[],
        imputer=None,
        rise_classifier=None,
        constant_rise_probability=1.0,
        increment_regressor=None,
        fallback_positive_increments=[1.0],
        calibration_history=calibration,
        training_rows=100,
        training_days=100,
        trained_through_local_date="2026-07-27",
    )
    labels = ["80-81\u00b0F", "82-83\u00b0F", "84\u00b0F or higher"]
    monkeypatch.setattr(
        "weather_alpha.analysis.paper_trading._predict_remaining_maximum",
        lambda row, fitted: (1.0, 2.0, [2.0]),
    )
    monkeypatch.setattr(
        "weather_alpha.analysis.paper_trading.conformal_final_distribution",
        lambda *args, **kwargs: {
            labels[0]: 0.20,
            labels[1]: 0.55,
            labels[2]: 0.25,
        },
    )
    row = {
        "day_history_complete": True,
        "continuity_eligible": True,
        "market_ws_healthy": True,
        "market_quote_coverage": 1.0,
        "minutes_since_last_observation": 5.0,
        "model_provider_count": 3,
        "model_best_match_age_minutes": 5.0,
        "model_ecmwf_age_minutes": 5.0,
        "model_gfs_hrrr_age_minutes": 5.0,
        "model_best_match_remaining_max_c": 28.0,
        "model_ecmwf_remaining_max_c": 28.0,
        "model_gfs_hrrr_remaining_max_c": 29.0,
        "model_consensus_remaining_max_c": 28.0,
        "model_spread_remaining_max_c": 1.0,
        "peer_station_coverage": 0.5,
        "market_distribution_json": json.dumps(
            {labels[0]: 0.25, labels[1]: 0.30, labels[2]: 0.45}
        ),
        "market_ask_json": json.dumps(
            {labels[0]: 0.27, labels[1]: 0.32, labels[2]: 0.47}
        ),
        "market_ask_size_json": json.dumps(
            {labels[0]: 20.0, labels[1]: 20.0, labels[2]: 20.0}
        ),
        "market_fee_rate_json": json.dumps({label: 0.0 for label in labels}),
        "market_fee_exponent_json": json.dumps(
            {label: 1.0 for label in labels}
        ),
        "market_fees_enabled_json": json.dumps(
            {label: False for label in labels}
        ),
        "resolution_running_max_market_unit": 80.0,
        "local_hour": 12.0,
        "decision_mode": "scheduled",
    }
    strategy = {
        **PERSISTENT_DISAGREEMENT_RULES,
        "weather_calibration_strategy": "legacy_oos_conformal",
    }

    result = _evaluate_record(
        row,
        station,
        model,
        cfg.paper_trading,
        0.02,
        calibration_strategy=PERSISTENT_DISAGREEMENT_STRATEGY,
        strategy_parameters=strategy,
    )

    assert result["rejection_reasons"] == ""
    assert result["hypothesis_only"] is True
    assert result["candidate_bucket"] == labels[1]
    assert result["candidate_provider_votes"] == 2
    assert result["consensus_forecast_bucket"] == labels[1]
    assert result["candidate_market_probability"] == pytest.approx(0.30)
    assert result["candidate_weather_market_gap"] == pytest.approx(0.25)

    row["market_distribution_json"] = json.dumps(
        {labels[0]: 0.70, labels[1]: 0.05, labels[2]: 0.25}
    )
    row["market_ask_json"] = json.dumps(
        {labels[0]: 0.72, labels[1]: 0.07, labels[2]: 0.27}
    )
    cheap_tail = _evaluate_record(
        row,
        station,
        model,
        cfg.paper_trading,
        0.02,
        calibration_strategy=PERSISTENT_DISAGREEMENT_STRATEGY,
        strategy_parameters=strategy,
    )
    assert cheap_tail["candidate_bucket"] is None
    assert (
        "no_persistent_midprice_disagreement"
        in cheap_tail["rejection_reasons"]
    )
