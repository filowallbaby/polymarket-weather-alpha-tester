from __future__ import annotations

from datetime import date, timedelta

import pytest

import weather_alpha.analysis.station_maximum_v6_experiment as experiment
from weather_alpha.analysis.station_maximum_v6_experiment import (
    _evaluate_episode,
    _experiment_summary,
    _weather_taker_fee,
)


def _protocol() -> dict:
    windows = {
        station_id: {
            "timezone": "UTC",
            "first_eligible_local_date": "2026-08-11",
            "last_eligible_local_date": "2026-08-30",
        }
        for station_id in experiment.EXPECTED_STATION_IDS
    }
    return {
        "experiment_id": "frozen",
        "experiment_name": experiment.EXPERIMENT_NAME,
        "deployed_at_ns": 100,
        "station_ids": list(experiment.EXPECTED_STATION_IDS),
        "station_windows": windows,
        "collection_end_utc": "2026-08-31T07:10:00+00:00",
        "signal_rule": {
            "minimum_model_probability": 0.40,
            "minimum_v6_minus_normalized_market_mid": 0.15,
        },
        "cost_rule": {"flat_execution_buffer_per_share": 0.02},
        "data_quality_rule": {
            "minimum_v6_prediction_rows_per_station_day": 80,
            "minimum_accepted_v6_rows_per_station_day": 40,
            "minimum_local_hour_span": 20.0,
        },
    }


def _episode(**overrides: object) -> dict:
    row = {
        "episode_id": "episode",
        "start_decision_id": "decision",
        "evaluation_phase": "prospective",
        "station_id": "EHAM",
        "local_date": "2026-08-11",
        "episode_start_wall_ns": 200,
        "signal_variant": "model_modal_bucket",
        "active_confirmation_count": 1,
        "trigger_source": "forecast_ecmwf",
        "information_class": "forecast_provider_revision",
        "target_bucket": "25C",
        "initial_model_probability": 0.40,
        "initial_weather_market_gap": 0.15,

        "winning_bucket": "25C",
        "target_won": True,
    }
    row.update(overrides)
    return row


def test_weather_fee_uses_frozen_polymarket_formula() -> None:
    assert _weather_taker_fee(0.20) == pytest.approx(0.008)


def test_episode_boundary_is_inclusive_and_outcome_blind() -> None:
    audit, trade = _evaluate_episode(
        _episode(), _protocol(), traded_days=set()
    )

    assert audit["eligible"] is True
    assert audit["outcome_used_for_selection"] is False
    assert trade is not None
    assert trade["normalized_market_mid"] == pytest.approx(0.25)
    assert trade["fee_per_share"] == pytest.approx(0.009375)
    assert trade["assumed_cost_per_share"] == pytest.approx(0.279375)
    assert "winning_bucket" not in trade

    losing_audit, losing_trade = _evaluate_episode(
        _episode(winning_bucket="26C", target_won=False),
        _protocol(),
        traded_days=set(),
    )
    assert losing_audit["eligible"] is True
    assert losing_trade is not None
    assert losing_trade["assumed_cost_per_share"] == trade[
        "assumed_cost_per_share"
    ]


def test_second_episode_for_station_day_cannot_open() -> None:
    audit, trade = _evaluate_episode(
        _episode(episode_id="second"),
        _protocol(),
        traded_days={("EHAM", "2026-08-11")},
    )

    assert trade is None
    assert audit["eligible"] is False
    assert "station_day_entry_already_taken" in audit["rejection_reasons"]


def test_thresholds_do_not_round_up_near_misses() -> None:
    _, low_probability = _evaluate_episode(
        _episode(initial_model_probability=0.399999),
        _protocol(),
        traded_days=set(),
    )
    _, low_gap = _evaluate_episode(
        _episode(initial_weather_market_gap=0.149999),
        _protocol(),
        traded_days=set(),
    )

    assert low_probability is None
    assert low_gap is None


def test_decision_is_locked_until_all_80_days_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "BOOTSTRAP_SAMPLES", 500)
    summary = _experiment_summary(
        _protocol(),
        [],
        [
            {
                "station_id": "EHAM",
                "local_date": "2026-08-11",
                "official_winning_bucket": "25C",
                "coverage_valid": True,
            }
        ],
    )

    assert summary["evaluation_complete"] is False
    assert summary["decision_locked_until_complete"] is True
    assert summary["alpha_demonstrated"] is False


def test_completed_uniformly_positive_holdout_passes_global_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "BOOTSTRAP_SAMPLES", 500)
    protocol = _protocol()
    opportunities = []
    trades = []
    first = date.fromisoformat("2026-08-11")
    for station_id in experiment.EXPECTED_STATION_IDS:
        for offset in range(20):
            local_date = (first + timedelta(days=offset)).isoformat()
            opportunities.append(
                {
                    "station_id": station_id,
                    "local_date": local_date,
                    "official_winning_bucket": "25C",
                    "coverage_valid": True,
                }
            )
            trades.append(
                {
                    "station_id": station_id,
                    "local_date": local_date,
                    "status": "paper_settled",
                    "won": True,
                    "normalized_market_mid": 0.20,
                    "model_probability": 0.50,
                    "realized_pnl_per_share": 0.77,
                }
            )

    summary = _experiment_summary(protocol, trades, opportunities)

    assert summary["evaluation_complete"] is True
    assert summary["global_gate_passed"] is True
    assert summary["alpha_demonstrated"] is True
    assert summary["decision"] == "eligible_for_separate_micro_stake_design"
