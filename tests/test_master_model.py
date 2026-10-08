from __future__ import annotations

import csv
import gzip
import json
from datetime import date, timedelta

import pytest

from weather_alpha.analysis.master_model import (
    conformal_final_distribution,
    evaluate_master_model,
    forecast_anchored_conformal_distribution,
    remaining_max_distribution,
    remaining_max_samples,
)
from weather_alpha.config import load_config


def test_remaining_maximum_support_is_monotonic_and_normalized() -> None:
    samples = remaining_max_samples(84.0, [-2.0, 0.5, 2.0])
    distribution = remaining_max_distribution(
        84.0,
        0.6,
        [-2.0, 0.5, 2.0],
        ["84°F", "85°F", "86°F or higher"],
        "F",
    )

    assert min(samples) >= 84.0
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["84°F"] >= 0.4


def test_conformal_distribution_uses_valid_support_and_smooths_sparse_days() -> None:
    labels = ["84\N{DEGREE SIGN}F", "85\N{DEGREE SIGN}F", "86\N{DEGREE SIGN}F or higher"]
    distribution = conformal_final_distribution(
        84.0,
        85.0,
        [-3.0] * 9 + [1.0],
        labels,
        "F",
    )

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution[labels[0]] < 0.9
    assert distribution[labels[1]] > 0.0


def test_forecast_anchored_calibration_uses_prior_skill_and_rejects_no_improvement() -> None:
    labels = ["80\N{DEGREE SIGN}F", "81\N{DEGREE SIGN}F", "82\N{DEGREE SIGN}F or higher"]
    forecast_skilled = [
        {
            "test_date": f"2026-06-{day:02d}",
            "decision_mode": "scheduled",
            "local_hour": 1.5,
            "raw_point_final": 90.0,
            "forecast_point_final": 80.0,
            "current_max": 70.0,
            "final_max": 80.0,
        }
        for day in range(1, 21)
    ]
    distribution, diagnostics = forecast_anchored_conformal_distribution(
        70.0,
        90.0,
        80.0,
        forecast_skilled,
        labels,
        "F",
        local_hour=1.5,
        decision_mode="scheduled",
    )

    assert diagnostics["challenger_accepted"] is True
    assert diagnostics["model_weight"] < 0.5
    assert sum(distribution.values()) == pytest.approx(1.0)

    raw_model_skilled = [
        {
            **row,
            "raw_point_final": 80.0,
            "forecast_point_final": 90.0,
        }
        for row in forecast_skilled
    ]
    _, rejected = forecast_anchored_conformal_distribution(
        70.0,
        80.0,
        90.0,
        raw_model_skilled,
        labels,
        "F",
        local_hour=1.5,
        decision_mode="scheduled",
    )

    assert rejected["challenger_accepted"] is False


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_master_model_evaluates_scheduled_and_event_modes(tmp_path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    cfg = load_config(project_root=".").model_copy(
        update={"reports_dir": tmp_path / "reports", "data_dir": tmp_path / "data"}
    )
    start = date(2026, 7, 1)
    fields = [
        "run_id",
        "station_id",
        "city",
        "source",
        "decision_mode",
        "trigger_source",
        "received_wall_ns",
        "observation_time",
        "observation_key",
        "local_date",
        "day_history_complete",
        "continuity_eligible",
        "local_hour",
        "temperature_c",
        "running_max_market_unit",
        "minutes_since_last_observation",
        "market_distribution_json",
        "market_bid_json",
        "market_ask_json",
        "market_fee_rate_json",
        "market_fee_exponent_json",
        "market_fees_enabled_json",
        "market_ws_healthy",
        "market_quote_coverage",
        "winning_bucket",
        "final_wu_max",
    ]
    for offset in range(10):
        day = start + timedelta(days=offset)
        final_value = 85.0 if offset % 2 else 84.0
        winner = "85°F" if final_value == 85.0 else "84°F"
        run_dir = cfg.reports_dir / f"run-{offset:02d}"
        run_dir.mkdir(parents=True)
        rows = []
        for mode_index, mode in enumerate(("event_driven", "scheduled")):
            rows.append(
                {
                    "run_id": f"run-{offset:02d}",
                    "station_id": "KLGA",
                    "city": "NYC",
                    "source": "awc_metar" if mode == "event_driven" else "scheduled",
                    "decision_mode": mode,
                    "trigger_source": "awc_metar" if mode == "event_driven" else "timer",
                    "received_wall_ns": 1_000_000_000_000 + offset * 10 + mode_index,
                    "observation_time": f"{day.isoformat()}T16:00:00+00:00",
                    "observation_key": f"{mode}|{day.isoformat()}",
                    "local_date": day.isoformat(),
                    "day_history_complete": True,
                    "continuity_eligible": True,
                    "local_hour": 12.0 + mode_index * 0.25,
                    "temperature_c": 28.0 + offset * 0.05,
                    "running_max_market_unit": 84.0,
                    "minutes_since_last_observation": mode_index * 15,
                    "market_distribution_json": json.dumps({"84°F": 0.5, "85°F": 0.5}),
                    "market_bid_json": json.dumps({"84°F": 0.45, "85°F": 0.45}),
                    "market_ask_json": json.dumps({"84°F": 0.55, "85°F": 0.55}),
                    "market_fee_rate_json": json.dumps({"84°F": 0.0, "85°F": 0.0}),
                    "market_fee_exponent_json": json.dumps({"84°F": 1.0, "85°F": 1.0}),
                    "market_fees_enabled_json": json.dumps({"84°F": False, "85°F": False}),
                    "market_ws_healthy": True,
                    "market_quote_coverage": 1.0,
                    "winning_bucket": winner,
                    "final_wu_max": final_value,
                }
            )
        with (run_dir / "predictive_features.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    pretraining_dir = cfg.reports_dir / "ncei_history"
    pretraining_dir.mkdir(parents=True)
    pretraining_fields = [
        "station_id",
        "city",
        "decision_mode",
        "received_wall_ns",
        "observation_key",
        "local_date",
        "day_history_complete",
        "continuity_eligible",
        "evaluation_eligible",
        "target_source",
        "local_hour",
        "temperature_c",
        "running_max_market_unit",
        "resolution_running_max_market_unit",
        "final_temperature_target",
    ]
    with gzip.open(
        pretraining_dir / "pretraining_features.csv.gz",
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pretraining_fields)
        writer.writeheader()
        for offset in range(6):
            day = start - timedelta(days=6 - offset)
            writer.writerow(
                {
                    "station_id": "KLGA",
                    "city": "NYC",
                    "decision_mode": "historical_pretraining",
                    "received_wall_ns": 900_000_000_000 + offset,
                    "observation_key": f"ncei|{day.isoformat()}",
                    "local_date": day.isoformat(),
                    "day_history_complete": True,
                    "continuity_eligible": True,
                    "evaluation_eligible": False,
                    "target_source": "ncei_global_hourly",
                    "local_hour": 12.0,
                    "temperature_c": 27.0 + offset * 0.1,
                    "running_max_market_unit": 82.0,
                    "resolution_running_max_market_unit": 82.0,
                    "final_temperature_target": 83.0 + offset % 2,
                }
            )

    paths = evaluate_master_model(
        cfg,
        min_train_days=5,
        min_calibration_days=2,
        edge_threshold=0.01,
        trees=50,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    modes = {item["decision_mode"] for item in summary["station_modes"]}
    assert modes == {"event_driven", "scheduled"}
    assert paths["importance"].exists()
    assert paths["readiness"].exists()
    readiness = next(
        item for item in summary["model_readiness"] if item["station_id"] == "KLGA"
    )
    assert readiness["finalized_station_days"] == 10
    assert readiness["training_station_days"] == 16
    assert readiness["pretraining_station_days"] == 6
    assert readiness["evaluation_station_days"] == 10
    assert readiness["chronological_validation_ready"] is True
    assert readiness["recommended_research_ready"] is False
    with paths["results"].open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    assert results
    assert all(row["test_date"] >= start.isoformat() for row in results)
    for row in results:
        assert float(row["point_prediction"]) >= float(row["current_running_max"])
        probabilities = json.loads(row["model_distribution_json"])
        assert sum(probabilities.values()) == pytest.approx(1.0)

    forward_paths = evaluate_master_model(
        cfg,
        min_train_days=5,
        min_calibration_days=2,
        edge_threshold=0.01,
        trees=50,
        evaluation_start_date=(start + timedelta(days=8)).isoformat(),
        calibration_seed_path=paths["results"],
        calibration_exclude_dates={(start + timedelta(days=2)).isoformat()},
        output_tag="forward_test",
    )
    assert forward_paths["results"].name.endswith("_forward_test.csv")
    forward_summary = json.loads(
        forward_paths["summary"].read_text(encoding="utf-8")
    )
    assert forward_summary["calibration_seed_rows"] > 0
    assert forward_summary["calibration_exclude_dates"] == [
        (start + timedelta(days=2)).isoformat()
    ]
    with forward_paths["results"].open(newline="", encoding="utf-8") as handle:
        forward_results = list(csv.DictReader(handle))
    assert forward_results
    assert all(
        row["test_date"] >= (start + timedelta(days=8)).isoformat()
        for row in forward_results
    )
