from __future__ import annotations

from weather_alpha.analysis.station_specific_alpha import (
    STATION_POLICIES,
    _load_csv,
    evaluate_station_policy_snapshot,
)


def test_load_csv_accepts_large_embedded_json_field(tmp_path) -> None:
    path = tmp_path / "resolution_ledger.csv"
    payload = "x" * (256 * 1024)
    path.write_text(f'id,contracts_json\n1,"{payload}"\n', encoding="utf-8")

    rows = _load_csv(path)

    assert rows == [{"id": "1", "contracts_json": payload}]


def test_amsterdam_policy_selects_weather_top_on_late_disagreement() -> None:
    result = evaluate_station_policy_snapshot(
        STATION_POLICIES["EHAM"],
        {"20°C": 0.55, "21°C": 0.45},
        {"20°C": 0.25, "21°C": 0.75},
    )

    assert result["eligible"] is True
    assert result["candidate_bucket"] == "20°C"
    assert result["candidate_price_proxy"] == 0.25


def test_lax_policy_requires_late_high_confidence_convergence() -> None:
    eligible = evaluate_station_policy_snapshot(
        STATION_POLICIES["KLAX"],
        {"78-79°F": 0.85, "80-81°F": 0.15},
        {"78-79°F": 0.90, "80-81°F": 0.10},
    )
    disagreement = evaluate_station_policy_snapshot(
        STATION_POLICIES["KLAX"],
        {"78-79°F": 0.85, "80-81°F": 0.15},
        {"78-79°F": 0.40, "80-81°F": 0.60},
    )

    assert eligible["eligible"] is True
    assert eligible["candidate_bucket"] == "78-79°F"
    assert disagreement["eligible"] is False
    assert "model_market_not_converged" in disagreement["rejection_reasons"]


def test_london_market_follow_keeps_station_active_without_forcing_trade() -> None:
    result = evaluate_station_policy_snapshot(
        STATION_POLICIES["EGLC"],
        {"24°C": 0.60, "25°C": 0.40},
        {"24°C": 0.35, "25°C": 0.65},
    )
    too_expensive = evaluate_station_policy_snapshot(
        STATION_POLICIES["EGLC"],
        {"24°C": 0.60, "25°C": 0.40},
        {"24°C": 0.20, "25°C": 0.80},
    )

    assert result["eligible"] is True
    assert result["candidate_bucket"] == "25°C"
    assert too_expensive["eligible"] is False
    assert (
        "price_proxy_outside_station_rule"
        in too_expensive["rejection_reasons"]
    )
