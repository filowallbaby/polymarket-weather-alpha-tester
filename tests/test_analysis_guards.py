from __future__ import annotations

from weather_alpha.analysis.predictive import _final_wu_max


def test_unfinalized_wu_snapshot_cannot_become_training_label() -> None:
    snapshots = [
        {
            "received_wall_ns": 1,
            "payload": {"daily_max_from_network": 20.0, "resolution_final": False},
        },
        {
            "received_wall_ns": 2,
            "payload": {"daily_max_from_network": 21.0, "resolution_final": True},
        },
    ]
    assert _final_wu_max(snapshots) == 21.0
