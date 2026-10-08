from __future__ import annotations

from weather_alpha.resources import process_memory


def test_process_memory_reports_current_pid_and_nonnegative_counters() -> None:
    metrics = process_memory()
    assert isinstance(metrics["pid"], int)
    for name in (
        "private_bytes",
        "working_set_bytes",
        "peak_private_bytes",
        "peak_working_set_bytes",
    ):
        value = metrics[name]
        assert value is None or value >= 0
