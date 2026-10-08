from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .analysis.alpha_discovery import discover_alpha
from .analysis.historical import backfill_historical_data
from .analysis.historical_alpha import (
    evaluate_historical_price_screen,
    evaluate_historical_sensitivity,
)
from .analysis.live_costs import estimate_live_execution_costs
from .analysis.master_model import evaluate_master_model
from .analysis.calibration_challenger import evaluate_calibration_challenger
from .analysis.ncei_history import (
    augment_historical_features_with_ghcnh_peers,
    backfill_ghcnh_overlap,
    backfill_ncei_global_hourly,
    build_ncei_pretraining_dataset,
)
from .analysis.paper_trading import (
    fit_calibration_challenger,
    fit_paper_models,
    fit_persistent_disagreement_challenger,
    run_paper_cycle,
)
from .analysis.persistent_disagreement import (
    replay_persistent_disagreement,
)
from .analysis.predictive import build_predictive_dataset
from .analysis.quality import audit_run
from .analysis.resolution_ledger import backfill_resolution_ledger
from .analysis.station_adaptive_alpha import (
    deploy_station_adaptive_policies,
    deploy_station_edge_band_policies,
    evaluate_station_adaptive_policies,
    run_station_adaptive_paper_cycle,
    run_station_edge_band_paper_cycle,
)
from .analysis.station_specific_alpha import (
    deploy_station_specific_policies,
    evaluate_station_specific_policies,
    run_station_specific_paper_cycle,
)
from .analysis.station_maximum_v5 import (
    deploy_station_maximum_v5_shadow,
    evaluate_station_maximum_v5,
    run_station_maximum_v5_shadow_cycle,
)
from .analysis.station_maximum_v6 import (
    deploy_station_maximum_v6_shadow,
    evaluate_station_maximum_v6,
    run_station_maximum_v6_shadow_cycle,
)
from .analysis.station_maximum_v6_experiment import (
    deploy_v6_residual_experiment,
    run_v6_residual_experiment_cycle,
)
from .analysis.station_tail_diagnostics import (
    analyze_station_tail_diagnostics,
)
from .analysis.station_tail_paper_v3 import (
    deploy_station_tail_v3,
    evaluate_station_tail_v3,
    run_station_tail_v3_paper_cycle,
)
from .analysis.station_tail_v4 import (
    deploy_station_tail_v4,
    evaluate_station_tail_v4,
    run_station_tail_v4_paper_cycle,
)
from .collect import run_collection
from .config import AppConfig, load_config
from .doctor import run_doctor
from .logging_setup import configure_logging
from .storage.archive import archive_run, event_files
from .storage.recovery import recover_interrupted_run

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Recorder and alpha tester for Polymarket daily maximum-temperature markets.",
)
console = Console()


def _config(
    app_config: Path,
    stations_config: Path,
    root: Path,
) -> AppConfig:
    root = root.resolve()
    load_dotenv(root / ".env", override=False)
    return load_config(app_config, stations_config, root)


@app.command()
def doctor(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:

    configure_logging(verbose)
    cfg = _config(app_config, stations_config, root)
    checks = asyncio.run(run_doctor(cfg))
    table = Table(title="Weather Alpha Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        status = "OK" if check.ok else ("FAIL" if check.required else "WARN")
        table.add_row(check.name, status, check.detail)
    console.print(table)
    if any(not check.ok and check.required for check in checks):
        raise typer.Exit(code=1)


@app.command()
def collect(
    duration: Annotated[
        float | None,
        typer.Option(help="Stop automatically after this many seconds; omit to run continuously."),
    ] = None,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    archive: Annotated[
        bool,
        typer.Option(
            "--archive/--no-archive", help="Compress and verify raw JSONL after stopping."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:

    configure_logging(verbose)
    cfg = _config(app_config, stations_config, root)
    run_id = asyncio.run(run_collection(cfg, duration_seconds=duration))
    console.print(f"Run completed: [bold]{run_id}[/bold]")
    console.print(f"Raw data: {cfg.data_dir / 'runs' / run_id}")
    if archive:
        _print_archive_summary(archive_run(cfg.data_dir, run_id))


@app.command("stop-collection")
def stop_collection_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest running run.")] = None,
    wait_seconds: Annotated[float, typer.Option(min=1, max=300)] = 120,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    runs_dir = cfg.data_dir / "runs"
    if not runs_dir.exists():
        raise typer.BadParameter("No collection runs were found")
    candidates = [runs_dir / run_id] if run_id else sorted(runs_dir.iterdir(), reverse=True)
    selected: Path | None = None
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "running":
            selected = candidate
            break
    if selected is None:
        raise typer.BadParameter("No running collection was found")

    request_path = selected / "stop.requested"
    temporary = request_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"requested_at": datetime.now(UTC).isoformat(), "requested_by": "weather-alpha"},
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(request_path)
    console.print(f"Graceful stop requested for [bold]{selected.name}[/bold]")

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "running":
            console.print(
                f"Run stopped: status={manifest.get('status')}, "
                f"reason={manifest.get('completion_reason')}"
            )
            return
        time.sleep(0.25)
    raise typer.Exit(code=1)


@app.command("archive-run")
def archive_run_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest run.")] = None,
    keep_originals: Annotated[
        bool,
        typer.Option(help="Keep uncompressed JSONL after the verified gzip copy is written."),
    ] = False,
    compression_level: Annotated[int, typer.Option(min=1, max=9)] = 6,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    summary = archive_run(
        cfg.data_dir,
        run_id,
        keep_originals=keep_originals,
        compression_level=compression_level,
    )
    _print_archive_summary(summary)


@app.command("recover-run")
def recover_run_command(
    run_id: Annotated[str, typer.Option(help="Run whose stale running manifest must be repaired.")],
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    summary = recover_interrupted_run(cfg.data_dir, run_id)
    console.print(
        f"Recovered [bold]{run_id}[/bold]: {summary['verified_files']} files, "
        f"{summary['verified_events']} verified events"
    )


@app.command("build-dataset")
def build_dataset_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest run.")] = None,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = build_predictive_dataset(cfg, run_id)
    _print_paths(paths)


@app.command("discover-alpha")
def discover_alpha_command(
    run_id: Annotated[
        str | None,
        typer.Option(help="Restrict discovery to one run; defaults to all built datasets."),
    ] = None,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = discover_alpha(cfg, run_id)
    _print_paths(paths)


@app.command("audit-run")
def audit_run_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest run.")] = None,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = audit_run(cfg, run_id)
    _print_paths(paths)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if not summary.get("ok"):
        raise typer.Exit(code=1)


@app.command("evaluate-master-model")
def evaluate_master_model_command(
    min_train_days: Annotated[int, typer.Option(min=5)] = 20,
    min_calibration_days: Annotated[int, typer.Option(min=2)] = 10,
    edge_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.04,
    trees: Annotated[int, typer.Option(min=50, max=2000)] = 300,
    max_trades_per_day: Annotated[int, typer.Option(min=0, max=100)] = 1,
    include_ncei_pretraining: Annotated[
        bool, typer.Option(help="Use the evaluation-ineligible long NCEI history when present.")
    ] = True,
    ncei_source_weight: Annotated[float, typer.Option(min=0.01, max=1.0)] = 0.5,
    recency_half_life_years: Annotated[float, typer.Option(min=0.5, max=50.0)] = 6.0,
    evaluation_start_date: Annotated[
        str | None,
        typer.Option(
            help="Evaluate only dates on or after YYYY-MM-DD; prior dates remain training data."
        ),
    ] = None,
    calibration_seed_path: Annotated[
        Path | None,
        typer.Option(
            help="Prior validation CSV used only for calibration before --evaluation-start-date."
        ),
    ] = None,
    calibration_exclude_dates: Annotated[
        str,
        typer.Option(
            help="Comma-separated seed dates to omit after a documented historical audit."
        ),
    ] = "",
    output_tag: Annotated[
        str | None,
        typer.Option(help="Suffix for forward-validation outputs; preserves canonical reports."),
    ] = None,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = evaluate_master_model(
        cfg,
        min_train_days=min_train_days,
        min_calibration_days=min_calibration_days,
        edge_threshold=edge_threshold,
        trees=trees,
        max_trades_per_day=max_trades_per_day,
        include_ncei_pretraining=include_ncei_pretraining,
        ncei_source_weight=ncei_source_weight,
        recency_half_life_years=recency_half_life_years,
        evaluation_start_date=evaluation_start_date,
        calibration_seed_path=calibration_seed_path,
        calibration_exclude_dates={
            value.strip()
            for value in calibration_exclude_dates.split(",")
            if value.strip()
        },
        output_tag=output_tag,
    )
    _print_paths(paths)


@app.command("evaluate-calibration-challenger")
def evaluate_calibration_challenger_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = evaluate_calibration_challenger(cfg)
    _print_paths(paths)


@app.command("fit-paper-model")
def fit_paper_model_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = fit_paper_models(cfg)
    _print_paths(paths)


@app.command("fit-paper-challenger")
def fit_paper_challenger_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = fit_calibration_challenger(cfg)
    _print_paths(paths)


@app.command("paper-cycle")
def paper_cycle_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest run.")] = None,
    rebuild_dataset: Annotated[
        bool,
        typer.Option(
            "--rebuild-dataset/--no-rebuild-dataset",
            help="Rebuild the selected run's point-in-time feature rows before scoring.",
        ),
    ] = True,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = run_paper_cycle(
        cfg,
        run_id=run_id,
        rebuild_dataset=rebuild_dataset,
    )
    _print_paths(paths)


@app.command("paper-challenger-cycle")
def paper_challenger_cycle_command(
    run_id: Annotated[str | None, typer.Option(help="Defaults to the latest run.")] = None,
    rebuild_dataset: Annotated[
        bool,
        typer.Option(
            "--rebuild-dataset/--no-rebuild-dataset",
            help="Rebuild the selected run's point-in-time feature rows before scoring.",
        ),
    ] = True,
    audit_only: Annotated[
        bool,
        typer.Option(
            "--audit-only",
            help=(
                "Build the point-in-time audit for downstream models without "
                "opening new challenger paper entries."
            ),
        ),
    ] = False,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = run_paper_cycle(
        cfg,
        run_id=run_id,
        rebuild_dataset=rebuild_dataset,
        output_namespace="paper_trading_challenger",
        audit_only=audit_only,
    )
    _print_paths(paths)


@app.command("fit-persistent-disagreement")
def fit_persistent_disagreement_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = fit_persistent_disagreement_challenger(cfg)
    _print_paths(paths)


@app.command("replay-persistent-disagreement")
def replay_persistent_disagreement_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = replay_persistent_disagreement(cfg)
    _print_paths(paths)


@app.command("paper-persistent-disagreement-cycle")
def paper_persistent_disagreement_cycle_command(
    run_id: Annotated[
        str | None, typer.Option(help="Defaults to the latest run.")
    ] = None,
    rebuild_dataset: Annotated[
        bool,
        typer.Option(
            "--rebuild-dataset/--no-rebuild-dataset",
            help="Rebuild the selected run's point-in-time rows before scoring.",
        ),
    ] = True,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = run_paper_cycle(
        cfg,
        run_id=run_id,
        rebuild_dataset=rebuild_dataset,
        output_namespace="paper_trading_persistent_disagreement",
    )
    _print_paths(paths)


@app.command("evaluate-station-strategies")
def evaluate_station_strategies_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_specific_policies(cfg))


@app.command("deploy-station-strategies")
def deploy_station_strategies_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_specific_policies(cfg))


@app.command("paper-station-strategies-cycle")
def paper_station_strategies_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_specific_paper_cycle(cfg))


@app.command("evaluate-station-adaptive")
def evaluate_station_adaptive_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_adaptive_policies(cfg))


@app.command("deploy-station-adaptive")
def deploy_station_adaptive_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_adaptive_policies(cfg))


@app.command("paper-station-adaptive-cycle")
def paper_station_adaptive_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_adaptive_paper_cycle(cfg))


@app.command("deploy-station-edge-band")
def deploy_station_edge_band_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_edge_band_policies(cfg))


@app.command("paper-station-edge-band-cycle")
def paper_station_edge_band_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_edge_band_paper_cycle(cfg))


@app.command("analyze-station-tail-v3")
def analyze_station_tail_v3_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(analyze_station_tail_diagnostics(cfg))


@app.command("evaluate-station-tail-v3")
def evaluate_station_tail_v3_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_tail_v3(cfg))


@app.command("deploy-station-tail-v3")
def deploy_station_tail_v3_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_tail_v3(cfg))


@app.command("paper-station-tail-v3-cycle")
def paper_station_tail_v3_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_tail_v3_paper_cycle(cfg))


@app.command("evaluate-station-tail-v4")
def evaluate_station_tail_v4_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_tail_v4(cfg))


@app.command("evaluate-station-maximum-v5")
def evaluate_station_maximum_v5_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_maximum_v5(cfg))


@app.command("evaluate-station-maximum-v6")
def evaluate_station_maximum_v6_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(evaluate_station_maximum_v6(cfg))


@app.command("deploy-station-maximum-v6-shadow")
def deploy_station_maximum_v6_shadow_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_maximum_v6_shadow(cfg))


@app.command("paper-station-maximum-v6-shadow-cycle")
def paper_station_maximum_v6_shadow_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_maximum_v6_shadow_cycle(cfg))


@app.command("backfill-station-maximum-v6-shadow")
def backfill_station_maximum_v6_shadow_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(
        run_station_maximum_v6_shadow_cycle(cfg, backfill_oos=True)
    )


@app.command("deploy-v6-residual-experiment")
def deploy_v6_residual_experiment_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_v6_residual_experiment(cfg))


@app.command("paper-v6-residual-experiment-cycle")
def paper_v6_residual_experiment_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_v6_residual_experiment_cycle(cfg))


@app.command("deploy-station-maximum-v5-shadow")
def deploy_station_maximum_v5_shadow_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_maximum_v5_shadow(cfg))


@app.command("paper-station-maximum-v5-shadow-cycle")
def paper_station_maximum_v5_shadow_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_maximum_v5_shadow_cycle(cfg))


@app.command("deploy-station-tail-v4")
def deploy_station_tail_v4_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(deploy_station_tail_v4(cfg))


@app.command("paper-station-tail-v4-cycle")
def paper_station_tail_v4_cycle_command(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    _print_paths(run_station_tail_v4_paper_cycle(cfg))


@app.command("backfill-history")
def backfill_history_command(
    start_date: Annotated[
        str | None, typer.Option(help="First market date (YYYY-MM-DD); defaults from --days.")
    ] = None,
    end_date: Annotated[
        str | None, typer.Option(help="Last market date (YYYY-MM-DD); defaults to yesterday UTC.")
    ] = None,
    days: Annotated[int, typer.Option(min=30, max=1500)] = 120,
    decision_hours: Annotated[
        str, typer.Option(help="Comma-separated local decision hours, for example 9,12,15.")
    ] = "9,12,15",
    observation_lag_minutes: Annotated[int, typer.Option(min=0, max=60)] = 5,
    refresh: Annotated[bool, typer.Option(help="Redownload cached historical sources.")] = False,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    first, last = _historical_window(start_date, end_date, days)
    hours = _decision_hours(decision_hours)
    paths = asyncio.run(
        backfill_historical_data(
            cfg,
            start_date=first,
            end_date=last,
            decision_hours=hours,
            observation_lag_minutes=observation_lag_minutes,
            refresh=refresh,
        )
    )
    _print_paths(paths)


@app.command("backfill-resolutions")
def backfill_resolutions_command(
    refresh: Annotated[
        bool,
        typer.Option(
            help="Refresh every closed event from Gamma keyset pagination."
        ),
    ] = True,
    comparison_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Chronological model comparison CSV; defaults to the canonical "
                "calibration-challenger report."
            )
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option(min=1, max=12)] = 4,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path(
        "config/app.yaml"
    ),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = asyncio.run(
        backfill_resolution_ledger(
            cfg,
            refresh=refresh,
            comparison_path=comparison_path,
            http_concurrency=concurrency,
        )
    )
    _print_paths(paths)


@app.command("backfill-ncei-history")
def backfill_ncei_history_command(
    start_year: Annotated[int, typer.Option(min=1901, max=2100)] = 2010,
    end_year: Annotated[int, typer.Option(min=1901, max=2100)] = 2025,
    concurrency: Annotated[int, typer.Option(min=1, max=12)] = 4,
    refresh: Annotated[bool, typer.Option(help="Redownload verified cached station-years.")] = False,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    if end_year < start_year:
        raise typer.BadParameter("end-year must not precede start-year")
    cfg = _config(app_config, stations_config, root)
    paths = asyncio.run(
        backfill_ncei_global_hourly(
            cfg,
            start_year=start_year,
            end_year=end_year,
            concurrency=concurrency,
            refresh=refresh,
        )
    )
    _print_paths(paths)


@app.command("build-ncei-pretraining")
def build_ncei_pretraining_command(
    start_year: Annotated[int, typer.Option(min=1901, max=2100)] = 2010,
    end_year: Annotated[int, typer.Option(min=1901, max=2100)] = 2025,
    decision_hours: Annotated[
        str, typer.Option(help="Comma-separated local decision hours, for example 9,12,15.")
    ] = "9,12,15",
    observation_lag_minutes: Annotated[int, typer.Option(min=0, max=60)] = 10,
    max_peer_age_minutes: Annotated[int, typer.Option(min=15, max=360)] = 120,
    minimum_primary_observations: Annotated[int, typer.Option(min=4, max=288)] = 12,
    maximum_primary_gap_minutes: Annotated[int, typer.Option(min=30, max=720)] = 180,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    if end_year < start_year:
        raise typer.BadParameter("end-year must not precede start-year")
    cfg = _config(app_config, stations_config, root)
    paths = build_ncei_pretraining_dataset(
        cfg,
        start_year=start_year,
        end_year=end_year,
        decision_hours=_decision_hours(decision_hours),
        observation_lag_minutes=observation_lag_minutes,
        max_peer_age_minutes=max_peer_age_minutes,
        minimum_primary_observations=minimum_primary_observations,
        maximum_primary_gap_minutes=maximum_primary_gap_minutes,
    )
    _print_paths(paths)


@app.command("backfill-ghcnh-overlap")
def backfill_ghcnh_overlap_command(
    year: Annotated[int, typer.Option(min=2024, max=2100)] = 2026,
    concurrency: Annotated[int, typer.Option(min=1, max=12)] = 4,
    refresh: Annotated[bool, typer.Option(help="Redownload verified GHCNh caches.")] = False,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = asyncio.run(
        backfill_ghcnh_overlap(
            cfg,
            year=year,
            concurrency=concurrency,
            refresh=refresh,
        )
    )
    _print_paths(paths)


@app.command("augment-historical-peers")
def augment_historical_peers_command(
    year: Annotated[int, typer.Option(min=2024, max=2100)] = 2026,
    max_peer_age_minutes: Annotated[int, typer.Option(min=15, max=360)] = 120,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = augment_historical_features_with_ghcnh_peers(
        cfg,
        year=year,
        max_peer_age_minutes=max_peer_age_minutes,
    )
    _print_paths(paths)


@app.command("evaluate-historical-alpha")
def evaluate_historical_alpha_command(
    edge_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.06,
    execution_buffer: Annotated[float, typer.Option(min=0.0, max=0.25)] = 0.02,
    minimum_model_probability: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
    minimum_market_price: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.02,
    maximum_price_age_seconds: Annotated[float, typer.Option(min=60, max=3600)] = 600,
    validation_path: Annotated[
        Path | None,
        typer.Option(help="Master validation CSV; defaults to the canonical full-history report."),
    ] = None,
    output_name: Annotated[
        str,
        typer.Option(help="Output filename prefix inside reports/historical_alpha."),
    ] = "historical_price_screen",
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = evaluate_historical_price_screen(
        cfg,
        validation_path=validation_path,
        edge_threshold=edge_threshold,
        execution_buffer=execution_buffer,
        minimum_model_probability=minimum_model_probability,
        minimum_market_price=minimum_market_price,
        maximum_price_age_seconds=maximum_price_age_seconds,
        output_name=output_name,
    )
    _print_paths(paths)


@app.command("run-historical-research")
def run_historical_research_command(
    start_date: Annotated[str | None, typer.Option()] = None,
    end_date: Annotated[str | None, typer.Option()] = None,
    days: Annotated[int, typer.Option(min=30, max=1500)] = 120,
    refresh: Annotated[bool, typer.Option()] = False,
    min_train_days: Annotated[int, typer.Option(min=5)] = 20,
    min_calibration_days: Annotated[int, typer.Option(min=2)] = 10,
    edge_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.06,
    execution_buffer: Annotated[float, typer.Option(min=0.0, max=0.25)] = 0.02,
    trees: Annotated[int, typer.Option(min=50, max=2000)] = 300,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    first, last = _historical_window(start_date, end_date, days)
    console.print(f"Historical window: [bold]{first}[/bold] through [bold]{last}[/bold]")
    backfill_paths = asyncio.run(
        backfill_historical_data(
            cfg,
            start_date=first,
            end_date=last,
            refresh=refresh,
        )
    )
    _print_paths(backfill_paths)
    model_paths = evaluate_master_model(
        cfg,
        min_train_days=min_train_days,
        min_calibration_days=min_calibration_days,
        edge_threshold=edge_threshold,
        trees=trees,
        max_trades_per_day=0,
    )
    _print_paths(model_paths)
    cost_paths = estimate_live_execution_costs(cfg)
    _print_paths(cost_paths)
    cost_summary = json.loads(cost_paths["summary"].read_text(encoding="utf-8"))
    live_buffer = float(cost_summary["recommended_historical_execution_buffer"])
    alpha_paths = evaluate_historical_price_screen(
        cfg,
        validation_path=model_paths["results"],
        edge_threshold=edge_threshold,
        execution_buffer=max(execution_buffer, live_buffer),
    )
    _print_paths(alpha_paths)


@app.command("estimate-live-costs")
def estimate_live_costs_command(
    shares: Annotated[float, typer.Option(min=1, max=10_000)] = 10.0,
    samples_per_file: Annotated[int, typer.Option(min=100, max=100_000)] = 5_000,
    minimum_price: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.02,
    maximum_price: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.95,
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    if maximum_price <= minimum_price:
        raise typer.BadParameter("maximum-price must exceed minimum-price")
    cfg = _config(app_config, stations_config, root)
    paths = estimate_live_execution_costs(
        cfg,
        shares=shares,
        samples_per_file=samples_per_file,
        minimum_price=minimum_price,
        maximum_price=maximum_price,
    )
    _print_paths(paths)


@app.command("historical-sensitivity")
def historical_sensitivity_command(
    validation_path: Annotated[
        Path | None,
        typer.Option(help="Master validation CSV; defaults to the canonical full-history report."),
    ] = None,
    output_prefix: Annotated[
        str,
        typer.Option(help="Prefix that keeps forward and canonical sensitivity reports separate."),
    ] = "historical",
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    paths = evaluate_historical_sensitivity(
        cfg,
        validation_path=validation_path,
        output_prefix=output_prefix,
    )
    _print_paths(paths)


@app.command("list-runs")
def list_runs(
    app_config: Annotated[Path, typer.Option("--app-config")] = Path("config/app.yaml"),
    stations_config: Annotated[Path, typer.Option("--stations-config")] = Path(
        "config/stations.yaml"
    ),
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:

    cfg = _config(app_config, stations_config, root)
    runs_dir = cfg.data_dir / "runs"
    table = Table(title="Collection runs")
    table.add_column("Run ID")
    table.add_column("Started")
    table.add_column("Files", justify="right")
    if runs_dir.exists():
        for path in sorted((item for item in runs_dir.iterdir() if item.is_dir())):
            manifest_path = path / "manifest.json"
            started = ""
            if manifest_path.exists():
                try:
                    started = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                        "started_at", ""
                    )
                except (json.JSONDecodeError, OSError):
                    started = "invalid manifest"
            table.add_row(path.name, started, str(len(event_files(path))))
    console.print(table)


def _historical_window(
    start_value: str | None, end_value: str | None, days: int
) -> tuple[date, date]:
    try:
        last = date.fromisoformat(end_value) if end_value else datetime.now(UTC).date() - timedelta(days=1)
        first = date.fromisoformat(start_value) if start_value else last - timedelta(days=days - 1)
    except ValueError as exc:
        raise typer.BadParameter("Historical dates must use YYYY-MM-DD") from exc
    if first > last:
        raise typer.BadParameter("start-date must not be after end-date")
    return first, last


def _decision_hours(value: str) -> tuple[int, ...]:
    try:
        hours = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise typer.BadParameter("decision-hours must be comma-separated integers") from exc
    if not hours or any(hour < 0 or hour > 23 for hour in hours):
        raise typer.BadParameter("decision-hours values must be between 0 and 23")
    return hours


def _print_paths(paths: dict[str, Path]) -> None:
    table = Table(title="Generated reports")
    table.add_column("Report")
    table.add_column("Path")
    for name, path in paths.items():
        table.add_row(name, str(path))
    console.print(table)


def _print_archive_summary(summary: dict) -> None:
    original = int(summary["original_bytes"])
    compressed = int(summary["compressed_bytes"])
    ratio = compressed / original if original else 0.0
    console.print(
        f"Archived [bold]{summary['run_id']}[/bold]: {len(summary['files'])} files, "
        f"{original / 1024**2:.1f} MiB -> {compressed / 1024**2:.1f} MiB "
        f"({ratio:.1%})"
    )


if __name__ == "__main__":
    app()
