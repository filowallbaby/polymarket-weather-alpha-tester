from __future__ import annotations

import gzip
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson


def archive_run(
    data_dir: Path,
    run_id: str | None = None,
    *,
    keep_originals: bool = False,
    compression_level: int = 6,
) -> dict[str, Any]:

    if not 1 <= compression_level <= 9:
        raise ValueError("compression_level must be between 1 and 9")
    run_dir = _resolve_run_dir(data_dir, run_id)
    manifest_path = run_dir / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    if manifest.get("status") not in {"completed", "cancelled", "interrupted"}:
        raise ValueError(
            "Only completed, gracefully cancelled, or recovered interrupted runs can be archived; "
            f"{run_dir.name} is {manifest.get('status')!r}"
        )

    files: list[dict[str, Any]] = []
    for source in sorted(run_dir.glob("*.jsonl")):
        files.append(
            compress_event_file(
                source,
                compression_level=compression_level,
                keep_original=keep_originals,
            )
        )

    original_bytes = sum(item["uncompressed_bytes"] for item in files)
    compressed_bytes = sum(item["compressed_bytes"] for item in files)
    archive_metadata = {
        "archived_at": datetime.now(UTC).isoformat(),
        "compression": "gzip",
        "compression_level": compression_level,
        "originals_kept": keep_originals,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "files": files,
    }
    manifest["archive"] = archive_metadata
    _write_json_atomic(manifest_path, manifest)
    return {"run_id": run_dir.name, **archive_metadata}


def compress_event_file(
    source: Path,
    *,
    compression_level: int = 6,
    keep_original: bool = False,
) -> dict[str, Any]:

    if not 1 <= compression_level <= 9:
        raise ValueError("compression_level must be between 1 and 9")
    if not source.name.endswith(".jsonl"):
        raise ValueError(f"Expected a .jsonl file: {source}")
    destination = source.with_name(f"{source.name}.gz")
    source_hash, source_bytes = _file_hash(source)
    if destination.exists():
        archived_hash, archived_bytes = _gzip_content_hash(destination)
        if archived_hash != source_hash or archived_bytes != source_bytes:
            raise ValueError(f"Existing archive does not match source: {destination}")
    else:
        _compress_atomic(source, destination, compression_level)
        archived_hash, archived_bytes = _gzip_content_hash(destination)
        if archived_hash != source_hash or archived_bytes != source_bytes:
            destination.unlink(missing_ok=True)
            raise OSError(f"Archive verification failed: {destination}")
    result = {
        "source": source.name,
        "archive": destination.name,
        "sha256_uncompressed": source_hash,
        "uncompressed_bytes": source_bytes,
        "compressed_bytes": destination.stat().st_size,
    }
    if not keep_original:
        source.unlink()
    return result


def event_files(run_dir: Path) -> list[Path]:

    plain = {path.name: path for path in run_dir.glob("*.jsonl")}
    compressed = {path.name.removesuffix(".gz"): path for path in run_dir.glob("*.jsonl.gz")}
    return sorted(
        (plain | {name: path for name, path in compressed.items() if name not in plain}).values()
    )


def open_event_file(path: Path):
    if path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _compress_atomic(source: Path, destination: Path, compression_level: int) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compression_level,
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output:
                while chunk := input_handle.read(1024 * 1024):
                    compressed_output.write(chunk)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _gzip_content_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_run_dir(data_dir: Path, run_id: str | None) -> Path:
    runs_dir = data_dir / "runs"
    if run_id:
        path = runs_dir / run_id
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_dir}")
    return candidates[-1]
