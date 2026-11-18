"""Run records: one JSON line per measured configuration, appended to runs.jsonl."""

from __future__ import annotations

import functools
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RUNS_FILE = "runs.jsonl"

_REQUIRED = {
    "schema_version": int,
    "run_id": str,
    "started_at": str,
    "layer": str,
    "status": str,
    "tooling_commit": str,
    "model": dict,
    "engine": dict,
    "device": dict,
    "params": dict,
    "metrics": dict,
    "raw_dir": str,
}
_OK_METRICS = ("prefill", "decode", "ttft_ms", "load_ms", "peak_rss_mb")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._]+", "-", text.lower()).strip("-")


def new_run_id(model: dict, engine: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    parts = [model["name"], model.get("quant_label") or "", engine["build_id"]]
    return "-".join([now.strftime("%Y%m%dT%H%M%SZ")] + [slug(p) for p in parts if p])


@functools.lru_cache(maxsize=1)
def tooling_commit() -> str:
    """Commit of this tooling, with '-dirty' when tooling/ has uncommitted changes.

    Read once per process: the code that runs is the code loaded at start, even if files change later.
    """
    tooling_dir = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(["git", "-C", str(tooling_dir), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unknown"
    dirty = git("status", "--porcelain", "--", ".")
    return f"{commit}-dirty" if dirty else commit


def validate(record: dict) -> None:
    for key, expected in _REQUIRED.items():
        if key not in record:
            raise ValueError(f"record is missing '{key}'")
        if not isinstance(record[key], expected):
            raise ValueError(f"record field '{key}' must be {expected.__name__}")
    if record["layer"] not in ("engine", "sdk"):
        raise ValueError("layer must be 'engine' or 'sdk'")
    if record["status"] == "ok":
        missing = [m for m in _OK_METRICS if record["metrics"].get(m) is None]
        if missing:
            raise ValueError(f"status is ok but metrics are missing: {missing}")
    elif record["status"] == "failed":
        if not record.get("error"):
            raise ValueError("a failed record needs an 'error'")
    else:
        raise ValueError("status must be 'ok' or 'failed'")
    for key in ("sha256", "file_bytes", "bits_per_param", "tensor_types_mb"):
        if key not in record["model"]:
            raise ValueError(f"model block is missing '{key}'")


def append(record: dict, results_dir: Path) -> Path:
    validate(record)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RUNS_FILE
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load(results_dir: Path) -> list[dict]:
    path = results_dir / RUNS_FILE
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
