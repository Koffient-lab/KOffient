"""Quality records: one JSON line per model and task (or per model and text corpus), in quality.jsonl.

Speed measurements live in runs.jsonl; the two files are joined by the model's sha256.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1
QUALITY_FILE = "quality.jsonl"

_REQUIRED = {
    "schema_version": int,
    "eval_id": str,
    "started_at": str,
    "kind": str,
    "status": str,
    "tooling_commit": str,
    "model": dict,
    "engine": dict,
    "params": dict,
    "metrics": dict,
    "raw_dir": str,
}
_KIND_BLOCK = {"task": "task", "closeness": "corpus"}


def validate(record: dict) -> None:
    for key, expected in _REQUIRED.items():
        if key not in record:
            raise ValueError(f"record is missing '{key}'")
        if not isinstance(record[key], expected):
            raise ValueError(f"record field '{key}' must be {expected.__name__}")
    if record["kind"] not in _KIND_BLOCK:
        raise ValueError("kind must be 'task' or 'closeness'")
    block = _KIND_BLOCK[record["kind"]]
    if not isinstance(record.get(block), dict):
        raise ValueError(f"a '{record['kind']}' record needs a '{block}' block")
    if "sha256" not in record["model"]:
        raise ValueError("model block is missing 'sha256'")
    if record["status"] == "ok":
        if record["kind"] == "task" and record["metrics"].get("value") is None:
            raise ValueError("status is ok but the task metric has no value")
        if record["kind"] == "closeness" and record["metrics"].get("ppl") is None:
            raise ValueError("status is ok but perplexity is missing")
    elif record["status"] == "failed":
        if not record.get("error"):
            raise ValueError("a failed record needs an 'error'")
    else:
        raise ValueError("status must be 'ok' or 'failed'")


def append(record: dict, results_dir: Path) -> Path:
    validate(record)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / QUALITY_FILE
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load(results_dir: Path) -> list[dict]:
    path = results_dir / QUALITY_FILE
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def variant_key(record: dict) -> str:
    """Identity of a measured variant: the model file plus an optional label (e.g. another thread count)."""
    return f"{record['model']['sha256']}|{record.get('label') or ''}"


def latest_ok(records: list[dict], kind: str) -> dict[tuple[str, str], dict]:
    """The most recent successful record for every (variant, task or corpus language)."""
    latest: dict[tuple[str, str], dict] = {}
    for record in records:
        if record["kind"] != kind or record["status"] != "ok":
            continue
        name = record["task"]["name"] if kind == "task" else record["corpus"]["lang"]
        latest[(variant_key(record), name)] = record
    return latest
