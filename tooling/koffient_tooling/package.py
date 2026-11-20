"""A release package of one model file for the SDK: the GGUF, the source licence and manifest.json.

Everything in the manifest comes from our own records (builds.jsonl, quality.jsonl, runs.jsonl), so a package can be
rebuilt and checked. The manifest layout is our proposal for the SDK side; the SDK's own model format may replace it.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from . import builds
from . import record as runs
from .model import sha256_file

MANIFEST_VERSION = 1

# Runtime settings every package carries: found by measurement on the emulator and on a OnePlus 9R
# (docs/models/METHODOLOGY.md).
COMMON_RUNTIME = {
    "engine": "llama.cpp",
    "engine_tag": "b12199",
    "load_mode": "none",  # LLAMA_LOAD_MODE_NONE: with mmap the repacked weights and the file pages are both resident
    "flash_attn": "on",  # required by the quantised V cache below
    "n_ctx": 4096,
    "kv_cache_type": "q8_0",  # K and V; quality unchanged on the fast set, -160/-220 MB at n_ctx 4096
    "kv_cache_note": "q8_0 slows reading a prompt on a phone CPU by 25-39%; with the prompt cache that hits only the "
                     "one-off computation of the fixed prefix and long new inputs (tool results, history). Use f16 if "
                     "tool results are long and memory allows",
    "threads": "number of big cores (4 on Snapdragon 870); little cores slow generation down (8 threads: -39%)",
    "sampling": {"temperature": 0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                 "dry_multiplier": 0.0},
    "sampling_note": "set every sampling parameter explicitly: the GGUF carries the vendor's generation_config "
                     "defaults, and llama.cpp applies them when a request leaves a parameter unset",
    "arm_build": "GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 (no i8mm: phones such as Snapdragon 870 have none)",
    "chat_template": "embedded in the GGUF (tokenizer.chat_template, Jinja)",
    "prompt_cache": "keep the model loaded and the KV state of the fixed prefix (system prompt, tool definitions); "
                    "save it to a file (llama_state_seq_save_file) and restore it after a restart "
                    "(llama_state_seq_load_file). On a OnePlus 9R the six-tool request goes from 35 s to ~1 s to the "
                    "first token; saving takes ~0.1 s, restoring ~0.05 s",
}


def _fetch_record(records: list[dict], build: dict) -> dict | None:
    """Walk input links from a build back to the fetch of its source."""
    current = build
    for _ in range(10):
        if current["kind"] == "fetch":
            return current
        parent_id = (current.get("input") or {}).get("build_id") or (current.get("input") or {}).get("fetch_build_id")
        current = next((r for r in records if r["build_id"] == parent_id), None)
        if current is None:
            return None
    return None


def _quality(results: Path, sha: str) -> dict:
    from .quality import record as quality
    from .quality.run import load_answers, score_block, split_answers
    from .quality.tasks import TASKS

    out: dict = {"tasks": {}, "closeness": {}}
    for (key, name), r in sorted(quality.latest_ok(quality.load(results), "task").items()):
        if not key.startswith(sha) or name not in TASKS:
            continue
        headline, _ = split_answers(TASKS[name], load_answers(results / r["raw_dir"]))
        block = score_block(list(headline.values()))
        out["tasks"][name] = {"value": block["value"], "ci95": block["ci95"], "n": block["n"],
                              "engine": r["engine"].get("backend"), "eval_id": r["eval_id"]}
    for (key, lang), r in quality.latest_ok(quality.load(results), "closeness").items():
        if key.startswith(sha) and "kld_mean" in r["metrics"]:
            out["closeness"][lang] = {"kld_mean": r["metrics"]["kld_mean"],
                                      "same_top_pct": r["metrics"]["same_top_pct"], "base": r["base"]["file"],
                                      "eval_id": r["eval_id"]}
    return out


def _invalid_runs(results: Path) -> set[str]:
    """run_ids marked invalid in invalid.jsonl (the same file marks bad quality records by eval_id)."""
    path = results / "invalid.jsonl"
    if not path.is_file():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {entry["run_id"] for entry in map(json.loads, filter(str.strip, handle)) if "run_id" in entry}


def _speed(results: Path, sha: str) -> list[dict]:
    out = []
    invalid = _invalid_runs(results)
    for r in runs.load(results):
        if r["status"] != "ok" or r["model"].get("sha256") != sha or r["run_id"] in invalid:
            continue
        m = r["metrics"]
        out.append({
            "run_id": r["run_id"], "label": r.get("label"), "device": r["device"].get("kind"),
            "speed_is_phone_grade": r["device"].get("speed_is_phone_grade"), "build": r["engine"]["build_id"],
            "load_mode": r["params"].get("load_mode"), "n_ctx": r["params"].get("n_ctx"),
            "prefill_tps": {str(p["n_prompt"]): p["tps"]["median"] for p in m.get("prefill") or []},
            "decode_tps": (m.get("decode") or [{}])[0].get("tps", {}).get("median"),
            "ttft_ms_warm": (m.get("ttft_ms") or {}).get("warm", {}).get("median"),
            "peak_rss_mb": m.get("peak_rss_mb"),
        })
    return out


def package(model_path: Path, role: str, out_root: Path, results: Path, *, notes: list[str] | None = None) -> Path:
    records = builds.load(results)
    sha = sha256_file(model_path)
    build = builds.find_by_sha(records, sha)
    if build is None:
        raise builds.BuildError(f"{model_path} has no build record: package only our own builds")
    fetch = _fetch_record(records, build)
    passport = build["output"]["passport"]
    package_id = f"{runs.slug(passport['name'])}-{build['recipe']}"
    out = out_root / package_id
    out.mkdir(parents=True, exist_ok=True)
    target = out / model_path.name
    if not target.exists():
        try:
            os.link(model_path, target)  # same volume: no second copy on disk
        except OSError:
            shutil.copy2(model_path, target)
    licence_file = None
    licence = (fetch or {}).get("source_repo", {}).get("license")
    if fetch is not None:
        src = Path(fetch["output"]["dir"])
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            if (src / name).is_file():
                shutil.copy2(src / name, out / "LICENSE")
                licence_file = "LICENSE (from the source repository)"
                break
    if licence_file is None and licence == "apache-2.0":
        # The repository states the licence only in its model card: ship the standard text, as Apache-2.0 requires.
        with resources.as_file(resources.files("koffient_tooling") / "licenses" / "Apache-2.0.txt") as text:
            shutil.copy2(text, out / "LICENSE")
        licence_file = "LICENSE (standard Apache-2.0 text; the source repository states the licence in its card)"
    speed = _speed(results, sha)
    measured = [s for s in speed if s["load_mode"] == "none" and s["peak_rss_mb"]]
    peak = [s["peak_rss_mb"] for s in measured]
    # The KV cache grows with n_ctx, so the peak holds only for the context it was measured at.
    peak_ctx = max((s["n_ctx"] for s in measured if s["n_ctx"]), default=None)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "id": package_id,
        "role": role,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tooling_commit": runs.tooling_commit(),
        "model": {
            "name": passport["name"], "architecture": passport.get("architecture"), "params": passport.get("params"),
            "source": build["source"], "licence": licence, "licence_file": licence_file,
        },
        "file": {"name": model_path.name, "bytes": model_path.stat().st_size, "sha256": sha, "format": "gguf",
                 "recipe": build["recipe"], "bits_per_param": passport.get("bits_per_param"),
                 "build_id": build["build_id"]},
        "runtime": {**COMMON_RUNTIME,
                    "min_ram_mb_measured": max(peak) if peak else None,
                    "min_ram_mb_measured_at_n_ctx": peak_ctx},
        "measured": {"quality": _quality(results, sha), "speed": speed},
        "notes": notes or [],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
                                       newline="\n")
    return out
