"""Closeness of a model variant to its fp16 base: perplexity and KL divergence via llama-perplexity.

The base model's logits over a fixed text are saved once; every variant is then compared against that file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PERPLEXITY_BINARY = "llama-perplexity.exe"
N_CTX = 512
# Vocabulary of ~152k tokens * 2 bytes per logit: about 0.3 MB per scored token, 3 GB for 40 chunks.
DEFAULT_CHUNKS = 40


class ClosenessError(RuntimeError):
    pass


def _number(pattern: str, text: str) -> float | None:
    found = re.search(pattern, text)
    return float(found.group(1)) if found else None


def parse_base(text: str) -> dict:
    ppl = _number(r"Final estimate: PPL = ([\d.]+)", text)
    if ppl is None:
        raise ClosenessError("no perplexity estimate in the llama-perplexity output (NaN logits?)")
    return {"ppl": ppl, "ppl_err": _number(r"Final estimate: PPL = [\d.]+ \+/- ([\d.]+)", text)}


def parse_kld(text: str) -> dict:
    metrics = {
        "ppl": _number(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", text),
        "ppl_base": _number(r"Mean PPL\(base\)\s*:\s*([\d.]+)", text),
        "ppl_ratio": _number(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)", text),
        "ppl_ratio_err": _number(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*[\d.]+\s*±\s*([\d.]+)", text),
        "kld_mean": _number(r"Mean\s+KLD:\s*([\d.]+)", text),
        "kld_mean_err": _number(r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)", text),
        "kld_median": _number(r"Median\s+KLD:\s*([\d.]+)", text),
        "kld_p99": _number(r"99\.0%\s+KLD:\s*([\d.]+)", text),
        "delta_p_mean_pct": _number(r"Mean\s+Δp:\s*(-?[\d.]+)", text),
        "delta_p_rms_pct": _number(r"RMS Δp\s*:\s*([\d.]+)", text),
        "same_top_pct": _number(r"Same top p:\s*([\d.]+)", text),
    }
    missing = [key for key in ("ppl", "kld_mean", "same_top_pct") if metrics[key] is None]
    if missing:
        raise ClosenessError(f"llama-perplexity output has no {missing}")
    return metrics


def _run(command: list[str], log_path: Path, timeout_s: int) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, capture_output=True, timeout=timeout_s)
    text = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    log_path.write_text(text, encoding="utf-8", newline="\n")
    if proc.returncode != 0:
        raise ClosenessError(f"llama-perplexity exited with code {proc.returncode}, see {log_path}")
    return text


def _gpu(gpu_layers: str | None) -> list[str]:
    return ["-ngl", gpu_layers] if gpu_layers is not None else []


def save_base_logits(engine_dir: Path, model: Path, corpus: Path, logits: Path, log_path: Path, *,
                     chunks: int = DEFAULT_CHUNKS, threads: int = 16, gpu_layers: str | None = None,
                     timeout_s: int = 3600) -> dict:
    logits.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and kept only when the run gave a finite perplexity: a half-written or NaN
    # logits file would otherwise be picked up by every later comparison as if it were a valid base.
    partial = logits.with_name(logits.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        text = _run([str(engine_dir / PERPLEXITY_BINARY), "-m", str(model), "-f", str(corpus), "-c", str(N_CTX),
                     "--chunks", str(chunks), "-t", str(threads), "--kl-divergence-base", str(partial),
                     *_gpu(gpu_layers)], log_path, timeout_s)
        metrics = parse_base(text)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(logits)
    return metrics


def compare_with_base(engine_dir: Path, model: Path, logits: Path, log_path: Path, *,
                      chunks: int = DEFAULT_CHUNKS, threads: int = 16, gpu_layers: str | None = None,
                      timeout_s: int = 3600) -> dict:
    text = _run([str(engine_dir / PERPLEXITY_BINARY), "-m", str(model), "-c", str(N_CTX), "--chunks", str(chunks),
                 "-t", str(threads), "--kl-divergence-base", str(logits), "--kl-divergence", *_gpu(gpu_layers)],
                log_path, timeout_s)
    return parse_kld(text)
