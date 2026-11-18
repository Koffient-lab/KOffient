"""Measurements on a device: llama-bench for throughput, llama-completion for load time and TTFT.

Peak memory comes from `time -v` (getrusage max RSS of the measured process).
"""

from __future__ import annotations

import json
import re
import shlex
import statistics
from dataclasses import dataclass
from pathlib import Path

from .device import Device

REMOTE_ROOT = "/data/local/tmp/koffient/runner"


class ProbeError(RuntimeError):
    """A measurement step ran but did not produce a usable result."""


@dataclass
class BenchParams:
    threads: int = 4
    n_prompt: tuple[int, ...] = (256, 1024)
    n_gen: int = 128
    repetitions: int = 5
    cold_repetitions: int = 3
    n_ctx: int = 2048
    timeout_s: int = 3600
    flash_attn: str | None = None  # on | off | auto; None leaves llama.cpp's default (auto)
    ubatch: int | None = None  # micro-batch size; None leaves llama.cpp's default (512)

    def engine_flags(self) -> str:
        """Flags understood the same way by llama-bench and llama-completion."""
        flags = []
        if self.flash_attn:
            flags += ["-fa", self.flash_attn]
        if self.ubatch:
            flags += ["-ub", str(self.ubatch)]
        return " ".join(flags)

    def as_record(self) -> dict:
        return {
            "threads": self.threads,
            "n_prompt": list(self.n_prompt),
            "n_gen": self.n_gen,
            "repetitions": self.repetitions,
            "cold_repetitions": self.cold_repetitions,
            "n_ctx": self.n_ctx,
            "warmup": True,
            "flash_attn": self.flash_attn,
            "ubatch": self.ubatch,
        }


def summarize(samples: list[float]) -> dict:
    return {
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
        "samples": [round(s, 3) for s in samples],
    }


def kib_to_mb(kib: int) -> float:
    return round(kib * 1024 / 1e6, 1)


# ---------------------------------------------------------------- parsers


def parse_llama_bench(text: str) -> dict:
    """Split `llama-bench -o json` output into prompt-processing and generation tests."""
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"llama-bench output is not JSON: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ProbeError("llama-bench returned no results")

    prefill, decode = [], []
    for entry in entries:
        stats = summarize(entry["samples_ts"])
        if entry["n_gen"] == 0 and entry["n_prompt"] > 0:
            prefill.append({"n_prompt": entry["n_prompt"], "tps": stats})
        elif entry["n_prompt"] == 0 and entry["n_gen"] > 0:
            decode.append({"n_gen": entry["n_gen"], "tps": stats})
    if not prefill or not decode:
        raise ProbeError("llama-bench output lacks a prompt-only or a generation-only test")
    first = entries[0]
    return {
        "prefill": sorted(prefill, key=lambda item: item["n_prompt"]),
        "decode": sorted(decode, key=lambda item: item["n_gen"]),
        "build_commit": first.get("build_commit"),
        "cpu_info": first.get("cpu_info"),
        "model_type": first.get("model_type"),
    }


def parse_time_v(text: str) -> dict:
    """Read the report toybox `time -v` appends to stderr."""
    rss = re.search(r"Max RSS \(KiB\):\s*(\d+)", text)
    real = re.search(r"Real time \(s\):\s*([\d.]+)", text)
    if not rss:
        raise ProbeError("no 'Max RSS' line in time -v output")
    return {"max_rss_kib": int(rss.group(1)), "real_s": float(real.group(1)) if real else None}


def parse_completion(text: str) -> dict:
    """Read llama.cpp perf timings and the runtime CPU feature line from llama-completion stderr."""
    load = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text)
    prompt = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", text)
    sampling = re.search(r"sampling time\s*=\s*([\d.]+)\s*ms", text)
    if not (load and prompt and sampling):
        raise ProbeError("llama-completion output lacks load/prompt-eval/sampling timings")
    info = re.search(r"system_info:.*?\|\s*CPU\s*:(.*)", text)
    return {
        "load_ms": float(load.group(1)),
        "prompt_eval_ms": float(prompt.group(1)),
        "n_prompt_tokens": int(prompt.group(2)),
        "sampling_ms": float(sampling.group(1)),
        "cpu_features": re.findall(r"(\w+)\s*=\s*1", info.group(1)) if info else [],
    }


# ---------------------------------------------------------------- runners


def _run_step(device: Device, name: str, command: str, raw_dir: Path, timeout_s: int) -> tuple[str, str]:
    """Run one command on the device under `time -v`; keep stdout/stderr in raw_dir."""
    run_dir = f"{REMOTE_ROOT}/run"
    script = (
        f"mkdir -p {run_dir} && cd {run_dir} && "
        f"/system/bin/time -v {command} > {name}.out 2> {name}.err < /dev/null; echo rc=$?"
    )
    result = device.shell(script, timeout=timeout_s)
    out = device.read_file(f"{run_dir}/{name}.out").decode("utf-8", errors="replace")
    err = device.read_file(f"{run_dir}/{name}.err").decode("utf-8", errors="replace")
    (raw_dir / f"{name}.out").write_text(out, encoding="utf-8")
    (raw_dir / f"{name}.err").write_text(err, encoding="utf-8")

    rc = re.search(r"rc=(\d+)", result.stdout)
    if not rc or rc.group(1) != "0":
        code = rc.group(1) if rc else "?"
        tail = " | ".join(err.strip().splitlines()[-3:])[-400:]
        raise ProbeError(f"{name} exited with code {code}: {tail}")
    return out, err


def run_llama_bench(device: Device, bin_dir: str, model: str, params: BenchParams, raw_dir: Path) -> dict:
    prompts = ",".join(str(n) for n in params.n_prompt)
    command = (
        f"{bin_dir}/llama-bench -m {shlex.quote(model)} -t {params.threads} "
        f"-p {prompts} -n {params.n_gen} -r {params.repetitions} -o json {params.engine_flags()}"
    ).rstrip()
    out, err = _run_step(device, "llama-bench", command, raw_dir, params.timeout_s)
    result = parse_llama_bench(out)
    result["peak_rss_kib"] = parse_time_v(err)["max_rss_kib"]
    return result


def run_completion(
    device: Device, bin_dir: str, model: str, prompt: str, params: BenchParams, raw_dir: Path, name: str
) -> dict:
    """One process start to first token: load the model, evaluate the prompt, sample one token."""
    command = (
        f"{bin_dir}/llama-completion -m {shlex.quote(model)} -t {params.threads} -c {params.n_ctx} "
        f"-n 1 --temp 0 -s 1 -no-cnv --no-display-prompt -f {shlex.quote(prompt)} {params.engine_flags()}"
    ).rstrip()
    _out, err = _run_step(device, name, command, raw_dir, params.timeout_s)
    result = parse_completion(err)
    result["ttft_ms"] = result["prompt_eval_ms"] + result["sampling_ms"]
    result["peak_rss_kib"] = parse_time_v(err)["max_rss_kib"]
    return result


def measure(device: Device, bin_dir: str, model: str, prompt: str, params: BenchParams, raw_dir: Path,
            metrics: dict, rooted: bool) -> dict:
    """Fill `metrics` step by step (so a failure keeps what was already measured). Returns run-time facts."""
    bench = run_llama_bench(device, bin_dir, model, params, raw_dir)
    metrics["prefill"] = bench["prefill"]
    metrics["decode"] = bench["decode"]
    metrics["bench_peak_rss_mb"] = kib_to_mb(bench["peak_rss_kib"])

    # The model file is in the page cache after llama-bench, so these runs are warm.
    warm = [
        run_completion(device, bin_dir, model, prompt, params, raw_dir, f"completion-warm-{i}")
        for i in range(params.repetitions)
    ]
    metrics["ttft_ms"] = {"warm": summarize([r["ttft_ms"] for r in warm]), "cold": None}
    metrics["load_ms"] = {"warm": summarize([r["load_ms"] for r in warm]), "cold": None}
    metrics["peak_rss_mb"] = kib_to_mb(max(r["peak_rss_kib"] for r in warm))

    if rooted and params.cold_repetitions > 0:
        cold = []
        for i in range(params.cold_repetitions):
            device.drop_caches()
            cold.append(run_completion(device, bin_dir, model, prompt, params, raw_dir, f"completion-cold-{i}"))
        metrics["ttft_ms"]["cold"] = summarize([r["ttft_ms"] for r in cold])
        metrics["load_ms"]["cold"] = summarize([r["load_ms"] for r in cold])

    return {
        "build_commit": bench["build_commit"],
        "cpu_features": warm[0]["cpu_features"],
        "n_prompt_tokens": warm[0]["n_prompt_tokens"],
    }
