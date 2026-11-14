"""Command line entry point: `python -m koffient_tooling bench run|show`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from . import record as rec
from .bench import REMOTE_ROOT, BenchParams, ProbeError, measure
from .device import Device, DeviceError
from .engine import BINARIES, EngineError, describe_build
from .model import describe_model, sha256_file


def _work_dir(args) -> Path | None:
    value = args.work_dir or os.environ.get("KOFFIENT_WORK")
    return Path(value) if value else None


def _results_dir(args) -> Path:
    if args.results_dir:
        return Path(args.results_dir)
    work = _work_dir(args)
    if not work:
        raise SystemExit("error: pass --results-dir or --work-dir (or set KOFFIENT_WORK)")
    return work / "results"


def _build_dir(args) -> Path:
    direct = Path(args.build)
    if direct.is_dir():
        return direct
    work = _work_dir(args)
    if work and (work / "build" / args.build).is_dir():
        return work / "build" / args.build
    raise SystemExit(f"error: build '{args.build}' not found (looked for a directory and under <work-dir>/build)")


def cmd_run(args) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"error: model file not found: {model_path}")
    build_dir = _build_dir(args)
    results_dir = _results_dir(args)
    params = BenchParams(
        threads=args.threads,
        n_prompt=tuple(int(n) for n in args.n_prompt.split(",")),
        n_gen=args.n_gen,
        repetitions=args.repetitions,
        cold_repetitions=0 if args.skip_cold else args.cold_repetitions,
        n_ctx=args.n_ctx,
        timeout_s=args.timeout,
    )

    try:
        device = Device(args.device)
        device_facts = device.facts()
        engine = describe_build(build_dir)
    except (DeviceError, EngineError) as exc:
        raise SystemExit(f"error: {exc}")
    if engine["abi"] != device_facts["abi"]:
        raise SystemExit(f"error: build is for {engine['abi']}, device is {device_facts['abi']}")

    print(f"model passport: {model_path.name}", flush=True)
    model = describe_model(model_path, source=args.source)
    started = datetime.now(timezone.utc)
    run_id = rec.new_run_id(model, engine, started)
    raw_dir = results_dir / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "schema_version": rec.SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer": "engine",
        "status": "failed",
        "error": None,
        "tooling_commit": rec.tooling_commit(),
        "model": model,
        "engine": engine,
        "device": device_facts,
        "params": params.as_record(),
        "metrics": {},
        "raw_dir": f"raw/{run_id}",
    }

    bin_dir = f"{REMOTE_ROOT}/bin/{engine['build_id']}"
    remote_model = f"{REMOTE_ROOT}/models/{model['sha256'][:12]}-{model_path.name}"
    remote_prompt = f"{REMOTE_ROOT}/prompts/ttft_en.txt"
    try:
        print(f"device: {device.serial} ({device_facts['kind']}, {device_facts['abi']}); pushing files", flush=True)
        for name in BINARIES:
            device.push_if_changed(build_dir / "bin" / name, f"{bin_dir}/{name}",
                                   engine["binaries_sha256"][name], executable=True)
        device.push_if_changed(model_path, remote_model, model["sha256"])
        with resources.as_file(resources.files("koffient_tooling") / "prompts" / "ttft_en.txt") as prompt_file:
            device.push_if_changed(prompt_file, remote_prompt, sha256_file(prompt_file))

        print(f"measuring: run {run_id}", flush=True)
        facts = measure(device, bin_dir, remote_model, remote_prompt, params, raw_dir,
                        record["metrics"], rooted=device_facts["rooted"])
        record["params"]["ttft_prompt_tokens"] = facts["n_prompt_tokens"]
        record["engine"]["runtime_cpu_features"] = facts["cpu_features"]
        if facts["build_commit"] and engine["commit"] and not engine["commit"].startswith(facts["build_commit"]) \
                and not facts["build_commit"].startswith(engine["commit"]):
            # The binary was built from another commit than the current checkout: trust the binary.
            record["engine"]["commit"] = facts["build_commit"]
            record["engine"]["tag"] = None
        record["status"] = "ok"
    except (DeviceError, ProbeError) as exc:
        record["error"] = str(exc)

    path = rec.append(record, results_dir)
    print(f"{record['status']}: appended to {path}")
    if record["status"] != "ok":
        print(f"reason: {record['error']}", file=sys.stderr)
        return 1
    _print_table([record])
    return 0


def _fmt(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _row(r: dict) -> list[str]:
    m, model = r["metrics"], r["model"]
    prefill = {p["n_prompt"]: p["tps"]["median"] for p in m.get("prefill") or []}
    decode = (m.get("decode") or [{}])[0].get("tps", {}).get("median")
    ttft, load = m.get("ttft_ms") or {}, m.get("load_ms") or {}

    def med(block, key):
        return (block.get(key) or {}).get("median")

    return [
        r["started_at"][:16].replace("T", " "),
        f"{model['name']} {model.get('quant_label') or ''}".strip(),
        _fmt(model["file_bytes"] / 1e6, 0),
        _fmt(model.get("bits_per_param"), 2),
        r["engine"]["build_id"],
        r["device"]["kind"],
        "/".join(_fmt(prefill.get(n), 0) for n in sorted(prefill)) or "-",
        _fmt(decode),
        _fmt(med(ttft, "warm"), 0),
        f"{_fmt(med(load, 'warm'), 0)}/{_fmt(med(load, 'cold'), 0)}",
        _fmt(m.get("peak_rss_mb"), 0),
        r["status"],
    ]


def _print_table(records: list[dict]) -> None:
    header = ["started (UTC)", "model", "file MB", "bits/param", "build", "device",
              "prefill t/s", "decode t/s", "ttft ms", "load ms warm/cold", "peak MB", "status"]
    rows = [header] + [_row(r) for r in records]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def cmd_show(args) -> int:
    records = rec.load(_results_dir(args))
    if not records:
        print("no runs recorded yet")
        return 0
    _print_table(records)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="koffient", description="Koffient model tooling")
    groups = parser.add_subparsers(dest="group", required=True)
    bench = groups.add_parser("bench", help="on-device benchmarks").add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--work-dir", help="folder with build/ and results/ (default: $KOFFIENT_WORK)")
        sub.add_argument("--results-dir", help="where runs.jsonl lives (default: <work-dir>/results)")

    run = bench.add_parser("run", help="measure one model on one build on one device")
    common(run)
    run.add_argument("--model", required=True, help="path to a GGUF file")
    run.add_argument("--build", required=True, help="build directory, or its name under <work-dir>/build")
    run.add_argument("--device", help="adb serial (default: the only connected device)")
    run.add_argument("--source", help="where the model file came from, e.g. hf:org/repo@revision")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--n-prompt", default="256,1024", help="prompt lengths for the throughput test")
    run.add_argument("--n-gen", type=int, default=128)
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--cold-repetitions", type=int, default=3)
    run.add_argument("--skip-cold", action="store_true", help="skip cold-start runs (they need root)")
    run.add_argument("--n-ctx", type=int, default=2048, help="context size for the load/TTFT/memory runs")
    run.add_argument("--timeout", type=int, default=3600, help="seconds per measurement step")
    run.set_defaults(func=cmd_run)

    show = bench.add_parser("show", help="print recorded runs as a table")
    common(show)
    show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
