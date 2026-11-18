"""Command line entry point: `python -m koffient_tooling bench run|show|report`, `... model fetch|convert|quantize|show`,
`... quality run|closeness|show`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from . import record as rec
from .bench import REMOTE_ROOT, BenchParams, ProbeError, measure
from .builds import attach
from .device import Device, DeviceError
from .engine import BINARIES, EngineError, describe_build
from .model import describe_model, sha256_file, variant_name


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
        flash_attn=args.flash_attn,
        ubatch=args.ubatch,
        load_mode=args.load_mode,
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
    model = attach(describe_model(model_path, source=args.source), results_dir)
    started = datetime.now(timezone.utc)
    run_id = rec.new_run_id(model, engine, started) + (f"-{rec.slug(args.label)}" if args.label else "")
    raw_dir = results_dir / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "schema_version": rec.SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer": "engine",
        "label": args.label,
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
        f"{model['name']} {variant_name(model)} {r.get('label') or ''}".strip(),
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


def cmd_report(args) -> int:
    from . import report  # imported here: matplotlib is an optional dependency

    results_dir = _results_dir(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(args.out) if args.out else results_dir / "reports" / f"{stamp}-{rec.slug(args.name)}"
    print(f"report written to {report.build(results_dir, out_dir, args.base)}")
    return 0


def _need_work(args) -> Path:
    work = _work_dir(args)
    if not work:
        raise SystemExit("error: pass --work-dir or set KOFFIENT_WORK")
    return work


def _report_build(record: dict) -> int:
    if record["status"] != "ok":
        print(f"FAILED: {record['error']}", file=sys.stderr)
        return 1
    out = record["output"]
    if record["kind"] == "fetch":
        print(f"fetched {len(out['files'])} files to {out['dir']}")
    elif record["kind"] == "imatrix":
        print(f"importance matrix: {out['file']}, sha256 {out['sha256'][:12]}")
    else:
        p = out["passport"]
        print(f"{record['recipe']}: {out['file']}\n  {out['file_bytes'] / 1e6:.1f} MB, {p['bits_per_param']} bits/param, "
              f"vocabulary tensors {out['vocab_tensor_types']}, sha256 {out['sha256'][:12]}")
    print(f"build {record['build_id']}")
    return 0


def cmd_model_fetch(args) -> int:
    from . import builds

    return _report_build(builds.fetch(args.repo, args.revision, _need_work(args), _results_dir(args)))


def cmd_model_convert(args) -> int:
    from . import builds

    work = _need_work(args)
    python = Path(args.python) if args.python else work / "venv-convert" / "Scripts" / "python.exe"
    llama_cpp = Path(args.llama_cpp) if args.llama_cpp else work / "llama.cpp"
    return _report_build(builds.convert(Path(args.source_dir), args.name, args.outtype, work, _results_dir(args),
                                        python=python, llama_cpp=llama_cpp))


def cmd_model_quantize(args) -> int:
    from . import builds
    from .quality.cli import DEFAULT_ENGINE

    work = _need_work(args)
    engine_dir = Path(args.engine_dir) if args.engine_dir else work / "host" / DEFAULT_ENGINE
    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(f"error: input file not found: {path}")
    code = 0
    for qtype in args.type.split(","):
        code |= _report_build(builds.quantize(path, qtype, _results_dir(args), engine_dir=engine_dir,
                                              embedding_type=args.embedding_type, threads=args.threads,
                                              imatrix_path=Path(args.imatrix) if args.imatrix else None,
                                              tensor_types=args.tensor_type))
    return code


def cmd_model_imatrix(args) -> int:
    from . import builds
    from .quality.cli import DEFAULT_ENGINE

    work = _need_work(args)
    engine_dir = Path(args.engine_dir) if args.engine_dir else work / "host" / DEFAULT_ENGINE
    return _report_build(builds.imatrix(Path(args.input), [Path(p) for p in args.calibration], _results_dir(args),
                                        engine_dir=engine_dir, chunks=args.chunks, gpu_layers=args.gpu_layers,
                                        threads=args.threads))


def cmd_model_show(args) -> int:
    from . import builds

    records = builds.load(_results_dir(args))
    if not records:
        print("no builds recorded yet")
        return 0
    rows = [["build", "recipe", "MB", "bits/param", "vocab tensors", "sha256", "status"]]
    for r in records:
        out = r["output"]
        if r["kind"] == "fetch":
            size = sum(f["bytes"] for f in out.get("files", {}).values()) / 1e6 if out else None
            rows.append([r["build_id"], r["source"], _fmt(size, 0), "-", "-", "-", r["status"]])
            continue
        if r["kind"] == "imatrix" or "passport" not in out:
            rows.append([r["build_id"], r["recipe"], "-", "-", "-", out.get("sha256", "-")[:12], r["status"]])
            continue
        rows.append([r["build_id"], r["recipe"], _fmt(out["file_bytes"] / 1e6, 1) if out else "-",
                     _fmt(out["passport"]["bits_per_param"], 2) if out else "-",
                     ",".join(f"{k.split('.')[0]}={v}" for k, v in out.get("vocab_tensor_types", {}).items()) or "-",
                     out.get("sha256", "-")[:12], r["status"]])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return 0


def register_model(groups, common) -> None:
    model = groups.add_parser("model", help="build our own model files: fetch, convert, quantize").add_subparsers(
        dest="command", required=True)

    fetch = model.add_parser("fetch", help="download a Hugging Face model at a fixed revision")
    common(fetch)
    fetch.add_argument("--repo", required=True, help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    fetch.add_argument("--revision", required=True, help="full commit hash of the repository")
    fetch.set_defaults(func=cmd_model_fetch)

    convert = model.add_parser("convert", help="convert a fetched model to GGUF")
    common(convert)
    convert.add_argument("--source-dir", required=True, help="folder made by `model fetch`")
    convert.add_argument("--name", required=True, help="model name written into the file, e.g. Qwen2.5-0.5B-Instruct")
    convert.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"])
    convert.add_argument("--python", help="interpreter with the converter's requirements "
                                          "(default: <work-dir>/venv-convert/Scripts/python.exe)")
    convert.add_argument("--llama-cpp", help="llama.cpp source tree of the engine version (default: <work-dir>/llama.cpp)")
    convert.set_defaults(func=cmd_model_convert)

    quant = model.add_parser("quantize", help="quantize a converted file with llama-quantize")
    common(quant)
    quant.add_argument("--input", required=True, help="GGUF made by `model convert`")
    quant.add_argument("--type", required=True, help="llama-quantize type(s), comma-separated, e.g. q8_0,q4_k_m")
    quant.add_argument("--embedding-type", help="type of the vocabulary table, e.g. q5_0 (default: llama.cpp's choice)")
    quant.add_argument("--threads", type=int, default=16)
    quant.add_argument("--engine-dir", help="folder with llama-quantize (default: <work-dir>/host/<engine>)")
    quant.add_argument("--imatrix", help="importance matrix made by `model imatrix` for the same input file")
    quant.add_argument("--tensor-type", action="append",
                       help="llama-quantize override PATTERN=TYPE, repeatable, e.g. ffn_down=q4_0")
    quant.set_defaults(func=cmd_model_quantize)

    imat = model.add_parser("imatrix", help="importance matrix of a converted model over calibration text")
    common(imat)
    imat.add_argument("--input", required=True, help="GGUF made by `model convert`")
    imat.add_argument("--calibration", required=True, nargs="+", help="text files; they must not overlap evaluation")
    imat.add_argument("--chunks", type=int, default=200, help="chunks of 512 tokens")
    imat.add_argument("--gpu-layers", help="e.g. 'all' with a CUDA build in --engine-dir")
    imat.add_argument("--threads", type=int, default=16)
    imat.add_argument("--engine-dir", help="folder with llama-imatrix (default: <work-dir>/host/<engine>)")
    imat.set_defaults(func=cmd_model_imatrix)

    show = model.add_parser("show", help="print recorded builds")
    common(show)
    show.set_defaults(func=cmd_model_show)


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
    run.add_argument("--flash-attn", choices=["on", "off", "auto"], help="flash attention (default: llama.cpp's auto)")
    run.add_argument("--ubatch", type=int, help="micro-batch size (default: llama.cpp's 512)")
    run.add_argument("--load-mode", choices=["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"],
                     help="model loading; 'none' avoids keeping both the mapped file and repacked copies (default: auto)")
    run.add_argument("--label", help="tag for a non-default setup, e.g. fa-off")
    run.set_defaults(func=cmd_run)

    show = bench.add_parser("show", help="print recorded runs as a table")
    common(show)
    show.set_defaults(func=cmd_show)

    rep = bench.add_parser("report", help="build charts and tables from the recorded runs")
    common(rep)
    rep.add_argument("--name", default="report", help="report name, used in the folder name")
    rep.add_argument("--out", help="output folder (default: <results-dir>/reports/<date>-<name>)")
    rep.add_argument("--base", help="part of the base variant's label (default: the largest model)")
    rep.set_defaults(func=cmd_report)

    register_model(groups, common)

    from .quality import cli as quality_cli  # imported here: bench commands must work without its dependencies

    quality_cli.register(groups, common)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
