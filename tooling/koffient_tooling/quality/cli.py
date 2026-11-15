"""`quality run | closeness | show` commands."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import record as runs
from ..model import describe_model, sha256_file
from . import closeness, record as rec
from .run import changed_answers, load_answers, run_task
from .scorers import ScoringError, make_scorer
from .server import ServerError, ServerPool, describe_host_engine
from .tasks import TASKS, Dirs, subset_sha256

DEFAULT_ENGINE = "llama-b12199-cpu-x64"
CORPORA = {
    "en": ("en.txt", "ggml-org/ci wikitext-2-raw-v1, wiki.test.raw"),
    "ru": ("ru.txt", "wikimedia/wikipedia 20231101.ru, first 40 articles, 6000 characters of each"),
}


def _work(args) -> Path:
    value = args.work_dir or os.environ.get("KOFFIENT_WORK")
    if not value:
        raise SystemExit("error: pass --work-dir or set KOFFIENT_WORK")
    return Path(value)


def _results(args) -> Path:
    return Path(args.results_dir) if args.results_dir else _work(args) / "results"


def _engine_dir(args) -> Path:
    return Path(args.engine_dir) if args.engine_dir else _work(args) / "host" / DEFAULT_ENGINE


def _model(args) -> tuple[Path, dict]:
    path = Path(args.model)
    if not path.is_file():
        raise SystemExit(f"error: model file not found: {path}")
    print(f"model passport: {path.name}", flush=True)
    return path, describe_model(path, source=args.source)


def _base_record(kind: str, eval_id: str, started: datetime, model: dict, engine: dict, params: dict,
                 label: str | None = None) -> dict:
    return {
        "label": label,
        "schema_version": rec.SCHEMA_VERSION,
        "eval_id": eval_id,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "status": "failed",
        "error": None,
        "tooling_commit": runs.tooling_commit(),
        "model": model,
        "engine": engine,
        "params": params,
        "metrics": {},
        "raw_dir": f"raw/quality/{eval_id}",
    }


def _eval_id(started: datetime, model: dict, suffix: str) -> str:
    parts = [model["name"], model.get("quant_label") or "", suffix]
    return "-".join([started.strftime("%Y%m%dT%H%M%SZ")] + [runs.slug(p) for p in parts if p])


def cmd_run(args) -> int:
    work, results, engine_dir = _work(args), _results(args), _engine_dir(args)
    names = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise SystemExit(f"error: unknown tasks {unknown}; known: {', '.join(TASKS)}")
    dirs = Dirs(work / "datasets", work / "third_party")
    try:
        engine = describe_host_engine(engine_dir)
    except ServerError as exc:
        raise SystemExit(f"error: {exc}")
    model_path, model = _model(args)
    params = {"temperature": 0, "seed": 1, "ctx": args.ctx, "servers": args.servers,
              "threads_per_server": args.threads, "limit": args.limit}

    session = datetime.now(timezone.utc)
    log_dir = results / "raw" / "quality" / _eval_id(session, model, "servers")
    failed = 0
    try:
        server = ServerPool(engine_dir, model_path, log_dir, servers=args.servers, ctx=args.ctx,
                            threads=args.threads).__enter__()
    except ServerError as exc:
        raise SystemExit(f"error: {exc}")
    try:
        for name in names:
            task = TASKS[name]
            started = datetime.now(timezone.utc)
            record = _base_record("task", _eval_id(started, model, name), started, model, engine,
                                  {**params, "max_tokens": task.max_tokens, "grammar": task.grammar}, args.label)
            record["task"] = {"name": task.name, "scenario": task.scenario, "role": task.role, "lang": task.lang,
                              "dataset": task.dataset, "revision": task.revision, "license": task.license}
            try:
                examples = task.load(dirs)
                if args.limit:
                    examples = examples[:args.limit]
                record["task"]["n"] = len(examples)
                record["task"]["subset_sha256"] = subset_sha256(examples)
                scorer = make_scorer(task, dirs, work / "nltk_data")
                record["metrics"] = run_task(server, task, examples, scorer, results / record["raw_dir"])
                if record["metrics"]["n_request_errors"] == len(examples):
                    raise ServerError("every request failed, see the server log")
                record["status"] = "ok"
            except (ScoringError, ServerError, OSError, KeyError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            rec.append(record, results)
            m = record["metrics"]
            line = (f"{m['value']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] n={m['n']} "
                    f"truncated={m['n_truncated']} errors={m['n_request_errors']} {m['wall_s']:.0f}s"
                    if record["status"] == "ok" else f"FAILED: {record['error']}")
            print(f"{name:12} {line}", flush=True)
    finally:
        server.__exit__(None, None, None)
    return 1 if failed else 0


def cmd_closeness(args) -> int:
    work, results, engine_dir = _work(args), _results(args), _engine_dir(args)
    base_path = Path(args.base)
    if not base_path.is_file():
        raise SystemExit(f"error: base model not found: {base_path}")
    try:
        engine = describe_host_engine(engine_dir)
    except ServerError as exc:
        raise SystemExit(f"error: {exc}")
    model_path, model = _model(args)
    base_sha = sha256_file(base_path)
    is_base = base_sha == model["sha256"]
    failed = 0
    for lang in args.langs.split(","):
        if lang not in CORPORA:
            raise SystemExit(f"error: unknown corpus language '{lang}'")
        file_name, source = CORPORA[lang]
        corpus = work / "datasets" / "corpus" / file_name
        logits = results / "kld" / f"{base_sha[:12]}-{lang}-c{closeness.N_CTX}x{args.chunks}.kld"
        started = datetime.now(timezone.utc)
        record = _base_record("closeness", _eval_id(started, model, f"closeness-{lang}"), started, model, engine,
                              {"n_ctx": closeness.N_CTX, "chunks": args.chunks, "threads": args.threads},
                              args.label)
        record["corpus"] = {"lang": lang, "file": file_name, "source": source}
        record["base"] = {"file": base_path.name, "sha256": base_sha}
        log = results / record["raw_dir"] / "llama-perplexity.log"
        try:
            record["corpus"]["sha256"] = sha256_file(corpus)
            if is_base or not logits.is_file():
                print(f"{lang}: saving base logits to {logits}", flush=True)
                base_metrics = closeness.save_base_logits(engine_dir, base_path, corpus, logits,
                                                          log.with_name("base.log"), chunks=args.chunks,
                                                          threads=args.threads)
            if is_base:
                record["metrics"] = base_metrics
            else:
                record["metrics"] = closeness.compare_with_base(engine_dir, model_path, logits, log,
                                                                chunks=args.chunks, threads=args.threads)
            record["status"] = "ok"
        except (closeness.ClosenessError, OSError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        rec.append(record, results)
        m = record["metrics"]
        if record["status"] != "ok":
            print(f"{lang}: FAILED: {record['error']}")
        elif is_base:
            print(f"{lang}: base perplexity {m['ppl']:.3f}")
        else:
            print(f"{lang}: PPL {m['ppl']:.3f} (base {m['ppl_base']:.3f}), KLD {m['kld_mean']:.4f}, "
                  f"same top token {m['same_top_pct']:.1f}%")
    return 1 if failed else 0


def _label(variant: dict) -> str:
    model = variant["model"]
    suffix = f" {variant['label']}" if variant["label"] else ""
    return f"{model.get('quant_label') or model['file']}{suffix} ({model['file_bytes'] / 1e6:.0f} MB)"


def comparison(records: list[dict], results: Path, base_hint: str | None) -> dict:
    """Variants, tasks and per-cell numbers for `show` and for the report charts."""
    tasks = rec.latest_ok(records, "task")
    close = rec.latest_ok(records, "closeness")
    variants = {}
    for record in list(tasks.values()) + list(close.values()):
        variants[rec.variant_key(record)] = {"key": rec.variant_key(record), "model": record["model"],
                                             "label": record.get("label")}
    if not variants:
        return {"variants": [], "base": None, "tasks": {}, "closeness": {}}
    order = sorted(variants.values(), key=lambda v: (-v["model"]["file_bytes"], v["label"] or ""))
    hinted = [v for v in order if base_hint and base_hint.lower() in _label(v).lower()]
    base = (hinted or order)[0]

    cells: dict[str, dict[str, dict]] = {}
    for (key, name), record in tasks.items():
        cell = {"value": record["metrics"]["value"], "ci95": record["metrics"]["ci95"], "n": record["metrics"]["n"],
                "truncated": record["metrics"]["n_truncated"], "role": record["task"]["role"],
                "scenario": record["task"]["scenario"], "lang": record["task"]["lang"], "changed": None}
        base_record = tasks.get((base["key"], name))
        if base_record and key != base["key"]:
            try:
                cell["changed"] = changed_answers(load_answers(results / base_record["raw_dir"]),
                                                  load_answers(results / record["raw_dir"]))
            except OSError:
                pass
        cells.setdefault(name, {})[key] = cell
    closeness_cells: dict[str, dict[str, dict]] = {}
    for (key, lang), record in close.items():
        closeness_cells.setdefault(lang, {})[key] = record["metrics"]
    return {"variants": order, "base": base, "tasks": cells, "closeness": closeness_cells}


def cmd_show(args) -> int:
    results = _results(args)
    data = comparison(rec.load(results), results, args.base)
    if not data["variants"]:
        print("no quality records yet")
        return 0
    models, base = data["variants"], data["base"]
    print(f"base for comparison: {_label(base)}")
    header = ["task", "role"] + [_label(m) for m in models]
    rows = [header]
    for name in TASKS:
        if name not in data["tasks"]:
            continue
        cells = data["tasks"][name]
        row = [name, next(iter(cells.values()))["role"]]
        for model in models:
            cell = cells.get(model["key"])
            if not cell:
                row.append("-")
                continue
            text = f"{cell['value']:.3f} [{cell['ci95'][0]:.2f}-{cell['ci95'][1]:.2f}]"
            if cell["changed"]:
                c = cell["changed"]
                text += f" changed {c['share'] * 100:.0f}% (-{c['worse']} +{c['better']})"
            row.append(text)
        rows.append(row)
    for lang, cells in sorted(data["closeness"].items()):
        row = [f"closeness-{lang}", "fp16"]
        for model in models:
            m = cells.get(model["key"])
            if not m:
                row.append("-")
            elif "kld_mean" in m:
                row.append(f"PPL {m['ppl']:.2f} KLD {m['kld_mean']:.4f} top {m['same_top_pct']:.1f}%")
            else:
                row.append(f"PPL {m['ppl']:.2f}")
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return 0


def register(groups, common) -> None:
    quality = groups.add_parser("quality", help="quality evaluation on the host").add_subparsers(
        dest="command", required=True)

    def host(sub):
        common(sub)
        sub.add_argument("--model", required=True, help="path to a GGUF file")
        sub.add_argument("--source", help="where the model file came from, e.g. hf:org/repo@revision")
        sub.add_argument("--label", help="tag for a non-default setup of the same model, e.g. threads4")
        sub.add_argument("--engine-dir", help=f"folder with llama.cpp host binaries "
                                              f"(default: <work-dir>/host/{DEFAULT_ENGINE})")

    run = quality.add_parser("run", help="run the fast task set for one model")
    host(run)
    run.add_argument("--tasks", default="all", help=f"comma-separated task names (default: all): {', '.join(TASKS)}")
    run.add_argument("--limit", type=int, help="use only the first N examples of every task (for a trial run)")
    run.add_argument("--servers", type=int, default=8, help="independent single-slot servers to run in parallel")
    run.add_argument("--threads", type=int, default=2, help="threads per server")
    run.add_argument("--ctx", type=int, default=4096, help="context size")
    run.set_defaults(func=cmd_run)

    close = quality.add_parser("closeness", help="perplexity and KL divergence against the fp16 base")
    host(close)
    close.add_argument("--base", required=True, help="path to the fp16 GGUF of the same model")
    close.add_argument("--langs", default="en,ru")
    close.add_argument("--chunks", type=int, default=closeness.DEFAULT_CHUNKS)
    close.add_argument("--threads", type=int, default=16)
    close.set_defaults(func=cmd_closeness)

    show = quality.add_parser("show", help="compare recorded models task by task")
    common(show)
    show.add_argument("--base", help="part of the base variant's label (default: the largest model)")
    show.set_defaults(func=cmd_show)
