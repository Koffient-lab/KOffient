"""Our own model files: fetch the source from Hugging Face, convert it to GGUF, quantize it.

Every step appends one record to builds.jsonl. The records chain: a quantized file points to the converted file it
was made from, the converted file points to the fetched source. Measurement records find their file here by sha256.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import record as runs
from .model import describe_model, sha256_file

SCHEMA_VERSION = 1
BUILDS_FILE = "builds.jsonl"
KINDS = ("fetch", "convert", "quantize")
CONVERTER = "convert_hf_to_gguf.py"
QUANTIZER = "llama-quantize.exe"
# Packages that decide what the converter writes; their versions go into the build record.
CONVERTER_PACKAGES = ("torch", "transformers", "numpy", "sentencepiece", "safetensors", "tokenizers")

_REQUIRED = {
    "schema_version": int,
    "build_id": str,
    "started_at": str,
    "kind": str,
    "status": str,
    "tooling_commit": str,
    "recipe": str,
    "output": dict,
    "raw_dir": str,
}


class BuildError(RuntimeError):
    pass


def recipe_name(qtype: str, embedding_type: str | None = None) -> str:
    """Short name of a quantization recipe: the llama-quantize type plus overrides, e.g. `q4_k_m-emb-q5_0`."""
    name = qtype.lower()
    return f"{name}-emb-{embedding_type.lower()}" if embedding_type else name


def new_build_id(kind: str, model_slug: str, recipe: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return "-".join([now.strftime("%Y%m%dT%H%M%SZ"), kind, runs.slug(model_slug), runs.slug(recipe)])


def validate(record: dict) -> None:
    for key, expected in _REQUIRED.items():
        if key not in record:
            raise ValueError(f"record is missing '{key}'")
        if not isinstance(record[key], expected):
            raise ValueError(f"record field '{key}' must be {expected.__name__}")
    if record["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if record["status"] == "ok":
        needed = ("dir", "files") if record["kind"] == "fetch" else ("file", "sha256", "passport")
        missing = [key for key in needed if key not in record["output"]]
        if missing:
            raise ValueError(f"status is ok but the output block is missing {missing}")
        if record["kind"] != "fetch" and not record.get("input"):
            raise ValueError(f"a '{record['kind']}' record needs an 'input' block")
    elif record["status"] == "failed":
        if not record.get("error"):
            raise ValueError("a failed record needs an 'error'")
    else:
        raise ValueError("status must be 'ok' or 'failed'")


def append(record: dict, results_dir: Path) -> Path:
    validate(record)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / BUILDS_FILE
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load(results_dir: Path) -> list[dict]:
    path = results_dir / BUILDS_FILE
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_by_sha(records: list[dict], sha256: str) -> dict | None:
    """The latest successful build that produced a file with this checksum."""
    found = None
    for record in records:
        if record["status"] == "ok" and record["output"].get("sha256") == sha256:
            found = record
    return found


def find_fetch(records: list[dict], source_dir: Path) -> dict | None:
    found = None
    for record in records:
        if record["kind"] == "fetch" and record["status"] == "ok" \
                and Path(record["output"]["dir"]).resolve() == source_dir.resolve():
            found = record
    return found


def build_ref(records: list[dict], sha256: str) -> dict | None:
    """What a measurement record stores about a file we built ourselves; None for files we did not build."""
    build = find_by_sha(records, sha256)
    if build is None:
        return None
    return {"build_id": build["build_id"], "recipe": build["recipe"], "source": build["source"]}


def attach(model: dict, results_dir: Path) -> dict:
    """Add the build reference to a model passport of a measurement record; fill `source` from it when empty."""
    ref = build_ref(load(results_dir), model["sha256"])
    model["build"] = ref
    if ref and not model.get("source"):
        model["source"] = ref["source"]
    return model


def _base_record(kind: str, build_id: str, started: datetime, recipe: str, source: str | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "status": "failed",
        "error": None,
        "tooling_commit": runs.tooling_commit(),
        "recipe": recipe,
        "source": source,
        "input": None,
        "tool": None,
        "args": None,
        "output": {},
        "raw_dir": f"raw/builds/{build_id}",
    }


def _file_passport(path: Path) -> dict:
    """Output block for a GGUF file: checksum and contents, plus the type the vocabulary table got."""
    import gguf

    passport = describe_model(path)
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(str(path)).tensors
             if t.name in ("token_embd.weight", "output.weight")}
    return {"file": str(path), "sha256": passport.pop("sha256"), "file_bytes": passport["file_bytes"],
            "vocab_tensor_types": types, "passport": passport}


def fetch(repo: str, revision: str, work: Path, results: Path) -> dict:
    """Download every file of a Hugging Face model repository at a fixed revision and record the checksums."""
    from huggingface_hub import HfApi, snapshot_download

    started = datetime.now(timezone.utc)
    source = f"hf:{repo}@{revision}"
    model_slug = repo.split("/")[-1].lower()
    record = _base_record("fetch", new_build_id("fetch", model_slug, revision[:7], started), started, "source",
                          source)
    record["source_repo"] = {"repo": repo, "revision": revision}
    try:
        if len(revision) != 40:
            raise BuildError("pass the full 40-character commit hash as --revision, not a branch name")
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
        card = info.card_data.to_dict() if info.card_data else {}
        record["source_repo"]["license"] = card.get("license")
        out_dir = work / "sources" / model_slug / revision[:7]
        snapshot_download(repo, revision=revision, local_dir=str(out_dir))
        files = {}
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and ".cache" not in p.parts):
            name = path.relative_to(out_dir).as_posix()
            digest = sha256_file(path)
            if name in expected and expected[name] != digest:
                raise BuildError(f"{name}: sha256 {digest} does not match the repository's {expected[name]}")
            files[name] = {"sha256": digest, "bytes": path.stat().st_size}
        record["output"] = {"dir": str(out_dir), "files": files}
        record["status"] = "ok"
    except (BuildError, OSError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    append(record, results)
    return record


def _converter_tool(python: Path, llama_cpp: Path) -> dict:
    def git(*args: str) -> str | None:
        proc = subprocess.run(["git", "-C", str(llama_cpp), *args], capture_output=True, text=True, timeout=60)
        return proc.stdout.strip() if proc.returncode == 0 else None

    probe = ("import importlib.metadata as m, json; "
             f"print(json.dumps({{p: m.version(p) for p in {list(CONVERTER_PACKAGES)!r}}}))")
    proc = subprocess.run([str(python), "-I", "-c", probe], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise BuildError(f"cannot read package versions of {python}: {proc.stderr.strip()[:300]}")
    return {
        "name": "llama.cpp " + CONVERTER,
        "tag": git("describe", "--tags", "--exact-match"),
        "commit": (git("rev-parse", "--short", "HEAD") or "unknown"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(llama_cpp / CONVERTER),
        "python": str(python),
        "packages": json.loads(proc.stdout),
    }


def convert(source_dir: Path, model_name: str, outtype: str, work: Path, results: Path, *,
            python: Path, llama_cpp: Path) -> dict:
    """Convert a fetched Hugging Face model to GGUF with the converter script from the engine's own source tree."""
    started = datetime.now(timezone.utc)
    fetched = find_fetch(load(results), source_dir)
    source = fetched["source"] if fetched else None
    model_slug = runs.slug(model_name)
    record = _base_record("convert", new_build_id("convert", model_slug, outtype, started), started, outtype, source)
    raw = results / record["raw_dir"]
    try:
        if fetched is None:
            raise BuildError(f"{source_dir} was not fetched with `model fetch`: its origin is unknown")
        weights = {name: entry for name, entry in fetched["output"]["files"].items()
                   if name.endswith((".safetensors", ".bin", ".json", ".txt", ".model"))}
        record["input"] = {"dir": str(source_dir), "fetch_build_id": fetched["build_id"], "files": weights}
        record["tool"] = _converter_tool(python, llama_cpp)
        out = work / "models" / model_slug / fetched["source_repo"]["revision"][:7] / f"{model_slug}-{outtype}.gguf"
        if out.exists():
            raise BuildError(f"{out} already exists; remove it first")
        out.parent.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        # The converter imports modules from its own folder, so it cannot run in isolated mode (-I). It reads the
        # downloaded folder as data only; -E -s keep environment variables and user site-packages out.
        args = ["-E", "-s", CONVERTER, str(source_dir), "--outtype", outtype, "--outfile", str(out),
                "--model-name", model_name]
        record["args"] = args
        with open(raw / "convert.log", "w", encoding="utf-8") as log:
            proc = subprocess.run([str(python), *args], cwd=llama_cpp, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=3600)
        if proc.returncode != 0 or not out.is_file():
            raise BuildError(f"converter exited with {proc.returncode}, see {raw / 'convert.log'}")
        record["output"] = _file_passport(out)
        record["status"] = "ok"
    except (BuildError, OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    append(record, results)
    return record


def quantize(input_path: Path, qtype: str, results: Path, *, engine_dir: Path, embedding_type: str | None = None,
             threads: int = 16) -> dict:
    """Quantize a file from `model convert` with llama-quantize from the official build of the same engine version."""
    from .quality.server import ServerError, describe_host_engine

    started = datetime.now(timezone.utc)
    recipe = recipe_name(qtype, embedding_type)
    records = load(results)
    input_sha = sha256_file(input_path)
    parent = find_by_sha(records, input_sha)
    model_slug = input_path.stem.rsplit("-", 1)[0]
    record = _base_record("quantize", new_build_id("quantize", model_slug, recipe, started), started, recipe,
                          parent["source"] if parent else None)
    raw = results / record["raw_dir"]
    try:
        if parent is None or parent["kind"] != "convert":
            raise BuildError(f"{input_path} was not made by `model convert`: quantize only our own converted files")
        record["input"] = {"file": str(input_path), "sha256": input_sha, "build_id": parent["build_id"]}
        quantizer = engine_dir / QUANTIZER
        if not quantizer.is_file():
            raise BuildError(f"{QUANTIZER} not found in {engine_dir}")
        tool = describe_host_engine(engine_dir)
        tool["binaries_sha256"][QUANTIZER] = sha256_file(quantizer)
        record["tool"] = tool
        out = input_path.with_name(f"{model_slug}-{recipe}.gguf")
        if out.exists():
            raise BuildError(f"{out} already exists; remove it first")
        overrides = []
        if embedding_type:
            # With a shared vocabulary table llama.cpp gives token_embd the output type; set both to be explicit.
            overrides = ["--output-tensor-type", embedding_type, "--token-embedding-type", embedding_type]
        args = [*overrides, str(input_path), str(out), qtype.upper(), str(threads)]
        record["args"] = args
        raw.mkdir(parents=True, exist_ok=True)
        with open(raw / "llama-quantize.log", "w", encoding="utf-8") as log:
            proc = subprocess.run([str(quantizer), *args], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if proc.returncode != 0 or not out.is_file():
            raise BuildError(f"llama-quantize exited with {proc.returncode}, see {raw / 'llama-quantize.log'}")
        record["output"] = _file_passport(out)
        record["status"] = "ok"
    except (BuildError, ServerError, OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    append(record, results)
    return record
