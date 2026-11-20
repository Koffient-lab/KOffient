import json

from koffient_tooling import builds, package


def build(build_id, kind, input_=None, output=None, **extra):
    record = {"schema_version": builds.SCHEMA_VERSION, "build_id": build_id, "started_at": "2026-11-19T00:00:00Z",
              "kind": kind, "status": "ok", "error": None, "tooling_commit": "abc", "recipe": extra.pop("recipe", kind),
              "source": "hf:org/m@" + "1" * 40, "input": input_, "tool": {}, "args": [], "output": output or {},
              "raw_dir": "raw/builds/x"}
    record.update(extra)
    return record


def test_fetch_record_is_found_through_the_build_chain():
    fetch = build("f", "fetch", output={"dir": "d", "files": {}}, source_repo={"repo": "org/m", "license": "apache-2.0"})
    convert = build("c", "convert", input_={"fetch_build_id": "f"}, output={"file": "m-f16.gguf", "sha256": "a" * 64,
                                                                        "passport": {}})
    quant = build("q", "quantize", input_={"build_id": "c"}, output={"file": "m-q4.gguf", "sha256": "b" * 64,
                                                                     "passport": {}})
    assert package._fetch_record([fetch, convert, quant], quant)["build_id"] == "f"
    assert package._fetch_record([convert, quant], quant) is None


def test_package_writes_manifest_and_licence(tmp_path):
    results, models, src = tmp_path / "results", tmp_path / "models", tmp_path / "src"
    for d in (results, models, src):
        d.mkdir()
    model_file = models / "m-q4_k_m.gguf"
    model_file.write_bytes(b"gguf bytes")
    from koffient_tooling.model import sha256_file

    sha = sha256_file(model_file)
    builds.append(build("f", "fetch", output={"dir": str(src), "files": {}},
                        source_repo={"repo": "org/m", "license": "apache-2.0"}), results)
    builds.append(build("q", "quantize", recipe="q4_k_m", input_={"build_id": "f"},
                        output={"file": str(model_file), "sha256": sha, "file_bytes": 10,
                                "passport": {"name": "M", "params": 1, "bits_per_param": 4.5}}), results)
    run = {"run_id": "r", "status": "ok", "label": None, "model": {"sha256": sha}, "device": {"kind": "emulator"},
           "engine": {"build_id": "x86"}, "params": {"load_mode": "none", "n_ctx": 2048},
           "metrics": {"peak_rss_mb": 1258.0}}
    bad = {**run, "run_id": "bad", "metrics": {"peak_rss_mb": 9999.0}}
    (results / "runs.jsonl").write_text(json.dumps(run) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    (results / "invalid.jsonl").write_text(json.dumps({"run_id": "bad", "reason": "overlapped"}) + "\n",
                                           encoding="utf-8")
    out = package.package(model_file, "primary", tmp_path / "release", results, notes=["note"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file"]["sha256"] == sha and manifest["role"] == "primary" and manifest["notes"] == ["note"]
    assert manifest["runtime"]["min_ram_mb_measured"] == 1258.0
    assert manifest["runtime"]["min_ram_mb_measured_at_n_ctx"] == 2048
    assert manifest["runtime"]["load_mode"] == "none" and manifest["runtime"]["sampling"]["repeat_penalty"] == 1.0
    assert (out / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")
    assert (out / model_file.name).read_bytes() == b"gguf bytes"
