import json
from datetime import datetime, timezone

import pytest

from koffient_tooling import record as rec


def make_record(**overrides) -> dict:
    stat = {"median": 1.0, "min": 1.0, "max": 1.0, "samples": [1.0]}
    record = {
        "schema_version": rec.SCHEMA_VERSION,
        "run_id": "20261115T000000Z-m-q4-b",
        "started_at": "2026-11-15T00:00:00Z",
        "layer": "engine",
        "status": "ok",
        "error": None,
        "tooling_commit": "abc1234",
        "model": {"name": "m", "sha256": "0" * 64, "file_bytes": 10, "bits_per_param": 4.5,
                  "tensor_types_mb": {"Q4_K": 1.0}},
        "engine": {"build_id": "b"},
        "device": {"kind": "emulator"},
        "params": {"threads": 4},
        "metrics": {
            "prefill": [{"n_prompt": 256, "tps": stat}],
            "decode": [{"n_gen": 128, "tps": stat}],
            "ttft_ms": {"warm": stat, "cold": None},
            "load_ms": {"warm": stat, "cold": None},
            "peak_rss_mb": 500.0,
        },
        "raw_dir": "raw/x",
    }
    record.update(overrides)
    return record


def test_run_id_is_sortable_and_filesystem_safe():
    now = datetime(2026, 11, 15, 12, 30, 5, tzinfo=timezone.utc)
    run_id = rec.new_run_id({"name": "Qwen2.5 0.5B Instruct", "quant_label": "q4_k_m"},
                            {"build_id": "android-x86_64-avx2"}, now)
    assert run_id == "20261115T123005Z-qwen2.5-0.5b-instruct-q4_k_m-android-x86_64-avx2"


def test_append_and_load_round_trip(tmp_path):
    rec.append(make_record(), tmp_path)
    rec.append(make_record(run_id="second"), tmp_path)
    loaded = rec.load(tmp_path)
    assert [r["run_id"] for r in loaded] == ["20261115T000000Z-m-q4-b", "second"]
    lines = (tmp_path / rec.RUNS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(line) for line in lines)


def test_ok_record_must_have_all_metrics():
    record = make_record()
    del record["metrics"]["peak_rss_mb"]
    with pytest.raises(ValueError, match="peak_rss_mb"):
        rec.validate(record)


def test_failed_record_keeps_partial_metrics_but_needs_a_reason(tmp_path):
    with pytest.raises(ValueError, match="error"):
        rec.validate(make_record(status="failed", metrics={}))
    rec.append(make_record(status="failed", error="llama-bench exited with code 137", metrics={}), tmp_path)
    assert rec.load(tmp_path)[0]["status"] == "failed"


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="layer"):
        rec.validate(make_record(layer="gpu"))


def test_load_of_missing_file_is_empty(tmp_path):
    assert rec.load(tmp_path) == []
