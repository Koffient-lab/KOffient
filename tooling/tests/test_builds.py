from datetime import datetime, timezone

import pytest

from koffient_tooling import builds
from koffient_tooling.model import variant_name


def make_build(**overrides) -> dict:
    record = {
        "schema_version": builds.SCHEMA_VERSION,
        "build_id": "20261117T000000Z-quantize-m-q4_k_m",
        "started_at": "2026-11-17T00:00:00Z",
        "kind": "quantize",
        "status": "ok",
        "error": None,
        "tooling_commit": "abc1234",
        "recipe": "q4_k_m",
        "source": "hf:org/m@" + "1" * 40,
        "input": {"file": "m-f16.gguf", "sha256": "a" * 64, "build_id": "convert-1"},
        "tool": {"name": "llama.cpp"},
        "args": [],
        "output": {"file": "m-q4_k_m.gguf", "sha256": "b" * 64, "file_bytes": 10, "passport": {}},
        "raw_dir": "raw/builds/x",
    }
    record.update(overrides)
    return record


def test_recipe_name_marks_the_vocabulary_override():
    assert builds.recipe_name("Q4_K_M") == "q4_k_m"
    assert builds.recipe_name("q4_k_m", "Q5_0") == "q4_k_m-emb-q5_0"
    assert builds.recipe_name("IQ4_XS", imatrix=True) == "iq4_xs-imat"
    assert builds.recipe_name("q4_0", imatrix=True, tensor_types=["ffn_down=Q4_0"]) == "q4_0-imat-ffn_down-q4_0"
    assert builds.recipe_name("q3_k_m", tensor_types=[r"blk\.[0-4]\.ffn_down=q4_k"]) == "q3_k_m-blk-.-0-4-.ffn_down-q4_k"
    builds.validate(make_build(kind="imatrix", recipe="imatrix", output={"file": "m-imatrix.gguf", "sha256": "e" * 64}))


def test_build_id_is_sortable_and_filesystem_safe():
    now = datetime(2026, 11, 17, 3, 4, 5, tzinfo=timezone.utc)
    assert builds.new_build_id("quantize", "Qwen2.5 0.5B", "q4_k_m-emb-q5_0", now) == \
        "20261117T030405Z-quantize-qwen2.5-0.5b-q4_k_m-emb-q5_0"


def test_valid_build_record_passes():
    builds.validate(make_build())
    builds.validate(make_build(kind="fetch", input=None, output={"dir": "d", "files": {}}))


@pytest.mark.parametrize("change, message", [
    ({"kind": "train"}, "kind must be"),
    ({"output": {"file": "x"}}, "output block is missing"),
    ({"input": None}, "needs an 'input' block"),
    ({"status": "failed"}, "needs an 'error'"),
])
def test_invalid_build_records_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        builds.validate(make_build(**change))


def test_append_and_find_by_checksum(tmp_path):
    first = make_build()
    failed = make_build(build_id="later-failed", status="failed", error="boom", output={})
    builds.append(first, tmp_path)
    builds.append(failed, tmp_path)
    records = builds.load(tmp_path)
    assert builds.find_by_sha(records, "b" * 64)["build_id"] == first["build_id"]
    assert builds.find_by_sha(records, "c" * 64) is None


def test_attach_links_a_measured_file_to_its_build(tmp_path):
    builds.append(make_build(), tmp_path)
    ours = builds.attach({"sha256": "b" * 64, "source": None, "quant_label": "q4_k_m", "file": "m.gguf"}, tmp_path)
    assert ours["build"]["recipe"] == "q4_k_m"
    assert ours["source"].startswith("hf:org/m@")
    assert variant_name(ours) == "built q4_k_m"
    official = builds.attach({"sha256": "d" * 64, "source": "hf:x", "quant_label": "q4_k_m", "file": "o.gguf"},
                             tmp_path)
    assert official["build"] is None and official["source"] == "hf:x"
    assert variant_name(official) == "q4_k_m"


def test_quantize_refuses_a_file_we_did_not_convert(tmp_path):
    stranger = tmp_path / "m-f16.gguf"
    stranger.write_bytes(b"not ours")
    record = builds.quantize(stranger, "q4_k_m", tmp_path, engine_dir=tmp_path)
    assert record["status"] == "failed"
    assert "was not made by `model convert`" in record["error"]
    assert builds.load(tmp_path)[-1]["build_id"] == record["build_id"]
