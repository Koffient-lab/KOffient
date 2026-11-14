"""Parsers are checked against outputs saved from real runs (see fixtures/)."""

from pathlib import Path

import pytest

from koffient_tooling import bench, device, engine, model

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_llama_bench_splits_prompt_and_generation_tests():
    result = bench.parse_llama_bench(read("llama_bench.json"))
    assert [p["n_prompt"] for p in result["prefill"]] == [256]
    assert [d["n_gen"] for d in result["decode"]] == [64]
    # samples_ts in the fixture: 56.6136, 58.6012, 58.5242
    assert result["prefill"][0]["tps"]["median"] == pytest.approx(58.524, abs=1e-3)
    assert len(result["prefill"][0]["tps"]["samples"]) == 3
    assert result["build_commit"] == "372dc5c"


def test_llama_bench_rejects_non_json():
    with pytest.raises(bench.ProbeError):
        bench.parse_llama_bench("error: failed to load model")


def test_llama_bench_rejects_output_without_generation_test():
    only_prompt = '[{"n_prompt": 256, "n_gen": 0, "samples_ts": [50.0]}]'
    with pytest.raises(bench.ProbeError):
        bench.parse_llama_bench(only_prompt)


def test_completion_timings():
    result = bench.parse_completion(read("completion.err"))
    assert result["load_ms"] == 889.00
    assert result["prompt_eval_ms"] == 1170.06
    assert result["n_prompt_tokens"] == 52
    assert result["sampling_ms"] == 19.01
    assert result["cpu_features"] == ["SSE3", "SSSE3", "REPACK"]


def test_completion_without_timings_is_an_error():
    with pytest.raises(bench.ProbeError):
        bench.parse_completion("error: unable to load model\n")


def test_time_v_report():
    result = bench.parse_time_v(read("completion.err") + read("time_v.txt"))
    assert result["max_rss_kib"] > 0
    assert result["real_s"] > 0


def test_time_v_missing_report_is_an_error():
    with pytest.raises(bench.ProbeError):
        bench.parse_time_v(read("completion.err"))


def test_kib_to_mb_uses_decimal_megabytes():
    assert bench.kib_to_mb(546820) == 559.9


def test_cmake_cache_flags():
    cache = engine.parse_cmake_cache(read("CMakeCache.txt"))
    assert cache["ANDROID_ABI"] == "arm64-v8a"
    assert cache["GGML_CPU_ARM_ARCH"] == "armv8.2-a+dotprod+i8mm"
    assert cache["GGML_NATIVE"] == "OFF"
    assert cache["CMAKE_HOME_DIRECTORY"] == "C:/koffient-work/llama.cpp"


def test_device_facts_emulator():
    text = (
        "abi=x86_64\nos_release=15\nsdk=35\nmodel=Android SDK built for x86_64\nqemu=11\nsoc=\nncpu=4\n"
        "mem_kb=8137032\ncpu_name= AMD Ryzen 9 9950X 16-Core Processor\n"
        "cpu_flags= fpu sse4_2 avx f16c fma avx2 bmi2 avx512f\n"
    )
    facts = device.parse_facts(text, "emulator-5554")
    assert facts["kind"] == "emulator"
    assert facts["os"] == "Android 15"
    assert facts["ram_mb"] == 8332
    assert facts["cpu_features"] == ["sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx512f"]
    assert facts["speed_is_phone_grade"] is False


def test_device_facts_phone():
    text = "abi=arm64-v8a\nos_release=14\nsdk=34\nmodel=Pixel 8\nqemu=\nsoc=Tensor G3\nncpu=9\nmem_kb=7600000\n" \
           "cpu_name=\ncpu_flags= fp asimd asimddp i8mm sve\n"
    facts = device.parse_facts(text, "39111FDJH000AB")
    assert facts["kind"] == "phone"
    assert facts["cpu"] == "Tensor G3"
    assert facts["cpu_features"] == ["asimd", "asimddp", "i8mm", "sve"]
    assert facts["speed_is_phone_grade"] is True


def test_tensor_summary_reports_real_bits_and_duplicate_vocabulary():
    tensors = [
        ("token_embd.weight", "Q8_0", 1000, 800),
        ("output.weight", "Q8_0", 1000, 800),
        ("blk.0.attn_q.weight", "Q5_0", 500, 800),
    ]
    summary = model.summarize_tensors(tensors)
    assert summary["params"] == 2400
    assert summary["bits_per_param"] == pytest.approx(2500 * 8 / 2400, abs=1e-3)
    assert list(summary["tensor_types_mb"]) == ["Q8_0", "Q5_0"]
    assert summary["separate_output_head"] is True
