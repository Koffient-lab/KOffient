"""Quality evaluation: answer parsing, scoring helpers, subsampling and records. No model or server needed."""

from pathlib import Path

import pytest

from koffient_tooling.quality import closeness, record as rec, run, scorers, tasks
from koffient_tooling.quality.tasks import Example

FIXTURES = Path(__file__).parent / "fixtures"


def example(reference, **kwargs) -> Example:
    return Example(id="x", pair_key="x", messages=[], reference=reference, **kwargs)


# --- answer parsing ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("16 - 3 - 4 = 9 eggs, 9 * 2 = 18", 18),
    ("So the total is \\(\\boxed{3}\\).", 3),
    ("Step 1: 5 + 5 = 10\nAnswer: $35,000", 35000),
    ("Итого 1 250 рублей.", 1250),
    ("Ответ: 2,5", 2.5),
    ("Answer: 4\n\nBut 2 + 1 = 3 bolts in total.", 3),  # the mark is not on the last line: take the last number
    ("no numbers here", None),
    ("", None),
])
def test_final_number(text, expected):
    assert scorers.extract_final_number(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("B", "B"),
    ("The correct answer is D. One hundred", "D"),
    ("В", "B"),  # a lone Cyrillic look-alike
    ("В данном случае верно первое", None),  # a Russian sentence that merely starts with "В"
    ("Answer", None),
])
def test_choice_letter(text, expected):
    assert scorers.extract_choice(text) == expected


def test_json_is_found_inside_code_fence_and_after_text():
    assert scorers.extract_json('Here it is:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert scorers.extract_json('{"a": [1, 2]} trailing words') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        scorers.extract_json("I cannot do that")


def test_json_schema_scoring():
    schema = '{"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}'
    assert scorers.score_json_schema(example(schema), {"content": '{"n": 3}'})["score"] == 1.0
    wrong = scorers.score_json_schema(example(schema), {"content": '{"n": "three"}'})
    assert wrong["score"] == 0.0 and wrong["detail"]["error"] == "schema_violation"
    assert scorers.score_json_schema(example(schema), {"content": "n is 3"})["detail"]["error"] == "not_json"


def test_rouge_tokenizer_keeps_cyrillic():
    assert scorers.WordTokenizer().tokenize("Привет, мир! Hello 42") == ["привет", "мир", "hello", "42"]
    rouge = scorers.make_rouge()
    same = scorers.score_rouge(rouge, example("Ханне нужен номер Бетти"), {"content": "Ханне нужен номер Бетти"})
    assert same["score"] == 1.0


def test_irrelevance_expects_no_call():
    reference = {"category": "irrelevance", "functions": [], "ground_truth": None}
    assert scorers.score_bfcl(None, example(reference), {"tool_calls": []})["score"] == 1.0
    called = {"tool_calls": [{"name": "f", "arguments": {}}]}
    assert scorers.score_bfcl(None, example(reference), called)["score"] == 0.0


def test_missing_or_malformed_call_is_wrong_without_calling_the_checker():
    reference = {"category": "simple", "functions": [{}], "ground_truth": [{}]}
    assert scorers.score_bfcl(None, example(reference), {"tool_calls": []})["detail"]["error"] == "no_call"
    bad = {"tool_calls": [{"name": "f", "arguments": "{not json"}]}
    assert scorers.score_bfcl(None, example(reference), bad)["detail"]["error"] == "arguments_not_json_object"


# --- request building and subsampling ---------------------------------------------------------------------------

def test_bfcl_function_doc_becomes_a_tool():
    tool = tasks.bfcl_tool({
        "name": "math.area",
        "description": "Area",
        "parameters": {"type": "dict", "required": ["base"], "properties": {
            "base": {"type": "float", "description": "Base."},
            "sides": {"type": "tuple", "items": {"type": "integer"}},
            "options": {"type": "dict", "properties": {"unit": {"type": "any"}}},
        }},
    })["function"]
    assert tool["name"] == "math_area"
    properties = tool["parameters"]["properties"]
    assert tool["parameters"]["type"] == "object"
    assert properties["base"] == {"type": "number", "format": "float", "description": "Base. This is a float type value."}
    assert properties["sides"] == {"type": "array", "items": {"type": "integer"}}
    assert properties["options"]["properties"]["unit"]["type"] == "string"


def test_translation_notes_are_not_sent_to_the_model():
    row = {"content": "Найди", "eng_content": "Find", "nested": [{"description": "x", "eng_description": "y"}]}
    assert tasks._strip_translation_notes(row) == {"content": "Найди", "nested": [{"description": "x"}]}


def test_subsample_is_stable_and_same_for_paired_languages():
    en = [Example(id=f"simple_{i}", pair_key=f"simple_{i}", messages=[], reference=None, stratum="a") for i in range(50)]
    ru = [Example(id=f"RU_simple_{i}", pair_key=f"simple_{i}", messages=[], reference=None, stratum="a")
          for i in reversed(range(50))]
    picked_en, picked_ru = tasks.subsample(en, {"a": 10}), tasks.subsample(ru, {"a": 10})
    assert [e.pair_key for e in picked_en] == [e.pair_key for e in picked_ru]
    assert len(picked_en) == 10
    assert tasks.subset_sha256(picked_en) == tasks.subset_sha256(list(reversed(picked_en)))


# --- summaries --------------------------------------------------------------------------------------------------

def test_confidence_interval():
    low, high = run.confidence_interval([1.0] * 50 + [0.0] * 50)
    assert low == pytest.approx(0.404, abs=1e-3) and high == pytest.approx(0.596, abs=1e-3)
    assert run.confidence_interval([0.0] * 20)[0] == 0.0
    low, high = run.confidence_interval([0.2, 0.4, 0.6])
    assert low < 0.4 < high


def test_changed_answers_counts_both_directions():
    base = {"a": {"score": 1.0}, "b": {"score": 1.0}, "c": {"score": 0.0}, "d": {"score": 0.0}}
    other = {"a": {"score": 1.0}, "b": {"score": 0.0}, "c": {"score": 1.0}, "d": {"score": 0.0}}
    assert run.changed_answers(base, other) == {"n": 4, "worse": 1, "better": 1, "share": 0.5}
    assert run.changed_answers({"a": {"score": 0.3}}, {"a": {"score": 0.4}}) is None


def test_perplexity_output_parsing():
    base = closeness.parse_base((FIXTURES / "perplexity_base.txt").read_text(encoding="utf-8"))
    assert base == {"ppl": 9.4385, "ppl_err": 0.24357}
    kld = closeness.parse_kld((FIXTURES / "perplexity_kld.txt").read_text(encoding="utf-8"))
    assert kld["ppl"] == 9.787876 and kld["ppl_base"] == 9.425751
    assert kld["kld_mean"] == 0.043584 and kld["kld_median"] == 0.0243
    assert kld["delta_p_mean_pct"] == -0.804 and kld["same_top_pct"] == 88.598
    with pytest.raises(closeness.ClosenessError):
        closeness.parse_kld("error: failed to load model")


# --- records ----------------------------------------------------------------------------------------------------

def make_record(**overrides) -> dict:
    record = {
        "schema_version": rec.SCHEMA_VERSION, "eval_id": "e1", "started_at": "2026-11-15T00:00:00Z", "kind": "task",
        "label": None, "status": "ok", "error": None, "tooling_commit": "abc1234",
        "model": {"sha256": "0" * 64, "file": "m.gguf", "file_bytes": 10}, "engine": {"build_id": "b"},
        "params": {}, "task": {"name": "bfcl-en"}, "metrics": {"value": 0.5, "ci95": [0.4, 0.6]}, "raw_dir": "raw/x",
    }
    record.update(overrides)
    return record


def test_quality_record_round_trip_and_latest(tmp_path):
    rec.append(make_record(metrics={"value": 0.5}), tmp_path)
    rec.append(make_record(eval_id="e2", metrics={"value": 0.7}), tmp_path)
    rec.append(make_record(eval_id="e3", label="threads4", metrics={"value": 0.6}), tmp_path)
    rec.append(make_record(eval_id="e4", status="failed", error="server died", metrics={}), tmp_path)
    latest = rec.latest_ok(rec.load(tmp_path), "task")
    assert {key: r["eval_id"] for key, r in latest.items()} == {
        ("0" * 64 + "|", "bfcl-en"): "e2", ("0" * 64 + "|threads4", "bfcl-en"): "e3"}


def test_quality_record_validation():
    with pytest.raises(ValueError, match="value"):
        rec.validate(make_record(metrics={}))
    with pytest.raises(ValueError, match="corpus"):
        rec.validate(make_record(kind="closeness"))
    with pytest.raises(ValueError, match="error"):
        rec.validate(make_record(status="failed", metrics={}))
