"""Quality evaluation: answer parsing, scoring helpers, subsampling and records. No model or server needed."""

import json
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


def test_side_group_is_reported_but_left_out_of_the_headline():
    task = tasks.TASKS["bfcl-en"]
    answers = {"a": {"score": 1.0, "stratum": "simple"}, "b": {"score": 0.0, "stratum": "parallel"},
               "c": {"score": 1.0, "stratum": "irrelevance"}, "d": {"score": 1.0, "stratum": "irrelevance"}}
    headline, groups = run.split_answers(task, answers)
    assert sorted(headline) == ["a", "b"] and sorted(groups) == ["irrelevance", "parallel", "simple"]
    assert run.score_block(list(headline.values()))["value"] == 0.5
    assert run.split_answers(tasks.TASKS["mgsm-en"], answers) == (answers, {})


def test_task_near_a_blind_guess_does_not_separate_variants():
    assert not run.separates_variants(tasks.TASKS["mgsm-ru"], 0.048)
    assert run.separates_variants(tasks.TASKS["mgsm-en"], 0.46)
    assert not run.separates_variants(tasks.TASKS["mmlu-ru"], 0.326)  # four options: a guess scores 0.25
    assert run.separates_variants(tasks.TASKS["mmlu-en"], 0.397)
    assert run.separates_variants(tasks.TASKS["samsum-ru"], 0.05)  # ROUGE has no guessing level


def test_comparison_uses_headline_groups_and_flat_mark(tmp_path):
    from koffient_tooling.quality.cli import comparison

    def write(eval_id: str, sha: str, size: int, name: str, rows: list[tuple[str, str, float]]):
        raw = tmp_path / "raw" / eval_id
        raw.mkdir(parents=True)
        (raw / run.ANSWERS_FILE).write_text("".join(
            json.dumps({"pair_key": key, "stratum": stratum, "score": score}) + "\n"
            for key, stratum, score in rows), encoding="utf-8")
        scores = [score for _, _, score in rows]
        rec.append(make_record(
            eval_id=eval_id, raw_dir=f"raw/{eval_id}",
            model={"sha256": sha, "file": f"{eval_id}.gguf", "file_bytes": size},
            task={"name": name, "role": "primary", "scenario": "x", "lang": "en"},
            metrics={"value": sum(scores) / len(scores), "ci95": [0.0, 1.0], "n": len(rows), "n_truncated": 3}),
            tmp_path)

    write("big-bfcl", "a" * 64, 200, "bfcl-en", [("1", "simple", 1.0), ("2", "simple", 1.0), ("3", "irrelevance", 0.0)])
    write("small-bfcl", "b" * 64, 100, "bfcl-en", [("1", "simple", 0.0), ("2", "simple", 1.0), ("3", "irrelevance", 1.0)])
    write("big-math", "a" * 64, 200, "mgsm-ru", [("1", "", 0.0), ("2", "", 0.0)])
    data = comparison(rec.load(tmp_path), tmp_path, None)
    small = data["tasks"]["bfcl-en"]["b" * 64 + "|"]
    assert data["base"]["model"]["sha256"] == "a" * 64 and data["flat"] == ["mgsm-ru"]
    assert small["value"] == 0.5 and small["n"] == 2 and small["truncated"] == 3
    assert small["changed"] == {"n": 2, "worse": 1, "better": 0, "share": 0.5}
    assert list(small["groups"]) == ["simple", "irrelevance"] and small["groups"]["irrelevance"]["side"]
    assert small["groups"]["irrelevance"]["changed"]["better"] == 1


def test_latest_skips_runs_with_failed_requests_or_hidden_penalties(tmp_path):
    rec.append(make_record(eval_id="good", metrics={"value": 0.5, "n_request_errors": 0}), tmp_path)
    rec.append(make_record(eval_id="errors", metrics={"value": 0.2, "n_request_errors": 40}), tmp_path)
    rec.append(make_record(eval_id="penalty", metrics={"value": 0.3},
                           model={"sha256": "0" * 64, "file": "m.gguf", "file_bytes": 10, "build": {"recipe": "x"}}),
               tmp_path)
    latest = rec.latest_ok(rec.load(tmp_path), "task")
    assert [r["eval_id"] for r in latest.values()] == ["good"]


def test_unparsed_output_is_a_wrong_answer_not_a_broken_run(tmp_path):
    from koffient_tooling.quality.server import ServerError, UnparsedOutput

    class Pool:
        size = 2

        def chat(self, messages, max_tokens, tools, grammar):
            text = messages[0]["content"]
            if text == "bad":
                raise UnparsedOutput('{"error":{"code":500,"message":"does not match the expected format"}}')
            if text == "down":
                raise ServerError("connection refused")
            return {"content": "ok", "tool_calls": [], "finish_reason": "stop", "prompt_tokens": 1,
                    "completion_tokens": 1}

    task = tasks.TASKS["mgsm-en"]
    examples = [tasks.Example(id=t, pair_key=t, messages=[{"role": "user", "content": t}], reference="1")
                for t in ("good", "bad", "down")]
    metrics = run.run_task(Pool(), task, examples, lambda ex, ans: {"score": 1.0, "detail": {}}, tmp_path)
    assert metrics["n_unparsed"] == 1 and metrics["n_request_errors"] == 1
    rows = run.load_answers(tmp_path)
    assert rows["good"]["score"] == 1.0
    assert rows["bad"]["score"] == 0.0 and rows["bad"]["error"] is None and rows["bad"]["detail"]["error"] == "unparsed_output"
    assert rows["down"]["error"].startswith("ServerError")


def test_pool_restarts_a_dead_server_and_asks_again(tmp_path):
    import requests

    from koffient_tooling.quality import server as srv

    class FlakyServer:
        def __init__(self):
            self.dead, self.calls, self.restarted = False, 0, 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.dead = True
                raise requests.ConnectionError("connection refused")
            return {"content": "answer"}

        def is_dead(self):
            return self.dead

        def restart(self):
            self.dead, self.restarted = False, self.restarted + 1

    pool = srv.ServerPool(Path("."), Path("m.gguf"), tmp_path, servers=1)  # not entered: the queue is empty
    flaky = FlakyServer()
    pool._free.put(flaky)
    assert pool.chat([], 4)["content"] == "answer"
    assert flaky.restarted == 1 and pool.restarts == 1

    class AliveButFailing(FlakyServer):
        def chat(self, *args, **kwargs):
            raise requests.ConnectionError("timeout")  # the process is alive: not a crash, do not retry

    pool._free.get_nowait()
    pool._free.put(AliveButFailing())
    with pytest.raises(requests.ConnectionError):
        pool.chat([], 4)

    class Unparsed(FlakyServer):
        def chat(self, *args, **kwargs):
            raise srv.UnparsedOutput("does not match the expected format")

        def is_dead(self):
            raise AssertionError("an answer from a live server must not trigger the crash check")

    pool._free.get_nowait()
    pool._free.put(Unparsed())
    with pytest.raises(srv.UnparsedOutput):
        pool.chat([], 4)


def test_every_request_switches_penalties_off(monkeypatch):
    from koffient_tooling.quality import server as srv

    sent = {}

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    def fake_post(url, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr(srv.requests, "post", fake_post)
    server = srv.LlamaServer(Path("."), Path("m.gguf"), Path("log.txt"))
    server.url = "http://127.0.0.1:1"
    server.chat([{"role": "user", "content": "hi"}], max_tokens=4)
    # A GGUF can carry its own sampling defaults (e.g. repeat_penalty 1.1); the request must override them.
    assert sent["temperature"] == 0 and sent["repeat_penalty"] == 1.0
    assert sent["presence_penalty"] == sent["frequency_penalty"] == sent["dry_multiplier"] == 0.0


def test_failed_base_run_leaves_no_logits_file(tmp_path, monkeypatch):
    logits = tmp_path / "kld" / "base-en.kld"

    def fake_run(command, log_path, timeout_s, ok):
        Path(command[command.index("--kl-divergence-base") + 1]).write_bytes(b"half-written")
        return "Final estimate: PPL = 12.5 +/- 0.3" if ok else "[1]nan,[2]nan,"

    monkeypatch.setattr(closeness, "_run", lambda c, l, t: fake_run(c, l, t, ok=False))
    with pytest.raises(closeness.ClosenessError):
        closeness.save_base_logits(tmp_path, tmp_path / "m.gguf", tmp_path / "c.txt", logits, tmp_path / "b.log")
    assert list((tmp_path / "kld").iterdir()) == []

    monkeypatch.setattr(closeness, "_run", lambda c, l, t: fake_run(c, l, t, ok=True))
    assert closeness.save_base_logits(tmp_path, tmp_path / "m.gguf", tmp_path / "c.txt", logits,
                                      tmp_path / "b.log")["ppl"] == 12.5
    assert [p.name for p in (tmp_path / "kld").iterdir()] == ["base-en.kld"]


def test_closeness_is_shown_only_against_the_chosen_base(tmp_path):
    from koffient_tooling.quality.cli import comparison

    official, ours, quant = "a" * 64, "b" * 64, "c" * 64

    def closeness_record(eval_id: str, model_sha: str, size: int, base_sha: str, kld: float | None):
        metrics = {"ppl": 10.0} if kld is None else {"ppl": 10.5, "kld_mean": kld, "same_top_pct": 90.0}
        rec.append(make_record(eval_id=eval_id, kind="closeness", task=None, corpus={"lang": "en"},
                               base={"sha256": base_sha}, metrics=metrics,
                               model={"sha256": model_sha, "file": f"{eval_id}.gguf", "file_bytes": size}), tmp_path)

    closeness_record("official-base", official, 300, official, None)
    closeness_record("quant-vs-official", quant, 100, official, 0.03)
    closeness_record("ours-base", ours, 200, ours, None)
    closeness_record("quant-vs-ours", quant, 100, ours, 0.01)
    by_default = comparison(rec.load(tmp_path), tmp_path, None)  # the largest file is the base
    assert by_default["closeness"]["en"][quant + "|"]["kld_mean"] == 0.03
    assert ours + "|" not in by_default["closeness"]["en"]
    chosen = comparison(rec.load(tmp_path), tmp_path, "ours-base")
    assert chosen["closeness"]["en"][quant + "|"]["kld_mean"] == 0.01
    assert chosen["closeness_ids"] == ["ours-base", "quant-vs-ours"]


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
