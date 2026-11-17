"""Scoring of a single answer.

The BFCL and IFEval checkers are not reimplemented here: they are loaded from the original projects
(<work-dir>/third_party/rfcb and the installed `ifeval` package). Everything else is a few lines each.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

from .tasks import Dirs, Example, Task

BFCL_MODEL_NAME = "koffient-local"


class ScoringError(RuntimeError):
    """A checker could not be loaded or crashed in a way that is not the model's fault."""


# --- function calling ---------------------------------------------------------------------------------------------

def load_bfcl_checker(third_party: Path):
    """Import `ast_checker` from the BFCL fork without its model handlers (they need dozens of SDKs)."""
    root = third_party / "rfcb"
    if not (root / "bfcl_eval").is_dir():
        raise ScoringError(f"BFCL checker not found under {root}")
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    # underscore_to_dot=True: tools are sent with '_' instead of '.', the checker maps expected names the same way.
    stub.MODEL_CONFIG_MAPPING = {BFCL_MODEL_NAME: types.SimpleNamespace(underscore_to_dot=True)}
    sys.modules.setdefault("bfcl_eval.constants.model_config", stub)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    except Exception as error:  # noqa: BLE001 - any import failure means the checker is unusable
        raise ScoringError(f"cannot import the BFCL checker: {error}") from error
    return ast_checker


def score_bfcl(ast_checker, example: Example, answer: dict) -> dict:
    reference = example.reference
    calls = answer.get("tool_calls") or []
    if reference["category"] == "irrelevance":
        return {"score": 0.0 if calls else 1.0, "detail": {"calls": len(calls)}}
    if not calls:
        return {"score": 0.0, "detail": {"error": "no_call"}}
    if any(not isinstance(call["arguments"], dict) for call in calls):
        return {"score": 0.0, "detail": {"error": "arguments_not_json_object"}}
    decoded = [{call["name"]: call["arguments"]} for call in calls]
    try:
        result = ast_checker(reference["functions"], decoded, reference["ground_truth"], "Python",
                             reference["category"], BFCL_MODEL_NAME)
    except Exception as error:  # noqa: BLE001 - malformed model output can crash the checker; count as wrong
        return {"score": 0.0, "detail": {"error": f"checker_crash: {type(error).__name__}: {error}"}}
    detail = {} if result["valid"] else {"error": result.get("error_type"), "message": result.get("error")}
    return {"score": 1.0 if result["valid"] else 0.0, "detail": detail}


# --- JSON by schema -----------------------------------------------------------------------------------------------

def extract_json(text: str):
    """The first JSON value in the answer; code fences around it are allowed."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON value in the answer")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return value


def score_json_schema(example: Example, answer: dict) -> dict:
    import jsonschema

    try:
        value = extract_json(answer.get("content") or "")
    except ValueError as error:
        return {"score": 0.0, "detail": {"error": "not_json", "message": str(error)[:200]}}
    try:
        schema = json.loads(example.reference)
        validator = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)(schema)
        error = next(iter(validator.iter_errors(value)), None)
    except Exception as crash:  # noqa: BLE001 - a broken schema fails for every model alike
        return {"score": 0.0, "detail": {"error": "schema_unusable", "message": str(crash)[:200]}}
    if error is not None:
        return {"score": 0.0, "detail": {"error": "schema_violation", "message": error.message[:200]}}
    return {"score": 1.0, "detail": {}}


# --- instruction following ----------------------------------------------------------------------------------------

def load_ifeval(lang: str, nltk_data: Path):
    try:
        import nltk

        if str(nltk_data) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data))
        import ifeval
    except Exception as error:  # noqa: BLE001
        raise ScoringError(f"cannot import the IFEval checker: {error}") from error
    registry = ifeval.instruction_registry if lang == "en" else ifeval.ru_instruction_registry
    return ifeval.Evaluator(registry), ifeval.InputExample


def score_ifeval(evaluator, input_cls, example: Example, answer: dict) -> dict:
    reference = example.reference
    inp = input_cls(instruction_id_list=reference["instruction_id_list"], prompt=reference["prompt"],
                    kwargs=reference["kwargs"])
    response = answer.get("content") or ""
    try:
        strict = evaluator.test_instruction_following_strict(inp, response)
        loose = evaluator.test_instruction_following_loose(inp, response)
    except Exception as error:  # noqa: BLE001 - checker bugs on odd answers; count as not followed
        return {"score": 0.0, "detail": {"error": f"checker_crash: {type(error).__name__}: {error}"}}
    followed = strict.follow_instruction_list
    return {
        "score": 1.0 if strict.follow_all_instructions else 0.0,
        "detail": {"loose": bool(loose.follow_all_instructions),
                   "instructions_followed": sum(followed), "instructions": len(followed)},
    }


# --- summarization ------------------------------------------------------------------------------------------------

class WordTokenizer:
    """rouge_score's default tokenizer keeps only [a-z0-9], which deletes Russian text entirely."""

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


def make_rouge():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], tokenizer=WordTokenizer())


def score_rouge(scorer, example: Example, answer: dict) -> dict:
    result = scorer.score(example.reference, answer.get("content") or "")["rougeL"]
    return {"score": round(result.fmeasure, 4),
            "detail": {"precision": round(result.precision, 4), "recall": round(result.recall, 4)}}


# --- math ---------------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:[,  ]\d{3})*(?:[.,]\d+)?")
_ANSWER_MARK = re.compile(r"(?:answer|ответ)\s*[:：]", re.I)


def parse_number(token: str) -> float | None:
    token = token.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"-?\d+(,\d{3})+(\.\d+)?", token):
        token = token.replace(",", "")
    try:
        return float(token.replace(",", "."))
    except ValueError:
        pass
    # Mixed separators such as "8,956,25" (the old rules could not read them): the last separator is the decimal
    # point, the others group thousands. Only tokens that used to fail reach this line, so earlier scores stand.
    head, _, tail = token.replace(",", ".").rpartition(".")
    try:
        return float(head.replace(".", "") + "." + tail)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """The number after 'Answer:' / 'Ответ:' when that mark is on the last line, otherwise the last number."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    marks = list(_ANSWER_MARK.finditer(lines[-1]))
    if marks:
        found = _NUMBER.search(lines[-1], marks[-1].end())
        if found:
            return parse_number(found.group())
    numbers = _NUMBER.findall(text)
    return parse_number(numbers[-1]) if numbers else None


def score_number(example: Example, answer: dict) -> dict:
    value = extract_final_number(answer.get("content") or "")
    correct = value is not None and abs(value - float(example.reference)) < 1e-6
    return {"score": 1.0 if correct else 0.0, "detail": {"parsed": value}}


# --- knowledge ----------------------------------------------------------------------------------------------------

_LATIN_CHOICE = re.compile(r"(?<!\w)([ABCD])(?!\w)")
_CYRILLIC_CHOICE = re.compile(r"^\s*([АВС])\s*[).:,]?\s*$")
_CYRILLIC_TO_LATIN = {"А": "A", "В": "B", "С": "C"}


def extract_choice(text: str) -> str | None:
    found = _LATIN_CHOICE.search(text)
    if found:
        return found.group(1)
    # Cyrillic look-alikes count only when they are the whole answer: "В" also starts ordinary Russian sentences.
    found = _CYRILLIC_CHOICE.match(text)
    return _CYRILLIC_TO_LATIN[found.group(1)] if found else None


def score_choice(example: Example, answer: dict) -> dict:
    choice = extract_choice(answer.get("content") or "")
    return {"score": 1.0 if choice == example.reference else 0.0, "detail": {"parsed": choice}}


# --- dispatch -----------------------------------------------------------------------------------------------------

def make_scorer(task: Task, dirs: Dirs, nltk_data: Path):
    """Return score(example, answer) -> {'score': float in [0, 1], 'detail': dict} for one task."""
    if task.scenario == "function_calling":
        checker = load_bfcl_checker(dirs.third_party)
        return lambda example, answer: score_bfcl(checker, example, answer)
    if task.scenario == "json_schema":
        return score_json_schema
    if task.scenario == "instructions":
        evaluator, input_cls = load_ifeval(task.lang, nltk_data)
        return lambda example, answer: score_ifeval(evaluator, input_cls, example, answer)
    if task.scenario == "summarization":
        rouge = make_rouge()
        return lambda example, answer: score_rouge(rouge, example, answer)
    if task.scenario == "math":
        return score_number
    if task.scenario == "knowledge":
        return score_choice
    raise ScoringError(f"no scorer for scenario '{task.scenario}'")
