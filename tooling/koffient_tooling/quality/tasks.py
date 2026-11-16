"""The fast evaluation set: where each task's data lives, how a request is built, which examples are used.

Datasets are not part of the repository. They are expected under <work-dir>/datasets and
<work-dir>/third_party at the revisions listed in TASKS (see tooling/README.md).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SUBSAMPLE_SEED = "koffient-fast-v1"


@dataclass
class Example:
    id: str
    pair_key: str  # the same example in the other language has the same pair_key
    messages: list[dict]
    reference: object
    tools: list[dict] | None = None
    stratum: str = ""


@dataclass(frozen=True)
class Dirs:
    datasets: Path
    third_party: Path


@dataclass(frozen=True)
class Task:
    name: str
    scenario: str  # function_calling | json_schema | instructions | summarization | math | knowledge
    role: str  # primary | secondary | control
    lang: str
    dataset: str
    revision: str
    license: str
    metric: str
    max_tokens: int
    load: Callable[[Dirs], list[Example]] = field(repr=False)
    grammar: str | None = None  # GBNF grammar that constrains the answer, if any
    chance: float | None = 0.0  # score of a blind guess; None when the metric has no such level
    # Groups of examples reported next to the headline score, not inside it.
    side_strata: tuple[str, ...] = ()


def _rank(pair_key: str) -> str:
    return hashlib.sha256(f"{SUBSAMPLE_SEED}:{pair_key}".encode()).hexdigest()


def subsample(examples: list[Example], quotas: dict[str, int]) -> list[Example]:
    """Pick a fixed pseudo-random subset per stratum; the choice depends only on pair_key."""
    picked: list[Example] = []
    for stratum, quota in quotas.items():
        pool = sorted((e for e in examples if e.stratum == stratum), key=lambda e: _rank(e.pair_key))
        picked.extend(pool[:quota])
    return sorted(picked, key=lambda e: _rank(e.pair_key))


def subset_sha256(examples: list[Example]) -> str:
    return hashlib.sha256("\n".join(sorted(e.id for e in examples)).encode()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# --- function calling: BFCL v3 and its Russian translation (RFCB) -------------------------------------------------

BFCL_QUOTAS = {"simple": 100, "multiple": 70, "parallel": 70, "irrelevance": 60}

# Type names used in BFCL function docs -> JSON Schema types (bfcl_eval/constants/type_mappings.py).
_BFCL_TYPES = {
    "integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
    "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object", "tuple": "array",
    "any": "string",
}


def _strip_translation_notes(node):
    """RFCB keeps the English original next to each translated field ('eng_content', 'eng_description')."""
    if isinstance(node, dict):
        return {k: _strip_translation_notes(v) for k, v in node.items() if not k.startswith("eng_")}
    if isinstance(node, list):
        return [_strip_translation_notes(v) for v in node]
    return node


def _to_json_schema(node: dict) -> dict:
    """Port of bfcl_eval `_cast_to_openai_type`, applied recursively to one parameter description."""
    original = node.get("type", "string")
    node["type"] = _BFCL_TYPES.get(original, "string")
    if original == "float":
        node["format"] = "float"
        node["description"] = node.get("description", "") + " This is a float type value."
    if node["type"] == "object" and isinstance(node.get("properties"), dict):
        for child in node["properties"].values():
            _to_json_schema(child)
    if node["type"] == "array" and isinstance(node.get("items"), dict):
        _to_json_schema(node["items"])
    return node


def bfcl_tool(function: dict) -> dict:
    """BFCL function doc -> OpenAI-style tool. Dots in names become underscores, as in BFCL's FC mode."""
    function = copy.deepcopy(function)
    parameters = function.get("parameters") or {"type": "dict", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": function["name"].replace(".", "_"),
            "description": function.get("description", ""),
            "parameters": _to_json_schema(parameters),
        },
    }


def _load_bfcl(prefix: str) -> Callable[[Dirs], list[Example]]:
    def load(dirs: Dirs) -> list[Example]:
        data = dirs.third_party / "rfcb" / "bfcl_eval" / "data"
        examples = []
        for category in BFCL_QUOTAS:
            answers_file = data / "possible_answer" / f"{prefix}{category}.json"
            answers = {row["id"]: row["ground_truth"] for row in _jsonl(answers_file)} if answers_file.is_file() else {}
            for row in _jsonl(data / f"{prefix}{category}.json"):
                row = _strip_translation_notes(row)
                messages = [{"role": m["role"], "content": m["content"]} for m in row["question"][0]]
                examples.append(Example(
                    id=row["id"],
                    pair_key=row["id"].removeprefix("RU_"),
                    messages=messages,
                    tools=[bfcl_tool(f) for f in row["function"]],
                    reference={"category": category, "functions": row["function"],
                               "ground_truth": answers.get(row["id"])},
                    stratum=category,
                ))
        return subsample(examples, BFCL_QUOTAS)

    return load


# --- JSON by schema: JSONSchemaBench ------------------------------------------------------------------------------

JSON_PROMPT = (
    "Generate one JSON object that is valid against the JSON Schema below. "
    "Output only the JSON object, without explanations.\n\n{schema}"
)
JSON_QUOTAS = {"Glaiveai2K": 100, "Github_easy": 100}


def _load_jsonschema(dirs: Dirs) -> list[Example]:
    examples = []
    for part in JSON_QUOTAS:
        for row in _parquet(dirs.datasets / "jsonschemabench" / f"{part}-test.parquet"):
            examples.append(Example(
                id=f"{part}:{row['unique_id']}", pair_key=f"{part}:{row['unique_id']}",
                messages=_user(JSON_PROMPT.format(schema=row["json_schema"])),
                reference=row["json_schema"], stratum=part,
            ))
    return subsample(examples, JSON_QUOTAS)


# --- instruction following: IFEval and ruIFEval -------------------------------------------------------------------

def _load_ifeval(lang: str) -> Callable[[Dirs], list[Example]]:
    def load(dirs: Dirs) -> list[Example]:
        examples = []
        for line in _jsonl(dirs.datasets / "ifeval" / f"{lang}.jsonl"):
            row = json.loads(line["content"]) if "content" in line else line
            examples.append(Example(
                id=str(row["key"]), pair_key=str(row["key"]), messages=_user(row["prompt"]),
                reference={"prompt": row["prompt"], "instruction_id_list": row["instruction_id_list"],
                           "kwargs": row["kwargs"]},
            ))
        return subsample(examples, {"": 300})

    return load


# --- dialogue summarization: SAMSum and its Russian machine translation -------------------------------------------

SUMMARY_PROMPT = {
    "en": "Summarize the following conversation in one or two sentences.\n\n{dialogue}",
    "ru": "Кратко перескажи эту переписку в одном-двух предложениях.\n\n{dialogue}",
}


def _load_samsum(lang: str) -> Callable[[Dirs], list[Example]]:
    def load(dirs: Dirs) -> list[Example]:
        if lang == "en":
            with open(dirs.datasets / "samsum" / "test.csv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = _parquet(next((dirs.datasets / "samsum-ru").glob("test-*.parquet")))
        examples = [
            Example(id=str(r["id"]), pair_key=str(r["id"]),
                    messages=_user(SUMMARY_PROMPT[lang].format(dialogue=r["dialogue"].replace("\r\n", "\n"))),
                    reference=r["summary"])
            for r in rows
        ]
        return subsample(examples, {"": 200})

    return load


# --- math: MGSM (the first 250 GSM8K test problems, translated by people) -----------------------------------------

MATH_PROMPT = {
    "en": "Solve the problem step by step. Finish with the final answer as a single number.\n\n{question}",
    "ru": "Реши задачу по шагам. В конце напиши итоговый ответ одним числом.\n\n{question}",
}


def _load_mgsm(lang: str) -> Callable[[Dirs], list[Example]]:
    def load(dirs: Dirs) -> list[Example]:
        rows = _parquet(dirs.datasets / "mgsm" / f"{lang}-test.parquet")
        return [
            Example(id=str(i), pair_key=str(i), messages=_user(MATH_PROMPT[lang].format(question=r["question"])),
                    reference=r["answer_number"])
            for i, r in enumerate(rows)
        ]

    return load


# --- knowledge: MMLU and ruMMLU -----------------------------------------------------------------------------------

MMLU_PROMPT = (
    "The following is a multiple choice question about {subject}. Only one of the options A, B, C, D is correct.\n"
    "{question}\nA {a}\nB {b}\nC {c}\nD {d}\nWrite the letter of the correct answer\nAnswer:"
)
MMLU_PER_SUBJECT = 5
# The answer is forced to one letter: the generation-side equivalent of comparing the letters' likelihoods.
MMLU_GRAMMAR = "root ::= [ABCD]"


def _mmlu_subsample(examples: list[Example]) -> list[Example]:
    subjects = sorted({e.stratum for e in examples})
    return subsample(examples, {s: MMLU_PER_SUBJECT for s in subjects})


def _load_mmlu(dirs: Dirs) -> list[Example]:
    examples, seen = [], {}
    for row in _parquet(dirs.datasets / "mmlu" / "test.parquet"):
        subject = row["subject"]
        index = seen[subject] = seen.get(subject, -1) + 1
        a, b, c, d = row["choices"]
        prompt = MMLU_PROMPT.format(subject=subject.replace("_", " "), question=row["question"], a=a, b=b, c=c, d=d)
        examples.append(Example(id=f"{subject}:{index}", pair_key=f"{subject}:{index}", messages=_user(prompt),
                                reference="ABCD"[row["answer"]], stratum=subject))
    return _mmlu_subsample(examples)


def _load_rummlu(dirs: Dirs) -> list[Example]:
    examples, seen = [], {}
    for row in _jsonl(dirs.datasets / "rummlu" / "test.jsonl"):
        subject = row["meta"]["domain"]
        index = seen[subject] = seen.get(subject, -1) + 1
        prompt = row["instruction"].format(**row["inputs"])
        # ruMMLU dropped 30 questions, so pair_key matches the English example only approximately.
        examples.append(Example(id=str(row["meta"]["id"]), pair_key=f"{subject}:{index}", messages=_user(prompt),
                                reference=row["outputs"], stratum=subject))
    return _mmlu_subsample(examples)


RFCB = ("sir-timio/RFCB", "a6b4896")
IFEVAL = ("kaleinaNyan/instruction-following-eval", "843295a")

TASKS: dict[str, Task] = {t.name: t for t in [
    Task("bfcl-en", "function_calling", "primary", "en", f"{RFCB[0]} (BFCL v3)", RFCB[1], "Apache-2.0",
         "ast_accuracy", 256, _load_bfcl("BFCL_v3_"), side_strata=("irrelevance",)),
    Task("bfcl-ru", "function_calling", "primary", "ru", f"{RFCB[0]} (BFCL v3 RU)", RFCB[1], "none stated",
         "ast_accuracy", 256, _load_bfcl("BFCL_v3_RU_"), side_strata=("irrelevance",)),
    Task("jsonschema", "json_schema", "primary", "en", "epfl-dlab/JSONSchemaBench", "5bd0f46", "MIT",
         "schema_valid", 512, _load_jsonschema),
    Task("ifeval-en", "instructions", "primary", "en", IFEVAL[0], IFEVAL[1], "Apache-2.0",
         "strict_prompt_accuracy", 1280, _load_ifeval("en")),
    Task("ifeval-ru", "instructions", "primary", "ru", IFEVAL[0], IFEVAL[1], "none stated upstream",
         "strict_prompt_accuracy", 1280, _load_ifeval("ru")),
    Task("samsum-en", "summarization", "secondary", "en", "knkarthick/samsum", "6b929ff", "CC BY-NC-ND 4.0",
         "rouge_l_f", 192, _load_samsum("en"), chance=None),
    Task("samsum-ru", "summarization", "secondary", "ru", "d0rj/samsum-ru", "1793e9d", "CC BY-NC-ND 4.0",
         "rouge_l_f", 192, _load_samsum("ru"), chance=None),
    Task("mgsm-en", "math", "control", "en", "juletxara/mgsm", "b2f13d4", "unclear (CC BY-SA 4.0 / MIT)",
         "exact_match", 512, _load_mgsm("en")),
    Task("mgsm-ru", "math", "control", "ru", "juletxara/mgsm", "b2f13d4", "unclear (CC BY-SA 4.0 / MIT)",
         "exact_match", 512, _load_mgsm("ru")),
    Task("mmlu-en", "knowledge", "control", "en", "cais/mmlu", "c30699e", "MIT",
         "accuracy", 2, _load_mmlu, MMLU_GRAMMAR, chance=0.25),
    Task("mmlu-ru", "knowledge", "control", "ru", "MERA-evaluation/MERA (ruMMLU)", "15c0e5b", "MIT",
         "accuracy", 2, _load_rummlu, MMLU_GRAMMAR, chance=0.25),
]}
