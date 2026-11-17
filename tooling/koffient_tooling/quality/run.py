"""Run one task against a running server: ask, save every answer, score, summarize."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .server import ServerError, ServerPool, UnparsedOutput
from .tasks import Example, Task

ANSWERS_FILE = "answers.jsonl"


def confidence_interval(scores: list[float]) -> list[float]:
    """95% interval for the mean: Wilson for 0/1 scores, normal approximation otherwise."""
    n = len(scores)
    if n == 0:
        return [0.0, 0.0]
    mean = sum(scores) / n
    z = 1.96
    if all(s in (0.0, 1.0) for s in scores):
        centre = (mean + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]
    if n == 1:
        return [round(mean, 4), round(mean, 4)]
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
    return [round(mean - z * sd / math.sqrt(n), 4), round(mean + z * sd / math.sqrt(n), 4)]


def summarize(task: Task, rows: list[dict], wall_s: float) -> dict:
    scores = [row["score"] for row in rows]
    return {
        "name": task.metric,
        "value": round(sum(scores) / len(scores), 4) if scores else None,
        "ci95": confidence_interval(scores),
        "n": len(rows),
        "n_request_errors": sum(1 for row in rows if row["error"]),
        "n_unparsed": sum(1 for row in rows if row.get("unparsed")),
        "n_truncated": sum(1 for row in rows if row["finish_reason"] == "length"),
        "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in rows),
        "completion_tokens": sum(row["completion_tokens"] or 0 for row in rows),
        "wall_s": round(wall_s, 1),
    }


def run_task(server: ServerPool, task: Task, examples: list[Example], scorer, raw_dir: Path) -> dict:
    """Returns the metrics block. Every answer with its score is written to raw_dir/answers.jsonl."""

    def ask(example: Example) -> dict:
        try:
            answer = server.chat(example.messages, task.max_tokens, example.tools, task.grammar)
            answer["error"] = None
        except UnparsedOutput as error:
            # The model did answer; the answer is wrong whatever the task (even "no call needed": it tried a call).
            answer = {"content": "", "tool_calls": [], "finish_reason": "unparsed", "prompt_tokens": None,
                      "completion_tokens": None, "error": None, "unparsed": str(error)[:300]}
        except (ServerError, OSError, ValueError, KeyError) as error:
            answer = {"content": "", "tool_calls": [], "finish_reason": None, "prompt_tokens": None,
                      "completion_tokens": None, "error": f"{type(error).__name__}: {error}"[:500]}
        return answer

    started = time.monotonic()
    restarts_before = getattr(server, "restarts", 0)
    with ThreadPoolExecutor(max_workers=server.size) as pool:
        answers = list(pool.map(ask, examples))
    wall_s = time.monotonic() - started
    restarts = getattr(server, "restarts", 0) - restarts_before

    rows = []
    for example, answer in zip(examples, answers):
        if answer["error"]:
            scored = {"score": 0.0, "detail": {"error": "request"}}
        elif answer.get("unparsed"):
            scored = {"score": 0.0, "detail": {"error": "unparsed_output"}}
        else:
            scored = scorer(example, answer)
        rows.append({"id": example.id, "pair_key": example.pair_key, "stratum": example.stratum, **answer, **scored})

    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / ANSWERS_FILE, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {**summarize(task, rows, wall_s), "n_server_restarts": restarts}


def load_answers(raw_dir: Path) -> dict[str, dict]:
    with open(raw_dir / ANSWERS_FILE, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["pair_key"]: row for row in rows}


# A task separates variants of a model only if the base variant stands at least this far above a blind guess:
# with 200-300 examples the interval is about +-0.05 wide, so a smaller gap is lost in it even after a full collapse.
MIN_HEADROOM = 0.10


def score_block(rows: list[dict]) -> dict:
    scores = [row["score"] for row in rows]
    return {"value": round(sum(scores) / len(scores), 4) if scores else None,
            "ci95": confidence_interval(scores), "n": len(rows)}


def split_answers(task: Task, answers: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    """Answers that make up the headline score, and all answers by group for a task that has side groups."""
    if not task.side_strata:
        return answers, {}
    groups: dict[str, dict[str, dict]] = {}
    for key, row in answers.items():
        groups.setdefault(row["stratum"], {})[key] = row
    headline = {key: row for key, row in answers.items() if row["stratum"] not in task.side_strata}
    return headline, groups


def separates_variants(task: Task, base_value: float) -> bool:
    return task.chance is None or base_value - task.chance >= MIN_HEADROOM


def changed_answers(base: dict[str, dict], other: dict[str, dict]) -> dict | None:
    """Share of examples whose verdict differs between two models (only meaningful for 0/1 scores)."""
    keys = sorted(set(base) & set(other))
    if not keys or any(row["score"] not in (0.0, 1.0) for row in list(base.values()) + list(other.values())):
        return None
    worse = sum(1 for k in keys if base[k]["score"] == 1.0 and other[k]["score"] == 0.0)
    better = sum(1 for k in keys if base[k]["score"] == 0.0 and other[k]["score"] == 1.0)
    return {"n": len(keys), "worse": worse, "better": better, "share": round((worse + better) / len(keys), 4)}
