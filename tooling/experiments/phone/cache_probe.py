"""Prompt-cache probe against llama-server running on the phone (model resident in memory, as the SDK will hold it).

Sends the six-tool chat request through the model's own chat template (server --jinja), max 1 token, and records the
server timings: 1) cold request; 2) requests with the same prefix (system + tools) and another question, alternating,
so the slot reuses the computed prefix; 3) save the slot state to a file, erase the slot, restore it, ask again
(app restart with the cache on disk). Writes one JSON line per step.

usage: python -I cache_probe.py <base url> <out .jsonl> <tag>
"""
import json
import pathlib
import sys
import time
import urllib.request

base, out, tag = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
src = pathlib.Path(__file__).with_name("tools_q1.txt").read_text(encoding="utf-8")
TOOLS = [{"type": "function", "function": t} for t in json.loads(src[src.index("["):src.rindex("]") + 1])]
SYSTEM = ("You are an assistant inside a mobile app. Call a function only when it is needed; otherwise answer in plain "
          "text. Today is Thursday, 12 March 2026, and the user is in Berlin.")
QUESTIONS = {
    "q1": src[src.rindex("User: ") + 6:src.rindex("\n\nAssistant:")],
    "q2": "How much is 250 US dollars in euros right now?",
    "q3": "Remind me to call my mother tomorrow at 7 pm.",
}


def post(path: str, body: dict | None = None, timeout: float = 600) -> tuple[dict, float]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    reply = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return reply, (time.perf_counter() - t0) * 1000


def ask(q: str) -> dict:
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTIONS[q]}],
            "tools": TOOLS, "max_tokens": 1, "temperature": 0, "seed": 1, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    reply, wall = post("/v1/chat/completions", body)
    t = reply.get("timings", {})
    return {"question": q, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"), "wall_ms": round(wall, 1)}


def log(step: str, row: dict) -> None:
    row = {"tag": tag, "step": step, **row}
    with open(out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


for i in range(120):
    try:
        if json.loads(urllib.request.urlopen(base + "/health", timeout=5).read()).get("status") == "ok":
            break
    except Exception:  # noqa: BLE001 - server still loading
        pass
    time.sleep(2)

log("cold", ask("q1"))
for q in ["q2", "q3", "q1", "q2", "q3"]:
    log("warm", ask(q))
saved, wall = post("/slots/0?action=save", {"filename": f"{tag}.bin"})
log("save", {"n_saved": saved.get("n_saved"), "save_ms": saved.get("timings", {}).get("save_ms"), "wall_ms": round(wall, 1)})
for k in range(3):
    post("/slots/0?action=erase")
    restored, wall = post("/slots/0?action=restore", {"filename": f"{tag}.bin"})
    log("restore", {"n_restored": restored.get("n_restored"),
                    "restore_ms": restored.get("timings", {}).get("restore_ms"), "wall_ms": round(wall, 1)})
    log("after-restore", ask(["q2", "q3", "q2"][k]))
