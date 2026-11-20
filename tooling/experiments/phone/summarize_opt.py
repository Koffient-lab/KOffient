"""Summary of the phone optimisation experiments (phone-opt.sh, phone-cache.sh, phone-threads.sh).

usage: python -I summarize_opt.py <phone-opt out dir>
"""
import json
import pathlib
import re
import sys

W = pathlib.Path(r"C:\koffient-work")
opt_dir = pathlib.Path(sys.argv[1])


def bench(path: pathlib.Path) -> list[str]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"  unreadable: {exc}"]
    out = []
    for r in rows:
        kind = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        knobs = " ".join(f"{k}={r[k]}" for k in ("n_threads", "n_ubatch", "flash_attn", "poll", "cpu_strict", "cpu_mask")
                         if k in r)
        out.append(f"  {kind:6s} {r['avg_ts']:7.1f} t/s  (±{r['stddev_ts']:.1f})  {knobs}")
    return out


def completion(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    load = re.search(r"load time =\s+([\d.]+) ms", text)
    prompt = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", text)
    match = re.search(r"session file matches (\d+) / (\d+)", text)
    sess = re.search(r"(\d+)\.(\d+)\.(\d+)\.\d+ I llama_completion: attempting to load saved session.*?\n"
                     r"(\d+)\.(\d+)\.(\d+)\.\d+ I llama_completion: (?:loaded a session|session file does not)", text)
    session_ms = None
    if sess:
        t = [int(x) for x in sess.groups()]
        session_ms = ((t[3] * 60 + t[4]) * 1000 + t[5]) - ((t[0] * 60 + t[1]) * 1000 + t[2])
    return (f"load {load.group(1) if load else '?'} ms, prompt {prompt.group(1) if prompt else '?'} ms / "
            f"{prompt.group(2) if prompt else '?'} tok, session load {session_ms} ms, "
            f"matched {match.group(1) + '/' + match.group(2) if match else '-'}")


for tag in ("g-q4km", "q-q4km", "g-q40", "g-q4km-2"):
    print(f"===== {tag}")
    for part in ("scale", "knobs", "poll", "strict", "prio"):
        f = opt_dir / f"{tag}-{part}.json"
        if f.exists():
            print(f" {part}:")
            print("\n".join(bench(f)))
    for state in sorted(opt_dir.glob(f"{tag}-*.state")):
        print(f" {state.stem}: " + " | ".join(state.read_text(encoding="utf-8").split("\n")).strip(" |"))
    for name in ("build", "cached1", "cached2", "cached3", "nocache1", "nocache2", "nocache3"):
        f = opt_dir / f"{tag}-{name}.err"
        if f.exists():
            print(f" {name:9s} {completion(f)}")

print("===== memory runs (runs.jsonl)")
invalid = {json.loads(l).get("run_id") for l in open(W / "results" / "invalid.jsonl", encoding="utf-8")}
for line in open(W / "results" / "runs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    label = r.get("label") or ""
    if not (label.startswith("phone-mem") or label in ("phone-ctx4096", "phone-tools-r4")) or r["status"] != "ok":
        continue
    m = r["metrics"]
    flag = " INVALID" if r["run_id"] in invalid else ""
    print(f"  {label:22s} {r['model']['file'][:40]:40s} kv={r['params'].get('kv_type')} ub={r['params'].get('ubatch')} "
          f"ctx={r['params'].get('n_ctx')} rss={m.get('peak_rss_mb')} MB tg={m['decode'][0]['tps']['median']:.1f} "
          f"pp1024={m['prefill'][-1]['tps']['median']:.1f} ttft={m['ttft_ms']['warm']['median'] / 1000:.1f}s{flag}")

cache = W / "results" / "raw" / "phone-cache" / "cache.jsonl"
if cache.exists():
    print("===== resident model, prompt cache (llama-server)")
    for line in open(cache, encoding="utf-8"):
        print("  " + line.strip())
threads = W / "results" / "raw" / "phone-threads" / "threads.json"
if threads.exists():
    print("===== threads with little cores")
    print("\n".join(bench(threads)))
