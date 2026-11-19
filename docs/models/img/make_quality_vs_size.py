"""Quality against file size for the model candidates (our q4_k_m files, GPU fast set), small multiples."""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from koffient_tooling.model import sha256_file
from koffient_tooling.quality import record as rec
from koffient_tooling.quality.run import load_answers, score_block, split_answers
from koffient_tooling.quality.tasks import TASKS

res = pathlib.Path(r"C:\koffient-work\results")
M = r"C:\koffient-work\models"
CANDIDATES = [
    ("Qwen2.5-0.5B", rf"{M}\qwen2.5-0.5b-instruct\7ae5576\qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Qwen3-0.6B", rf"{M}\qwen3-0.6b\c1899de\qwen3-0.6b-tied-q4_k_m-tied.gguf"),
    ("Qwen3.5-0.8B", rf"{M}\qwen3.5-0.8b\2fc0636\qwen3.5-0.8b-q4_k_m.gguf"),
    ("LFM2-1.2B", rf"{M}\lfm2-1.2b\40f3da0\lfm2-1.2b-q4_k_m.gguf"),
    ("Vikhr-Llama-1B", rf"{M}\vikhr-llama-3.2-1b-instruct\7fa9d06\vikhr-llama-3.2-1b-instruct-q4_k_m.gguf"),
    ("Granite-4.0-1B", rf"{M}\granite-4.0-1b\6a7381b\granite-4.0-1b-q4_k_m.gguf"),
    ("SmallThinker-4B-A0.6B", rf"{M}\smallthinker-4ba0.6b-instruct\a24932b\smallthinker-4ba0.6b-instruct-q4_k_m.gguf"),
]
PANELS = [("Вызов функций (англ.)", "bfcl-en"), ("JSON по схеме", "jsonschema"), ("Инструкции (англ.)", "ifeval-en"),
          ("Вызов функций (рус.)", "bfcl-ru")]
CHOSEN = {"Granite-4.0-1B": "#2a78d6", "Qwen3-0.6B": "#eb6834"}
OTHER = "#898781"
# Label offsets (points) where neighbours collide.
OFFSETS = {("Qwen2.5-0.5B", "bfcl-en"): (6, -12), ("LFM2-1.2B", "bfcl-en"): (6, 4),
           ("Qwen3.5-0.8B", "ifeval-en"): (6, -12), ("LFM2-1.2B", "ifeval-en"): (-58, 6),
           ("Qwen2.5-0.5B", "bfcl-ru"): (6, -12), ("Qwen2.5-0.5B", "jsonschema"): (6, -14),
           ("Vikhr-Llama-1B", "jsonschema"): (6, 4), ("LFM2-1.2B", "jsonschema"): (6, -12)}

latest = rec.latest_ok(rec.load(res), "task")
points = []
for name, path in CANDIDATES:
    p = pathlib.Path(path)
    sha = sha256_file(p)
    row = {"name": name, "mb": p.stat().st_size / 1e6}
    for _, task in PANELS:
        r = latest[(f"{sha}|cuda", task)]
        head, _ = split_answers(TASKS[task], load_answers(res / r["raw_dir"]))
        row[task] = score_block(list(head.values()))
    points.append(row)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.4), sharey=True)
for ax, (title, task) in zip(axes, PANELS):
    for row in points:
        color = CHOSEN.get(row["name"], OTHER)
        lo, hi = row[task]["ci95"]
        v = row[task]["value"]
        ax.errorbar(row["mb"], v, yerr=[[v - lo], [hi - v]], fmt="o", color=color, ms=7 if row["name"] in CHOSEN else 5,
                    ecolor=color, elinewidth=1, capsize=0)
        dx, dy = OFFSETS.get((row["name"], task), (6, 4))
        ax.annotate(row["name"], (row["mb"], v), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    color="#0b0b0b" if row["name"] in CHOSEN else "#52514e")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([400, 700, 1000, 2600])
    ax.set_xticklabels(["400", "700", "1000", "2600"])
    ax.set_xlim(330, 4200)
    ax.set_xlabel("размер файла q4_k_m, МБ (лог. шкала)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e1e0d9")
axes[0].set_ylabel("доля верных ответов (95% интервал)")
fig.suptitle("Качество против размера: кандидаты для MVP (наши файлы q4_k_m, быстрый набор на видеокарте; "
             "вызов функций — без группы «вызов не нужен»)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(out)
