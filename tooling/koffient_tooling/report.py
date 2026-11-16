"""Static charts and tables built from runs.jsonl and quality.jsonl.

A report is a frozen folder: PNG charts, the numbers behind them (report.md, data.json) and the ids of the
records they were built from, so that a decision in the journal can point at it and it can be rebuilt.
Chart labels are in Russian: the charts go into the project journal and the defence slides.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import record as runs
from .quality import record as quality
from .quality.cli import _label, comparison
from .quality.run import MIN_HEADROOM
from .quality.tasks import TASKS

# Palette: validated categorical slots 1-3 and chart chrome for the light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

SCENARIOS = {
    "function_calling": "Вызов функций",
    "json_schema": "JSON по схеме",
    "instructions": "Следование инструкциям",
    "summarization": "Суммаризация переписки",
    "math": "Математика",
    "knowledge": "Знания",
}
ROLES = {"primary": "основной", "secondary": "вторичный", "control": "контрольный"}
METRICS = {
    "function_calling": "доля верных вызовов", "json_schema": "доля валидных ответов",
    "instructions": "доля выполненных заданий", "summarization": "ROUGE-L",
    "math": "доля верных ответов", "knowledge": "доля верных ответов",
}
LANGS = {"en": "английский", "ru": "русский"}
GROUPS = {
    "simple": "Одна функция", "multiple": "Выбор из нескольких функций", "parallel": "Несколько вызовов сразу",
    "irrelevance": "Вызов не нужен",
}
FLAT_NOTE = "не различает варианты этой модели"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": AXIS,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "ytick.left": False, "legend.frameon": False,
    })
    return plt


def _variant_ticks(ax, variants: list[dict]) -> None:
    """Variants sit at even steps in order of file size; the size itself is in the tick label."""
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{v['model'].get('quant_label') or '?'}\n{v['model']['file_bytes'] / 1e6:.0f} МБ"
                        for v in variants])
    ax.set_xlim(-0.5, len(variants) - 0.5)


def chart_quality_vs_size(data: dict, path: Path) -> list[dict]:
    """Small multiples: one panel per scenario, score against file size, one line per language."""
    plt = _pyplot()
    variants = sorted((v for v in data["variants"] if not v["label"]), key=lambda v: v["model"]["file_bytes"])
    position = {v["key"]: i for i, v in enumerate(variants)}
    scenarios = [s for s in SCENARIOS if any(TASKS[t].scenario == s for t in data["tasks"])]
    if not variants or not scenarios:
        return []
    columns = 3
    rows_n = -(-len(scenarios) // columns)
    fig, axes = plt.subplots(rows_n, columns, figsize=(4.3 * columns, 3.4 * rows_n), squeeze=False)
    table = []
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    for ax, scenario in zip(axes.flat, scenarios):
        top = 0.0
        flat_langs: list[str] = []
        names = [t for t in data["tasks"] if TASKS[t].scenario == scenario]
        for name in sorted(names, key=lambda t: list(LANGS).index(TASKS[t].lang)):
            task = TASKS[name]
            color = SERIES[list(LANGS).index(task.lang)]
            points = [(v, data["tasks"][name].get(v["key"])) for v in variants]
            points = [(v, cell) for v, cell in points if cell]
            # Two languages share the same file sizes: shift them apart slightly so the intervals stay readable.
            shift = (list(LANGS).index(task.lang) - 0.5) * 0.08 if len(names) > 1 else 0.0
            xs = [position[v["key"]] + shift for v, _ in points]
            ys = [cell["value"] for _, cell in points]
            lows = [cell["value"] - cell["ci95"][0] for _, cell in points]
            highs = [cell["ci95"][1] - cell["value"] for _, cell in points]
            label = LANGS[task.lang] if len(names) > 1 else None
            ax.errorbar(xs, ys, yerr=[lows, highs], color=color, linewidth=2, marker="o", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1, capsize=3, label=label,
                        solid_capstyle="round", solid_joinstyle="round")
            top = max([top] + [cell["ci95"][1] for _, cell in points])
            flat = name in data["flat"]
            if flat and points:
                flat_langs.append(LANGS[task.lang])
            for v, cell in points:
                table.append({"task": name, "scenario": scenario, "lang": task.lang, "variant": _label(v),
                              "file_mb": round(v["model"]["file_bytes"] / 1e6), "value": cell["value"],
                              "ci95": cell["ci95"], "n": cell["n"], "truncated": cell["truncated"],
                              "note": FLAT_NOTE if flat else ""})
        role = ROLES[TASKS[names[0]].role]
        ax.set_title(f"{SCENARIOS[scenario]} ({role})")
        ax.set_ylabel(METRICS[scenario])
        if flat_langs:
            # Under the axis, not next to the line: inside the panel the note collides with the other language.
            who = f"{', '.join(flat_langs)}: " if len(names) > 1 else ""
            ax.set_xlabel(f"{who}{FLAT_NOTE}", color=MUTED, fontsize=9)
        ax.set_ylim(0, min(1.0, top * 1.15) if top else 1.0)
        _variant_ticks(ax, variants)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 0.995))
    name = variants[0]["model"]["name"]
    fig.suptitle(f"Качество против размера файла: {name}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, "Отрезки — 95% доверительный интервал. Подвыборки быстрого набора. "
                          "Вызов функций — без группы «вызов не нужен».", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def chart_call_groups(data: dict, path: Path) -> list[dict]:
    """Function calling by group of examples: one panel per group, one line per language."""
    plt = _pyplot()
    variants = sorted((v for v in data["variants"] if not v["label"]), key=lambda v: v["model"]["file_bytes"])
    position = {v["key"]: i for i, v in enumerate(variants)}
    names = [t for t in TASKS if any(cell["groups"] for cell in data["tasks"].get(t, {}).values())]
    groups = [g for g in GROUPS if any(g in cell["groups"] for t in names for cell in data["tasks"][t].values())]
    if not variants or not groups:
        return []
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.7), squeeze=False, sharey=True)
    table = []
    for ax, group in zip(axes.flat, groups):
        for name in sorted(names, key=lambda t: list(LANGS).index(TASKS[t].lang)):
            lang = TASKS[name].lang
            points = [(v, (data["tasks"][name].get(v["key"]) or {"groups": {}})["groups"].get(group))
                      for v in variants]
            points = [(v, cell) for v, cell in points if cell]
            shift = (list(LANGS).index(lang) - 0.5) * 0.08 if len(names) > 1 else 0.0
            ax.errorbar([position[v["key"]] + shift for v, _ in points], [cell["value"] for _, cell in points],
                        yerr=[[cell["value"] - cell["ci95"][0] for _, cell in points],
                              [cell["ci95"][1] - cell["value"] for _, cell in points]],
                        color=SERIES[list(LANGS).index(lang)], linewidth=2, marker="o", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1, capsize=3, label=LANGS[lang],
                        solid_capstyle="round", solid_joinstyle="round")
            for v, cell in points:
                table.append({"task": name, "group": group, "variant": _label(v), "value": cell["value"],
                              "ci95": cell["ci95"], "n": cell["n"],
                              "in_score": "нет" if cell["side"] else "да"})
        ax.set_title(GROUPS[group])
        ax.set_ylim(0, 1)
        _variant_ticks(ax, variants)
    axes.flat[0].set_ylabel("доля верных ответов")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"Вызов функций по группам примеров: {variants[0]['model']['name']}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, "Отрезки — 95% доверительный интервал. В группе «вызов не нужен» верный ответ — "
                          "не вызывать функцию; в общую оценку она не входит.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def chart_closeness(data: dict, path: Path) -> list[dict]:
    plt = _pyplot()
    variants = sorted((v for v in data["variants"] if not v["label"]), key=lambda v: v["model"]["file_bytes"])
    position = {v["key"]: i for i, v in enumerate(variants)}
    if not data["closeness"] or not variants:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    panels = [("kld_mean", "KL-расхождение с fp16 (меньше — ближе)", 0.0),
              ("same_top_pct", "Совпадение первого токена с fp16, %", 100.0)]
    table = []
    for ax, (field, title, base_value) in zip(axes, panels):
        for lang, cells in sorted(data["closeness"].items()):
            points = [(v, cells.get(v["key"])) for v in variants]
            points = [(v, m) for v, m in points if m]
            xs = [position[v["key"]] for v, _ in points]
            ys = [m.get(field, base_value) for _, m in points]
            ax.plot(xs, ys, color=SERIES[list(LANGS).index(lang)], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=LANGS[lang], solid_capstyle="round")
            if field == "kld_mean":
                for v, m in points:
                    table.append({"lang": lang, "variant": _label(v), "ppl": m["ppl"],
                                  "kld_mean": m.get("kld_mean"), "same_top_pct": m.get("same_top_pct")})
        ax.set_title(title)
        _variant_ticks(ax, variants)
        if field == "kld_mean":
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(top=100)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def chart_changed_answers(data: dict, path: Path) -> list[dict]:
    """Share of answers whose verdict differs from the base model, per task and variant."""
    plt = _pyplot()
    others = [v for v in data["variants"] if v["key"] != data["base"]["key"]][:len(SERIES)]
    names = [t for t in TASKS if any((data["tasks"].get(t, {}).get(v["key"]) or {}).get("changed") for v in others)]
    if not names:
        return []
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) * max(1, len(others)) + 1.6))
    height = min(0.8 / len(others), 0.36)
    table = []
    for index, variant in enumerate(others):
        for row, name in enumerate(names):
            changed = (data["tasks"][name].get(variant["key"]) or {}).get("changed")
            if not changed:
                continue
            y = row + (index - (len(others) - 1) / 2) * height
            share = changed["share"] * 100
            ax.barh(y, share, height=height * 0.85, color=SERIES[index],
                    label=_label(variant) if row == 0 else None)
            ax.text(share + 0.4, y, f"{share:.0f}%  (хуже {changed['worse']}, лучше {changed['better']})",
                    va="center", color=INK_2, fontsize=9)
            table.append({"task": name, "variant": _label(variant), **changed})
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("доля примеров, %")
    ax.set_xlim(0, max(r["share"] for r in table) * 100 * 1.45 + 2)
    ax.set_title(f"Ответы, изменившиеся относительно {_label(data['base'])}")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if len(unique) > 1:
        ax.legend(unique.values(), unique.keys(), loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def chart_speed(records: list[dict], path: Path) -> list[dict]:
    """One panel per speed metric, one bar per engine build (latest successful run of each)."""
    plt = _pyplot()
    latest = {}
    for r in records:
        if r["status"] == "ok":
            latest[(r["model"]["sha256"], r["engine"]["build_id"], r["device"].get("serial"))] = r
    if not latest:
        return []
    chosen = list(latest.values())

    def stat(block):
        return (block["median"], block["min"], block["max"]) if block else (None, None, None)

    panels = [
        ("Обработка промпта, ток/с", lambda m: stat(m["prefill"][0]["tps"])),
        ("Генерация, ток/с", lambda m: stat(m["decode"][0]["tps"])),
        ("Время до первого токена, мс", lambda m: stat(m["ttft_ms"]["warm"])),
        ("Пик памяти, МБ", lambda m: (m["peak_rss_mb"], None, None)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    table = []
    labels = [f"{r['engine']['build_id']}\n{r['model'].get('quant_label') or ''}" for r in chosen]
    for ax, (title, pick) in zip(axes, panels):
        values = [pick(r["metrics"]) for r in chosen]
        for x, (median, low, high) in enumerate(values):
            ax.bar(x, median, width=0.45, color=SERIES[0])
            if low is not None:
                ax.plot([x, x], [low, high], color=INK_2, linewidth=1)
            ax.text(x, max(median, high or median), f" {median:.0f}", ha="center", va="bottom", color=INK_2)
        ax.set_xticks(range(len(chosen)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(max(v[0], v[2] or 0) for v in values) * 1.18)
        ax.set_xlim(-0.6, len(chosen) - 0.4)
        table.append({"metric": title, "values": {r["run_id"]: v[0] for r, v in zip(chosen, values)}})
    kinds = sorted({r["device"]["kind"] for r in chosen})
    note = "Замеры на эмуляторе x86_64: годятся только для сравнения сборок между собой." \
        if kinds == ["emulator"] else f"Устройства: {', '.join(kinds)}."
    fig.text(0.01, 0.005, note + " Отрезки — минимум и максимум по повторам.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return table


def _md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_нет данных_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def build(results_dir: Path, out_dir: Path, base_hint: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_records = runs.load(results_dir)
    quality_records = quality.load(results_dir)
    data = comparison(quality_records, results_dir, base_hint)
    tables = {
        "quality_vs_size": chart_quality_vs_size(data, out_dir / "quality_vs_size.png") if data["variants"] else [],
        "call_groups": chart_call_groups(data, out_dir / "function_calls_by_group.png") if data["variants"] else [],
        "closeness": chart_closeness(data, out_dir / "closeness.png") if data["variants"] else [],
        "changed_answers": chart_changed_answers(data, out_dir / "changed_answers.png") if data["variants"] else [],
        "speed": chart_speed(run_records, out_dir / "speed.png"),
    }
    sources = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tooling_commit": runs.tooling_commit(),
        "base": _label(data["base"]) if data["base"] else None,
        "run_ids": [r["run_id"] for r in run_records if r["status"] == "ok"],
        "eval_ids": sorted(r["eval_id"] for r in list(quality.latest_ok(quality_records, "task").values())
                           + list(quality.latest_ok(quality_records, "closeness").values())),
    }
    (out_dir / "data.json").write_text(json.dumps({"sources": sources, "tables": tables}, ensure_ascii=False,
                                                  indent=1), encoding="utf-8", newline="\n")
    parts = [
        f"# Отчёт от {sources['built_at'][:10]}\n",
        f"Построен из `runs.jsonl` и `quality.jsonl`, код `{sources['tooling_commit']}`. "
        f"Модель для сравнения: {sources['base'] or 'нет'}. Идентификаторы записей — в `data.json`.\n",
        "## Качество против размера\n\n![](quality_vs_size.png)\n",
        "`truncated` — сколько ответов оборвано по лимиту токенов. Оценка `bfcl` — без группы «вызов не нужен». "
        f"Пометка «{FLAT_NOTE}» стоит там, где модель для сравнения выше уровня угадывания меньше чем на "
        f"{MIN_HEADROOM:.2f}: порог «не хуже fp16» к такой задаче не применяется.\n",
        _md_table(tables["quality_vs_size"], ["task", "variant", "file_mb", "value", "ci95", "n", "truncated",
                                              "note"]),
        "## Вызов функций по группам примеров\n\n![](function_calls_by_group.png)\n",
        _md_table(tables["call_groups"], ["task", "group", "variant", "value", "ci95", "n", "in_score"]),
        "## Близость к fp16\n\n![](closeness.png)\n",
        _md_table(tables["closeness"], ["lang", "variant", "ppl", "kld_mean", "same_top_pct"]),
        "## Изменившиеся ответы\n\n![](changed_answers.png)\n",
        _md_table(tables["changed_answers"], ["task", "variant", "n", "worse", "better", "share"]),
        "## Скорость и память\n\n![](speed.png)\n",
        _md_table([{"metric": row["metric"], **row["values"]} for row in tables["speed"]],
                  ["metric"] + (list(tables["speed"][0]["values"]) if tables["speed"] else [])),
    ]
    (out_dir / "report.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return out_dir
