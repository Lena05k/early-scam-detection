#!/usr/bin/env python3
"""Графики слоя 4. Числа берутся из runs/layer4/results.json (+ results_control.json,
runs/layer0/results.json, runs/layer2/results.json и снимков xlm-r) — ничего не досочиняется.

  python scripts/make_layer4_plots.py
"""
import argparse
import csv
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

DPI = 150
LAYER_COLOR = {"слой 0": "#937860", "слой 1": "#DD8452", "слой 2": "#55A868",
               "слой 4": "#8172B3", "файнтюн": "#4C72B0"}
CONTROL_COLOR = "#C44E52"        # контроль цикличности — отдельный цвет, не «слой»
MODE_RU = {"zero": "zero-shot", "few": "few-shot (6 примеров)",
           "few_rules": "few-shot + правила"}
GOLD_V2 = "data/gold/gold_test_v2.csv"


def _load(path, default=None):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def prompt_modes(p, path):
    """F1 трёх режимов: слева отбор на dev, справа замер на тесте."""
    modes = p["modes"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(modes))
    w = 0.36
    for k, (split, col, lab) in enumerate(
            (("dev", "#4C72B0", "gold-dev (n=100) — по нему выбран режим"),
             ("test", "#8172B3", "gold-test v2 (n=303) — замер"))):
        # на тесте может быть прогнан только выбранный режим (--test-best-only)
        present = [m for m in modes if m in p[split]]
        xs = np.array([modes.index(m) for m in present], dtype=float)
        mu = [p[split][m]["f1"]["mean"] for m in present]
        sd = [p[split][m]["f1"]["std"] for m in present]
        ax.bar(xs + (k - 0.5) * w, mu, w, yerr=sd, capsize=4, color=col, label=lab,
               edgecolor="white")
        for xi, m_, s_ in zip(xs + (k - 0.5) * w, mu, sd):
            ax.text(xi, m_ + s_ + 0.012, f"{m_:.3f}", ha="center", fontsize=9)
    best = p["best_mode_by_gold_dev"]
    ax.axvspan(modes.index(best) - 0.5, modes.index(best) + 0.5, color="#FFF3C4",
               zorder=0, alpha=0.7)
    ax.set_xticks(x, [MODE_RU.get(m, m) for m in modes], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 по is_scam")
    ax.set_title(f"Слой 4: режимы промпта, {p['model']} (temperature=0, "
                 f"{p['n_runs']} прогона)\nжёлтым — режим, выбранный по gold-dev", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.012, "Доля разобранных ответов: dev — " +
             ", ".join(f"{MODE_RU.get(m, m)} {p['dev'][m]['parse_rate']['mean']:.3f}"
                       for m in modes) + "; тест — " +
             ", ".join(f"{MODE_RU.get(m, m)} {p['test'][m]['parse_rate']['mean']:.3f}"
                       for m in modes if m in p["test"]),
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def ladder_final(p, l0, l2, path, is_control=False):
    """Полная лесенка: слои 0, 1, 2, 4 и файнтюн.

    is_control=True — прогнан не бейзлайн слоя 4, а контроль цикличности
    (DeepSeek, автор корпуса и silver-меток). Он показывается ОТДЕЛЬНЫМ цветом
    со штриховкой и подписью «НЕ бейзлайн»: его F1 измеряет согласие модели с
    собственной разметкой, а не качество детекции, и в ряд слоёв не встаёт.
    """
    rows = [{"model": l0["predictors"][k]["name_ru"],
             "f1": l0["predictors"][k]["metrics"]["f1"], "layer": "слой 0"}
            for k in ("const_scam", "const_benign", "length", "url", "keywords", "logreg3")]
    rows.append({"model": "TF-IDF + LogReg", "f1": 0.8638, "layer": "слой 1"})
    if l2:
        be, ba = l2["best_config"].split("|")
        rows.append({"model": f"эмбеддинги {be} + LightGBM",
                     "f1": l2["summary"]["f1"]["mean"],
                     "std": l2["summary"]["f1"]["std"], "layer": "слой 2"})
    if is_control:
        rows.append({"model": f"контроль: {p['model']} — автор корпуса, НЕ бейзлайн",
                     "f1": p["summary"]["f1"]["mean"], "std": p["summary"]["f1"]["std"],
                     "layer": "контроль"})
    else:
        rows.append({"model": f"LLM {p['model']} ({MODE_RU.get(p['best_mode_by_gold_dev'])})",
                     "f1": p["summary"]["f1"]["mean"], "std": p["summary"]["f1"]["std"],
                     "layer": "слой 4"})
    rows += [{"model": "mDeBERTa-v3-base", "f1": 0.8942, "layer": "файнтюн"},
             {"model": "xlm-roberta-base", "f1": 0.9344, "layer": "файнтюн"}]
    rows.sort(key=lambda d: d["f1"])

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r["f1"] for r in rows], xerr=[r.get("std", 0.0) for r in rows],
                   capsize=3, edgecolor="white", height=0.72,
                   color=[CONTROL_COLOR if r["layer"] == "контроль"
                          else LAYER_COLOR[r["layer"]] for r in rows])
    for b, r in zip(bars, rows):
        if r["layer"] == "контроль":
            b.set_hatch("//")
            b.set_edgecolor("white")
    ax.set_yticks(y, [r["model"] for r in rows], fontsize=9.5)
    for lab_, r in zip(ax.get_yticklabels(), rows):
        if r["layer"] == "контроль":
            lab_.set_color(CONTROL_COLOR)
    for yi, r in zip(y, rows):
        lab = f"{r['f1']:.4f}" + (f" ± {r['std']:.4f}" if r.get("std") else "")
        ax.text(r["f1"] + r.get("std", 0.0) + 0.01, yi, lab, va="center", fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("F1 на gold-test (n=303)")
    title = "Полная лесенка is_scam: от константы до файнтюна"
    if is_control:
        title += "\n⚠ слот «слой 4» пуст: основной LLM-бейзлайн не прогнан"
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    keys = ["слой 0", "слой 1", "слой 2"] + ([] if is_control else ["слой 4"]) + ["файнтюн"]
    names = {"слой 0": "слой 0: тривиальные", "слой 1": "слой 1: TF-IDF",
             "слой 2": "слой 2: эмбеддинги + бустинг", "слой 4": "слой 4: LLM без обучения",
             "файнтюн": "файнтюн трансформера"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLOR[k]) for k in keys]
    labels = [names[k] for k in keys]
    if is_control:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=CONTROL_COLOR, hatch="//"))
        labels.append("контроль цикличности (не участвует в лесенке)")
    ax.legend(handles, labels, loc="lower right", fontsize=9)
    note = ("TF-IDF и mDeBERTa — опубликованные числа на gold v1, остальные на v2 "
            "(различие 3 метки из 303).\nУсы там, где считался разброс: слой 2 (сиды "
            "LightGBM), LLM (повторные прогоны при temperature=0).")
    if is_control:
        note += ("\n⚠ Красная штриховка: deepseek-chat породил корпус и silver-метки. "
                 "Его F1 измеряет согласие с собственной разметкой (0.990), а не качество "
                 "детекции — сравнивать его с файнтюном НЕЛЬЗЯ.")
    fig.text(0.01, 0.012, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.09 if is_control else 0.06, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def silver_final(p, l0, l2, ctrl, path):
    """Кто копирует разметку DeepSeek — с LLM и контрольным прогоном DeepSeek."""
    items = {
        "слой 0: ключевые слова": (l0["diagnostics"]["silver_agreement_on_test"]["keywords"],
                                   "слой 0"),
        "слой 0: логрегрессия-3": (l0["diagnostics"]["silver_agreement_on_test"]["logreg3"],
                                   "слой 0"),
        "слой 1: TF-IDF": (0.9868, "слой 1"),
        "файнтюн: mDeBERTa": (0.9395, "файнтюн"),
        "файнтюн: xlm-r": (0.8372, "файнтюн"),
    }
    if l2:
        items[f"слой 2: {l2['best_config'].split('|')[0]}"] = (
            l2["summary"]["silver_agreement"]["mean"], "слой 2")
    if p is not None:          # основной прогон слоя 4 (не контроль)
        items[f"слой 4: {p['model']}"] = (p["summary"]["silver_agreement"]["mean"], "слой 4")
    if ctrl:
        items[f"контроль: {ctrl['model']} (автор корпуса)"] = (
            ctrl["summary"]["silver_agreement"]["mean"], "слой 4")

    order = sorted(items.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(order))
    ax.barh(y, [v for _, (v, _) in order],
            color=[LAYER_COLOR[lay] for _, (_, lay) in order],
            edgecolor="white", height=0.68)
    ax.set_yticks(y, [k for k, _ in order], fontsize=10)
    for yi, (_, (v, _)) in zip(y, order):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=10)
    ref = l0["diagnostics"]["silver_vs_gold_on_test"]
    ax.axvline(ref, color="#C44E52", ls="--", lw=1.6)
    ax.text(ref - 0.012, len(order) - 0.5, f"сам silver\nпротив gold {ref:.3f}",
            color="#C44E52", fontsize=9, va="top", ha="right")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlabel("доля диалогов gold-test, где предсказание совпало с silver-меткой DeepSeek")
    ax.set_title("Кто воспроизводит черновую разметку генератора", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.012, "Выше — ближе к копии DeepSeek. Контрольный прогон самого DeepSeek "
                          "показывает потолок цикличности: модель сверяется сама с собой.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _xlmr_predictions():
    """Предсказания xlm-r на 303: выбранная эпоха 1, голосование по трём сидам."""
    files = sorted(glob.glob("runs/epoch_cost/data/xlmr/gold_test_per_epoch_seed*.json"))
    if not files:
        return None, None
    votes, ids = [], None
    for f in files:
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        i = d["epochs"].index(1)                     # выбранная эпоха у xlm-r везде 1
        votes.append(np.array(d["per_epoch"][i]["y_pred"]))
        ids = d["dialog_ids"]
    return ids, (np.mean(votes, axis=0) >= 0.5).astype(int)


def llm_vs_xlmr(p, path, is_control=False):
    """Где ошибается LLM, где xlm-r, где оба — 2×2 плюс состав спорной зоны."""
    ids, xp = _xlmr_predictions()
    if ids is None:
        print("  ⚠ снимков xlm-r нет — llm_vs_xlmr_errors.png пропущен")
        return
    gold = {}
    with open(GOLD_V2, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gold[row["dialog_id"]] = int(row["is_scam"])
    # hard_negative берётся из КОРПУСА: в gold-CSV эта колонка сплошь нули —
    # разметчик её не проставил ни разу (403/403), корпусные значения сохранены
    hardneg = {}
    with open("data/corpus/corpus_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                hardneg[r["id"]] = int(r.get("hard_negative", 0))

    best = p["best_mode_by_gold_dev"]
    tag = p["tag"]
    per_dialog = {}
    for k in range(1, p["n_runs"] + 1):
        cf_ = pathlib.Path(f"runs/layer4/cache/{tag}_{best}_test_run{k}.jsonl")
        for line in cf_.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["parsed"] is not None:
                per_dialog.setdefault(r["dialog_id"], []).append(r["parsed"]["is_scam"])
    common = [i for i in ids if i in per_dialog and i in gold]
    llm = np.array([1 if np.mean(per_dialog[i]) >= 0.5 else 0 for i in common])
    xlm = np.array([xp[ids.index(i)] for i in common])
    y = np.array([gold[i] for i in common])
    hn = np.array([hardneg[i] for i in common])

    llm_ok, xlm_ok = (llm == y), (xlm == y)
    cm = np.array([[int((xlm_ok & llm_ok).sum()), int((xlm_ok & ~llm_ok).sum())],
                   [int((~xlm_ok & llm_ok).sum()), int((~xlm_ok & ~llm_ok).sum())]])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.imshow(cm, cmap="Purples", vmin=0, vmax=cm.max())
    labels = [["оба правы", "прав только xlm-r"], ["прав только LLM", "ошиблись оба"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=10.5, color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xticks([0, 1], ["LLM прав", "LLM ошибся"])
    ax.set_yticks([0, 1], ["xlm-r прав", "xlm-r ошибся"])
    ax.set_title(f"Согласие по ошибкам, n={len(common)}", fontsize=11)

    ax = axes[1]
    groups = {"оба правы": (xlm_ok & llm_ok), "прав только xlm-r": (xlm_ok & ~llm_ok),
              "прав только LLM": (~xlm_ok & llm_ok), "ошиблись оба": (~xlm_ok & ~llm_ok)}
    names = list(groups)
    tot = [int(groups[g].sum()) for g in names]
    hn_share = [int((groups[g] & (hn == 1)).sum()) for g in names]
    x = np.arange(len(names))
    ax.bar(x, tot, 0.6, color="#B0A8C4", edgecolor="white", label="всего диалогов")
    ax.bar(x, hn_share, 0.6, color="#8172B3", edgecolor="white",
           label="из них hard-negative корпуса")
    for xi, (t_, h_) in enumerate(zip(tot, hn_share)):
        ax.text(xi, t_ + 2, f"{t_}", ha="center", fontsize=10)
        if h_:
            ax.text(xi, h_ / 2, f"{h_}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x, names, fontsize=9, rotation=12)
    ax.set_ylabel("диалогов")
    ax.set_title("Состав зон: сколько там hard-negatives", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    who = (f"контроль {p['model']} (автор корпуса)" if is_control
           else f"LLM {p['model']}")
    fig.suptitle(f"Где ошибается {who}, {MODE_RU.get(best)}, и где xlm-roberta-base",
                 fontsize=13)
    note = ("LLM — голосование по 3 прогонам, xlm-r — голосование по 3 сидам на выбранной "
            "эпохе 1 (снимки runs/epoch_cost).\n⚠ Пер-эпохные снимки xlm-r отличаются от "
            "финального predict на 1–3 пограничных диалога (fp16-шум).")
    if is_control:
        note += ("\n⚠ Это НЕ сравнение «LLM против файнтюна»: deepseek-chat породил корпус "
                 "и silver-метки, его ошибки — это ошибки самого корпуса.")
    fig.text(0.01, 0.012, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.09 if is_control else 0.06, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_language(p, path, is_control=False):
    langs = sorted(p["per_language"])
    mets = ["f1", "precision", "recall"]
    cols = {"f1": CONTROL_COLOR if is_control else "#8172B3",
            "precision": "#4C72B0", "recall": "#DD8452"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(langs))
    w = 0.26
    for k, m in enumerate(mets):
        mu = [p["per_language"][lg][m]["mean"] for lg in langs]
        sd = [p["per_language"][lg][m]["std"] for lg in langs]
        ax.bar(x + (k - 1) * w, mu, w, yerr=sd, capsize=4, color=cols[m], label=m,
               edgecolor="white")
        for xi, mm, ss in zip(x + (k - 1) * w, mu, sd):
            ax.text(xi, mm + ss + 0.014, f"{mm:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, langs, fontsize=11)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("значение на gold-test v2")
    head = ("Контроль цикличности по языкам" if is_control else "Слой 4 по языкам")
    ax.set_title(f"{head}: {p['model']}, {MODE_RU.get(p['best_mode_by_gold_dev'])}\n"
                 f"усы = std по {p['n_runs']} прогонам", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if is_control:
        fig.text(0.01, 0.012, "⚠ deepseek-chat — автор корпуса и silver-меток; это не "
                              "качество детекции, а воспроизводимость своей разметки.",
                 fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_plots(payload, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l0 = _load("runs/layer0/results.json")
    l2 = _load("runs/layer2/results.json")
    ctrl = _load("runs/layer4/results_control.json")
    prompt_modes(payload, out_dir / "prompt_modes.png")
    if l0:
        ladder_final(payload, l0, l2, out_dir / "ladder_final.png")
        silver_final(payload, l0, l2, ctrl, out_dir / "silver_agreement_final.png")
    llm_vs_xlmr(payload, out_dir / "llm_vs_xlmr_errors.png")
    per_language(payload, out_dir / "layer4_per_language.png")
    print(f"графики: {out_dir} (dpi={DPI})")


def make_control_only_plots(ctrl, out_dir):
    """Основного прогона слоя 4 ещё нет — строим то, ради чего делался контроль.

    Строятся все пять, но контрольный прогон ВЕЗДЕ помечен как контроль, а не как
    бейзлайн слоя 4: отдельный цвет со штриховкой в лесенке, красные подписи,
    сноски. DeepSeek — автор корпуса и silver-меток, его F1 в ряд слоёв не встаёт.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l0 = _load("runs/layer0/results.json")
    l2 = _load("runs/layer2/results.json")
    if not l0:
        raise SystemExit("нет runs/layer0/results.json — график согласия не построить")
    silver_final(None, l0, l2, ctrl, out_dir / "silver_agreement_final.png")
    prompt_modes(ctrl, out_dir / "prompt_modes_control.png")
    ladder_final(ctrl, l0, l2, out_dir / "ladder_final.png", is_control=True)
    llm_vs_xlmr(ctrl, out_dir / "llm_vs_xlmr_errors.png", is_control=True)
    per_language(ctrl, out_dir / "layer4_per_language.png", is_control=True)
    print(f"графики контроля: {out_dir} — silver_agreement_final.png, "
          f"prompt_modes_control.png, ladder_final.png, llm_vs_xlmr_errors.png, "
          f"layer4_per_language.png (dpi={DPI})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/layer4/results.json")
    ap.add_argument("--out", default="runs/layer4/plots")
    ap.add_argument("--control-only", action="store_true",
                    help="основного прогона нет: строить только графики контроля")
    a = ap.parse_args()
    if a.control_only:
        make_control_only_plots(_load("runs/layer4/results_control.json"), a.out)
    else:
        make_plots(json.loads(pathlib.Path(a.results).read_text(encoding="utf-8")), a.out)
