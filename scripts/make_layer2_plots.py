#!/usr/bin/env python3
"""Графики слоя 2. Числа берутся только из runs/layer2/results.json.

  python scripts/make_layer2_plots.py
"""
import argparse
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
               "файнтюн": "#4C72B0"}
AGG_RU = {"mean": "среднее", "mean_max": "[среднее, максимум]",
          "pos_mean": "взвешенное по позиции"}


def encoder_comparison(p, path):
    """Два энкодера × три агрегации: слева отбор (gold-dev), справа замер (gold-test)."""
    encs = list(dict.fromkeys(c["encoder"] for c in p["configs"].values()))
    aggs = p["aggregations"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = ["#4C72B0", "#C44E52"]
    for ax, field, title in ((axes[0], "gold_dev_f1", "Отбор: gold-dev (n=100)"),
                             (axes[1], "gold_test_f1", "Замер: gold-test v2 (n=303)")):
        x = np.arange(len(aggs))
        w = 0.36
        for k, enc in enumerate(encs):
            mu = [p["configs"][f"{enc}|{a}"][field]["mean"] for a in aggs]
            sd = [p["configs"][f"{enc}|{a}"][field]["std"] for a in aggs]
            ax.bar(x + (k - 0.5) * w, mu, w, yerr=sd, capsize=4, color=cols[k],
                   label=enc, edgecolor="white")
            for xi, m, s in zip(x + (k - 0.5) * w, mu, sd):
                ax.text(xi, m + s + 0.008, f"{m:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, [AGG_RU[a] for a in aggs], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].legend(fontsize=9, loc="upper left")
    be, ba = p["best_config"].split("|")
    fig.suptitle(f"Слой 2: энкодеры × агрегации, усы = std по 3 сидам.  "
                 f"Выбрано по gold-dev: {be} / {AGG_RU[ba]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def ladder_full(p, path):
    """Полная лесенка: слой 0 → TF-IDF → слой 2 → файнтюн."""
    rows = p["ladder"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(rows))
    ax.barh(y, [r["f1"] for r in rows],
            xerr=[r.get("std", 0.0) for r in rows], capsize=3,
            color=[LAYER_COLOR[r["layer"]] for r in rows], edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['model']}   (gold {r['gold']})" for r in rows], fontsize=9.5)
    for yi, r in zip(y, rows):
        lab = f"{r['f1']:.4f}" + (f" ± {r['std']:.4f}" if r.get("std") else "")
        ax.text(r["f1"] + r.get("std", 0.0) + 0.01, yi, lab, va="center", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 на gold-test (n=303)")
    ax.set_title("Лесенка качества is_scam: от константы до файнтюна", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=LAYER_COLOR[k]) for k in LAYER_COLOR],
              ["слой 0: тривиальные", "слой 1: TF-IDF", "слой 2: эмбеддинги + бустинг",
               "файнтюн трансформера"], loc="lower right", fontsize=9)
    fig.text(0.01, 0.012,
             "Слой 0, слой 2 и xlm-r — на gold v2; TF-IDF и mDeBERTa — опубликованные числа "
             "на gold v1 (различие 3 метки из 303).\n"
             "Усы там, где считались сиды: слой 2 (LightGBM недетерминирован при "
             "feature_fraction<1). Слой 0 детерминирован, у него усов нет.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def confusion(p, path):
    cm = np.array(p["confusion_matrix_mean"])
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=19,
                    color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xticks([0, 1], ["benign", "scam"])
    ax.set_yticks([0, 1], ["benign", "scam"])
    ax.set_xlabel("предсказание")
    ax.set_ylabel("gold")
    s = p["summary"]
    be, ba = p["best_config"].split("|")
    ax.set_title(f"Слой 2, {be} / {AGG_RU[ba]}\ngold-test v2 (n=303), среднее по 3 сидам\n"
                 f"F1 {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f} · "
                 f"κ {s['cohen_kappa']['mean']:.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def silver_agreement(p, path):
    """Кто копирует разметку DeepSeek."""
    items = list(p["silver_agreement_by_model"].items())
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(items))
    cols = []
    for k, _ in items:
        lay = k.split(":")[0]
        cols.append(LAYER_COLOR.get(lay if lay != "файнтюн" else "файнтюн", "#937860"))
    ax.barh(y, [v for _, v in items], color=cols, edgecolor="white", height=0.68)
    ax.set_yticks(y, [k for k, _ in items], fontsize=10)
    for yi, (_, v) in zip(y, items):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=10)
    ref = p["silver_vs_gold"]
    ax.axvline(ref, color="#C44E52", ls="--", lw=1.6)
    ax.text(ref - 0.012, len(items) - 0.55, f"сам silver\nпротив gold {ref:.3f}",
            color="#C44E52", fontsize=9, va="top", ha="right")
    ax.set_ylim(-0.7, len(items) - 0.1)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("доля диалогов gold-test, где предсказание совпало с silver-меткой DeepSeek")
    ax.set_title("Кто воспроизводит черновую разметку генератора", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.012,
             "Выше — модель ближе к копии DeepSeek; ниже — самостоятельнее. Красный пунктир: "
             "точность самой silver-разметки относительно человека.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_language(p, path):
    langs = sorted(p["per_language"])
    mets = ["f1", "precision", "recall"]
    names = {"f1": "F1", "precision": "precision", "recall": "recall"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(langs))
    w = 0.26
    cols = {"f1": "#55A868", "precision": "#4C72B0", "recall": "#DD8452"}
    for k, m in enumerate(mets):
        mu = [p["per_language"][lg][m]["mean"] for lg in langs]
        sd = [p["per_language"][lg][m]["std"] for lg in langs]
        ax.bar(x + (k - 1) * w, mu, w, yerr=sd, capsize=4, color=cols[m],
               label=names[m], edgecolor="white")
        for xi, mm, ss in zip(x + (k - 1) * w, mu, sd):
            ax.text(xi, mm + ss + 0.012, f"{mm:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"{lg} (n=101)" for lg in langs], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("значение на gold-test v2")
    be, ba = p["best_config"].split("|")
    ax.set_title(f"Слой 2 по языкам: {be} / {AGG_RU[ba]}, усы = std по 3 сидам", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.012, "⚠ Все три языка были в обучении бустинга — это in-language "
                          "качество, а НЕ кросс-язычный перенос.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_plots(payload, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_comparison(payload, out_dir / "encoder_comparison.png")
    ladder_full(payload, out_dir / "ladder_full.png")
    confusion(payload, out_dir / "layer2_confusion.png")
    silver_agreement(payload, out_dir / "silver_agreement_by_layer.png")
    per_language(payload, out_dir / "per_language_layer2.png")
    print(f"графики: {out_dir} — encoder_comparison.png, ladder_full.png, "
          f"layer2_confusion.png, silver_agreement_by_layer.png, "
          f"per_language_layer2.png (dpi={DPI})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/layer2/results.json")
    ap.add_argument("--out", default="runs/layer2/plots")
    a = ap.parse_args()
    make_plots(json.loads(pathlib.Path(a.results).read_text(encoding="utf-8")), a.out)
