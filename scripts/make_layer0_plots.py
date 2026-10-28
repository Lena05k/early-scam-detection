#!/usr/bin/env python3
"""Графики слоя 0. Числа берутся только из runs/layer0/results.json.

  python scripts/make_layer0_plots.py            # перерисовать по готовому results.json
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
C_L0 = "#937860"        # слой 0
C_TRAINED = "#4C72B0"   # обученные модели
ORDER = ["const_scam", "const_benign", "length", "url", "keywords", "logreg3"]


def ladder(payload, path):
    """F1 всех предсказателей слоя 0 плюс три обученные модели, по возрастанию."""
    rows = payload["ladder"]
    names = [r["model"] for r in rows]
    vals = [r["f1"] for r in rows]
    cols = [C_TRAINED if r["layer"] == "обученные" else C_L0 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}   (gold {r['gold']})" for n, r in zip(names, rows)], fontsize=10)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("F1 на gold-test (n=303)")
    ax.set_title("Лесенка качества is_scam: слой 0 против обученных моделей", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_L0),
               plt.Rectangle((0, 0), 1, 1, color=C_TRAINED)]
    ax.legend(handles, ["слой 0 (без обучения нейросетей)", "обученные модели"],
              loc="lower right", fontsize=10)
    fig.text(0.01, 0.012,
             "Слой 0 и xlm-r — на gold v2; TF-IDF и mDeBERTa — опубликованные числа на gold v1\n"
             "(различие 3 метки из 303; контрольный пересчёт TF-IDF на v2 даёт 0.8713).\n"
             "Сиды не нужны: все предсказатели слоя 0 детерминированы.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def confusions(payload, path):
    """Матрицы ошибок шести тривиальных предсказателей."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, key in zip(axes.ravel(), ORDER):
        p = payload["predictors"][key]
        cm = np.array(p["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() else 1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "#222")
        ax.set_xticks([0, 1], ["benign", "scam"])
        ax.set_yticks([0, 1], ["benign", "scam"])
        ax.set_xlabel("предсказание", fontsize=9)
        ax.set_ylabel("gold", fontsize=9)
        ax.set_title(f"{p['name_ru']}\nF1 {p['metrics']['f1']:.4f} · "
                     f"κ {p['metrics']['cohen_kappa']:.4f}", fontsize=10)
    fig.suptitle("Матрицы ошибок слоя 0, gold-test v2 (n=303)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def tuning(payload, path):
    """Что именно подбиралось на gold-dev: порог длины, порог N слов, C логрегрессии."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    sw = payload["predictors"]["length"]["tuning"]["sweep"]
    sel = payload["predictors"]["length"]["tuning"]["selected"]
    for direction, lab, col in (("ge", "число реплик ≥ порога → scam", "#4C72B0"),
                                ("le", "число реплик ≤ порога → scam", "#DD8452")):
        pts = [d for d in sw if d["direction"] == direction]
        axes[0].plot([d["threshold"] for d in pts], [d["dev_f1"] for d in pts],
                     "o-", color=col, label=lab, ms=4)
    axes[0].plot([sel["threshold"]], [sel["dev_f1"]], "*", ms=18, color="#C44E52",
                 label="выбрано", zorder=5)
    axes[0].set_xlabel("порог по числу реплик")
    axes[0].set_title("3. Длина диалога", fontsize=11)
    axes[0].legend(fontsize=8)

    sw = payload["predictors"]["keywords"]["tuning"]["sweep"]
    axes[1].plot([d["n"] for d in sw], [d["dev_f1"] for d in sw], "o-", color="#55A868", ms=5)
    nsel = payload["predictors"]["keywords"]["tuning"]["selected_n"]
    axes[1].plot([nsel], [payload["predictors"]["keywords"]["tuning"]["dev_f1"]], "*",
                 ms=18, color="#C44E52", zorder=5)
    axes[1].set_xlabel("порог: сработало N и более ключевых слов")
    axes[1].set_title(f"5. Ключевые слова (выбрано N={nsel})", fontsize=11)

    sw = payload["predictors"]["logreg3"]["tuning"]["sweep"]
    axes[2].plot([d["C"] for d in sw], [d["dev_f1"] for d in sw], "o-", color="#8172B3", ms=5)
    csel = payload["predictors"]["logreg3"]["tuning"]["selected_C"]
    axes[2].plot([csel], [payload["predictors"]["logreg3"]["tuning"]["dev_f1"]], "*",
                 ms=18, color="#C44E52", zorder=5)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("сила регуляризации C (лог. шкала)")
    axes[2].set_title(f"6. Логрегрессия, 3 признака (выбрано C={csel})", fontsize=11)

    for ax in axes:
        ax.set_ylabel("F1 на gold-dev (n=100)")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("Подбор порогов — только на gold-dev, тест не участвует", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_language(payload, path):
    """F1 слоя 0 по языкам."""
    langs = sorted(payload["predictors"]["url"]["per_language"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ORDER))
    w = 0.26
    cols = {"en": "#4C72B0", "ru": "#DD8452", "zh": "#55A868"}
    for k, lg in enumerate(langs):
        vals = [payload["predictors"][p]["per_language"][lg]["f1"] for p in ORDER]
        n = payload["predictors"][ORDER[0]]["per_language"][lg]["n"]
        ax.bar(x + (k - 1) * w, vals, w, color=cols.get(lg, "#937860"),
               label=f"{lg} (n={n})", edgecolor="white")
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    short = {"const_scam": "константа\n«всегда scam»", "const_benign": "константа\n«всегда benign»",
             "length": "длина\nдиалога", "url": "наличие\nURL", "keywords": "ключевые\nслова",
             "logreg3": "логрегрессия\nна 3 признаках"}
    ax.set_xticks(x, [short[p] for p in ORDER], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 на gold-test v2")
    ax.set_title("Слой 0 по языкам", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_plots(payload, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder(payload, out_dir / "layer0_ladder.png")
    confusions(payload, out_dir / "layer0_confusion.png")
    tuning(payload, out_dir / "layer0_tuning.png")
    per_language(payload, out_dir / "layer0_per_language.png")
    print(f"графики: {out_dir} — layer0_ladder.png, layer0_confusion.png, "
          f"layer0_tuning.png, layer0_per_language.png (dpi={DPI})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/layer0/results.json")
    ap.add_argument("--out", default="runs/layer0/plots")
    a = ap.parse_args()
    make_plots(json.loads(pathlib.Path(a.results).read_text(encoding="utf-8")), a.out)
