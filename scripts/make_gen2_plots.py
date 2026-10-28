#!/usr/bin/env python3
"""Графики переноса на второй генератор. Числа только из runs/gen2/eval.json.

  python scripts/make_gen2_plots.py

Без eval.json не падает — говорит, что сначала нужен прогон scripts/gen2_eval.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

DPI = 150
ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "runs" / "gen2" / "eval.json"
PLOTS = ROOT / "runs" / "gen2" / "plots"

NAME_RU = {"keywords": "ключевые слова\n(слой 0)", "tfidf": "TF-IDF\n(слой 1)",
           "layer2": "эмбеддинги e5\n(слой 2)", "xlmr": "xlm-roberta-base\n(файнтюн)"}
COLOR_V1, COLOR_G2 = "#4C72B0", "#C44E52"


def cross_generator_transfer(res, path):
    """F1 каждой модели на v1 и на gen2 рядом + падение подписью."""
    models = [m for m in ("keywords", "tfidf", "layer2", "xlmr") if m in res["models"]]
    v1 = [res["models"][m]["v1_test_vs_silver"]["f1"] for m in models]
    g2 = [res["models"][m]["gen2_vs_silver"]["f1"] for m in models]
    x = np.arange(len(models))
    w = 0.38

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    ax.bar(x - w / 2, v1, w, label=f"v1 test, n={res['n_v1_test']} (свой генератор)",
           color=COLOR_V1)
    ax.bar(x + w / 2, g2, w, label=f"gen2, n={res['n_gen2']} (чужой генератор)",
           color=COLOR_G2)
    for i, (a, b) in enumerate(zip(v1, g2)):
        ax.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
        drop = a - b
        ax.annotate(f"{-drop:+.3f}", xy=(i, max(a, b) + 0.055), ha="center",
                    fontsize=10, fontweight="bold",
                    color="#C44E52" if drop > 0.02 else "#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([NAME_RU[m] for m in models], fontsize=9)
    ax.set_ylabel("F1 против silver-меток своего генератора")
    ax.set_ylim(0, 1.12)
    ax.set_title("Перенос на второй генератор: падение F1")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # согласие с silver: если модель ловит стиль, на чужом стиле согласие обязано упасть
    sa1 = [res["models"][m]["silver_agreement_v1"] for m in models]
    sa2 = [res["models"][m]["silver_agreement_gen2"] for m in models]
    ax2.bar(x - w / 2, sa1, w, color=COLOR_V1)
    ax2.bar(x + w / 2, sa2, w, color=COLOR_G2)
    for i, (a, b) in enumerate(zip(sa1, sa2)):
        ax2.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
        ax2.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([NAME_RU[m] for m in models], fontsize=9)
    ax2.set_ylabel("доля совпадений с silver-метками")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Согласие с silver: свой стиль против чужого")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Стилевой отпечаток генератора: те же модели на чужой партии",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def keyword_transfer(res, path):
    """Срабатывание ключевых слов v1 на gen2: главный индикатор стилевого отпечатка."""
    kw = res["models"].get("keywords")
    if not kw:
        return False
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    groups = ["scam", "benign"]
    v1 = [kw["mean_hits_v1"][g] for g in groups]
    g2 = [kw["mean_hits_gen2"][g] for g in groups]
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, v1, w, label="v1 (свой генератор)", color=COLOR_V1)
    ax.bar(x + w / 2, g2, w, label="gen2 (чужой)", color=COLOR_G2)
    for i, (a, b) in enumerate(zip(v1, g2)):
        ax.text(i - w / 2, a + 0.1, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.1, f"{b:.2f}", ha="center", fontsize=9)
    ax.axhline(kw["threshold"], color="#937860", ls="--", lw=1.4,
               label=f"порог решения N={kw['threshold']}")
    ax.set_xticks(x)
    ax.set_xticklabels(["scam", "benign"])
    ax.set_ylabel("сработало ключевых слов на диалог (среднее)")
    ax.set_title("Срабатывание 20 слов на язык, добытых из v1")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    langs = sorted(kw["hits_by_lang_gen2"])
    xv = np.arange(len(langs))
    ax2.bar(xv - w / 2, [kw["hits_by_lang_v1"][ln] for ln in langs], w,
            label="v1", color=COLOR_V1)
    ax2.bar(xv + w / 2, [kw["hits_by_lang_gen2"][ln] for ln in langs], w,
            label="gen2", color=COLOR_G2)
    ax2.axhline(kw["threshold"], color="#937860", ls="--", lw=1.4)
    ax2.set_xticks(xv)
    ax2.set_xticklabels(langs)
    ax2.set_ylabel("сработало ключевых слов на диалог (среднее)")
    ax2.set_title("По языкам (все диалоги набора)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    f1v1 = kw["v1_test_vs_silver"]["f1"]
    f1g2 = kw["gen2_vs_silver"]["f1"]
    fig.suptitle(f"Ключевые слова слоя 0 на чужом генераторе: "
                 f"F1 {f1v1:.3f} → {f1g2:.3f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Графики переноса на второй генератор")
    ap.add_argument("--eval", default=str(EVAL))
    ap.add_argument("--out-dir", default=str(PLOTS))
    a = ap.parse_args()

    src = pathlib.Path(a.eval)
    if not src.exists():
        print(f"нет {src} — сначала прогон: python scripts/gen2_eval.py")
        print("(а до него нужны данные gen2: python scripts/generate_gen2.py, "
              "нужен ANTHROPIC_API_KEY)")
        return 0
    res = json.loads(src.read_text(encoding="utf-8"))
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cross_generator_transfer(res, out / "cross_generator_transfer.png")
    print(f"записано: {out / 'cross_generator_transfer.png'}")
    if keyword_transfer(res, out / "keyword_transfer.png"):
        print(f"записано: {out / 'keyword_transfer.png'}")
    else:
        print("keyword_transfer.png пропущен: в eval.json нет блока keywords")
    return 0


if __name__ == "__main__":
    sys.exit(main())
