#!/usr/bin/env python3
"""Сводное сравнение бейзлайнов на gold-test: TF-IDF, xlm-roberta-base, mDeBERTa-v3.

Все три обучены по одному протоколу: train 1392 (silver) → отбор по gold-dev 100
(человеческие метки) → замер на gold-test 303. Числа берутся ТОЛЬКО из results.json
соответствующих прогонов, ничего не пересчитывается и не досочиняется.

  python scripts/plot_model_comparison.py
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
COLORS = {"TF-IDF + LogReg": "#937860",
          "xlm-roberta-base": "#4C72B0",
          "mDeBERTa-v3-base": "#C44E52"}


def load(path, name):
    """results.json любого из трёх прогонов → единая структура."""
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    s = d["summary"]
    out = {"name": name, "summary": s,
           "cm": np.array(d["confusion_matrix_mean"]),
           "n_test": None, "per_epoch": None, "best_epochs": d.get("best_epochs")}
    if "runs" in d and d["runs"] and "per_epoch" in d["runs"][0]:
        out["per_epoch"] = [(r["seed"], r["per_epoch"], r["best_epoch"]) for r in d["runs"]]
        out["n_test"] = d["runs"][0]["n_test"]
    else:                                    # TF-IDF: нет эпох, есть сетка по C
        out["n_test"] = d["split"]["n_test"]
        out["per_lang"] = {lg: {m: float(np.mean([r["selected_by_gold_dev"]["per_language"][lg][m]
                                                  for r in d["runs"]]))
                                for m in ("f1", "precision", "recall")}
                           for lg in d["runs"][0]["selected_by_gold_dev"]["per_language"]}
    if "per_lang" not in out:
        out["per_lang"] = {lg: {m: float(np.mean([r["per_language"][lg][m] for r in d["runs"]]))
                                for m in ("f1", "precision", "recall")}
                           for lg in d["runs"][0]["per_language"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfidf", default="runs/tfidf_gold_dev/results.json")
    ap.add_argument("--xlmr", default="runs/seeds_gold_dev/results.json")
    ap.add_argument("--mdeberta", default="runs/mdeberta_gold_dev/results.json")
    ap.add_argument("--out", default="runs/model_comparison")
    args = ap.parse_args()

    models = [load(args.tfidf, "TF-IDF + LogReg"),
              load(args.xlmr, "xlm-roberta-base"),
              load(args.mdeberta, "mDeBERTa-v3-base")]
    out_dir = pathlib.Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    n_test = models[0]["n_test"]

    # ── 1. сводные метрики со std ──
    KEYS = [("f1", "F1"), ("precision", "precision"), ("recall", "recall"),
            ("cohen_kappa", "Cohen κ"), ("roc_auc", "ROC AUC")]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(KEYS))
    w = 0.26
    for i, m in enumerate(models):
        mu = [m["summary"][k]["mean"] for k, _ in KEYS]
        sd = [m["summary"][k]["std"] for k, _ in KEYS]
        ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=4,
               color=COLORS[m["name"]], label=m["name"])
    ax.set_xticks(x, [lab for _, lab in KEYS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("значение метрики")
    ax.set_title(f"Бейзлайны на золотом тесте (n={n_test}), отбор чекпоинта по gold-dev\n"
                 f"усы — стандартное отклонение по 3 сидам")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "models_metrics.png", dpi=DPI)
    plt.close(fig)

    # ── 2. расхождение silver/gold: воспроизводится ли эффект на обеих архитектурах ──
    have = [m for m in models if m["per_epoch"]]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(12, 5.4), sharey=True)
        if len(have) == 1:
            axes = [axes]
        for ax_, m in zip(axes, have):
            for seed, pe, best in m["per_epoch"]:
                eps = [p["epoch"] for p in pe]
                ax_.plot(eps, [p["silver_val_f1"] for p in pe], marker="o",
                         linewidth=1.8, alpha=0.85, color=COLORS[m["name"]], linestyle="-")
                ax_.plot(eps, [p["gold_dev_f1"] for p in pe], marker="s",
                         linewidth=1.8, alpha=0.85, color=COLORS[m["name"]], linestyle="--")
                ax_.scatter([best], [next(p["gold_dev_f1"] for p in pe if p["epoch"] == best)],
                            s=170, facecolors="none", edgecolors="black", linewidths=1.8, zorder=5)
            ax_.set_title(f"{m['name']}\nлучшие эпохи: {m['best_epochs']}")
            ax_.set_xlabel("эпоха")
            ax_.set_xticks(sorted({p["epoch"] for _, pe, _ in m["per_epoch"] for p in pe}))
        axes[0].set_ylabel("F1")
        fig.suptitle("Расхождение silver_val (сплошные) и gold_dev (пунктир) по эпохам, 3 сида\n"
                     "чёрный кружок — выбранный чекпоинт")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "divergence_by_model.png", dpi=DPI)
        plt.close(fig)

    # ── 3. по языкам ──
    langs = sorted(models[0]["per_lang"])
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(langs))
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * w, [m["per_lang"][lg]["f1"] for lg in langs], w,
               color=COLORS[m["name"]], label=m["name"])
    ax.set_xticks(x, langs)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("язык")
    ax.set_ylabel("F1 (класс scam)")
    ax.set_title(f"F1 по языкам на золотом тесте (n={n_test // 3} на язык)\n"
                 "⚠ in-language, не кросс-язычный трансфер: все модели видели все три языка")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "models_per_language.png", dpi=DPI)
    plt.close(fig)

    # ── 4. матрицы ошибок рядом ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax_, m in zip(axes, models):
        cm = m["cm"]
        im = ax_.imshow(cm, cmap="Blues", vmin=0, vmax=max(mm["cm"].max() for mm in models))
        rs = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                ax_.text(j, i, f"{cm[i,j]:.0f}\n{cm[i,j]/rs[i,0]:.0%}", ha="center",
                         va="center", fontsize=13,
                         color="white" if cm[i, j] > cm.max() / 2 else "#222")
        ax_.set_xticks([0, 1], ["benign", "scam"])
        ax_.set_yticks([0, 1], ["benign", "scam"])
        ax_.set_xlabel("предсказание")
        ax_.set_title(f"{m['name']}\nF1 {m['summary']['f1']['mean']:.4f}")
        ax_.grid(False)
    axes[0].set_ylabel("gold")
    fig.suptitle(f"Матрицы ошибок на золотом тесте (n={n_test}), усреднены по 3 сидам")
    fig.colorbar(im, ax=axes, shrink=0.75, label="диалогов")
    fig.savefig(out_dir / "plots" / "models_confusion.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ── сводная таблица ──
    print(f"gold-test, n={n_test}, отбор чекпоинта по gold-dev, 3 сида\n")
    hdr = f"{'модель':>20} " + " ".join(f"{lab:>18}" for _, lab in KEYS)
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        row = f"{m['name']:>20} "
        for k, _ in KEYS:
            row += f"{m['summary'][k]['mean']:>11.4f}±{m['summary'][k]['std']:.4f}"
        print(row)
    print(f"\n{'модель':>20} {'совпадение с silver':>22} {'лучшие эпохи':>16}")
    for m in models:
        sa = m["summary"].get("silver_agreement")
        be = m["best_epochs"] if m["best_epochs"] else "— (нет эпох)"
        print(f"{m['name']:>20} {sa['mean']:>16.4f}±{sa['std']:.4f} {str(be):>16}")

    summary = {m["name"]: {k: m["summary"][k] for k, _ in KEYS} for m in models}
    (out_dir / "comparison.json").write_text(
        json.dumps({"n_test": n_test, "metrics": summary,
                    "best_epochs": {m["name"]: m["best_epochs"] for m in models}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nграфики:")
    for f in sorted((out_dir / "plots").glob("*.png")):
        print("  ", f.resolve())
    print("  ", (out_dir / "comparison.json").resolve())


if __name__ == "__main__":
    main()
