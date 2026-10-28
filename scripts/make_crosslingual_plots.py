#!/usr/bin/env python3
"""Графики кросс-язычного переноса. Числа только из runs/crosslingual/results.json.

Отсутствующая модель (например, xlm-r до прогона на Kaggle) просто не рисуется —
число панелей подстраивается.

  python scripts/make_crosslingual_plots.py
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
HOW = "dev_source"                      # честный zero-shot; контроль — val_source
MODEL_RU = {"tfidf": "слой 1: TF-IDF",
            "layer2": "слой 2: e5 + LightGBM",
            "xlmr": "файнтюн: xlm-roberta-base"}
MODEL_COLOR = {"tfidf": "#DD8452", "layer2": "#55A868", "xlmr": "#4C72B0"}
LANGS = ["en", "ru", "zh"]
KS_POS = {0: 0, 10: 1, 50: 2, 100: 3}   # равные интервалы вместо линейной шкалы
ROW_RU = {"single_en": "только en", "single_ru": "только ru", "single_zh": "только zh",
          "loo_en+ru": "en+ru (928)", "loo_en+zh": "en+zh (929)", "loo_ru+zh": "ru+zh (927)",
          "loo_en+ru_n464": "en+ru (464)", "loo_en+zh_n464": "en+zh (464)",
          "loo_ru+zh_n464": "ru+zh (464)", "all3": "все три (1392)",
          "all3_n464": "все три (464)"}


def models_of(p):
    return [m for m in ("tfidf", "layer2", "xlmr") if m in p["models"]]


def transfer_matrix(p, path):
    """Тепловая карта: строки — обучение, столбцы — тест. Панель на модель."""
    ms = models_of(p)
    fig, axes = plt.subplots(1, len(ms), figsize=(6.2 * len(ms), 7), squeeze=False)
    for ax, m in zip(axes[0], ms):
        mat = p["models"][m]["by_selection"][HOW]["matrix"]
        rows = [r for r in ROW_RU if r in mat]
        M = np.array([[mat[r][lg]["f1"]["median"] for lg in LANGS] for r in rows])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for i, r in enumerate(rows):
            for j, lg in enumerate(LANGS):
                zero = lg not in mat[r]["_sources"]
                bad = (mat[r][lg].get("n_collapsed_benign", 0)
                       + mat[r][lg].get("n_collapsed_scam", 0))
                txt = f"{M[i, j]:.3f}"
                if zero:
                    txt += "\nzero-shot"
                if bad:
                    txt += f"\n⚠{bad} сид(а) вырожд."
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if zero else "normal",
                        color="#111")
                if zero:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#222", lw=2.2))
        ax.set_xticks(range(len(LANGS)), [f"тест {lg}" for lg in LANGS], fontsize=10)
        ax.set_title(MODEL_RU[m], fontsize=12)
        ax.set_xlabel("язык теста")
        if m == ms[0]:                       # подписи строк только у левой панели —
            ax.set_yticks(range(len(rows)), [ROW_RU[r] for r in rows], fontsize=9)
            ax.set_ylabel("языки обучения")  # иначе они налезают на соседние клетки
        else:
            ax.set_yticks(range(len(rows)), [""] * len(rows))
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02, label="F1 на gold-test v2")
    fig.suptitle("Матрица кросс-язычного переноса: обучали на X → тест на Y (медиана по 3 сидам)\n"
                 "жирная рамка — целевой язык в обучении не участвовал (zero-shot)",
                 fontsize=13)
    fig.text(0.01, 0.01, "Отбор гиперпараметра/эпохи — по gold-dev языков-источников, "
                         "целевой язык в отборе не участвует. В клетках МЕДИАНА по 3 сидам: у части\nконфигураций отдельные сиды схлопываются в вырожденный предсказатель, и среднее описывает режим, которого не было ни в одном прогоне.",
             fontsize=8, color="#555")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def transfer_gap(p, path):
    """Деградация zero-shot: полный вариант против выровненного по размеру."""
    ms = models_of(p)
    fig, axes = plt.subplots(1, len(ms), figsize=(5.6 * len(ms), 5.2), squeeze=False,
                             sharey=True)
    for ax, m in zip(axes[0], ms):
        g = p["models"][m]["by_selection"][HOW]["gaps"]
        x = np.arange(len(LANGS))
        w = 0.38
        for k, (tag, lab, col) in enumerate((
                ("loo_full", "два источника, полный train", "#4C72B0"),
                ("loo_matched", "два источника, выровнено до 464", "#C44E52"))):
            vals = [g[tag].get(lg, {}).get("gap", np.nan) for lg in LANGS]
            ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=lab, edgecolor="white")
            for xi, v in zip(x + (k - 0.5) * w, vals):
                if np.isfinite(v):
                    ax.text(xi, v + (0.006 if v >= 0 else -0.02), f"{v:+.3f}",
                            ha="center", fontsize=8.5)
        # одиночные источники — точками, чтобы видеть разброс направлений
        for j, lg in enumerate(LANGS):
            for src, d in sorted(g["single_source"].get(lg, {}).items()):
                ax.plot(j, d["gap"], "o", ms=7, color="#333", zorder=5)
                ax.annotate(f"{src}→", (j, d["gap"]), textcoords="offset points",
                            xytext=(9, -3), fontsize=8, color="#333")
        ax.axhline(0, color="#222", lw=1.2)
        ax.set_xticks(x, [f"цель {lg}" for lg in LANGS], fontsize=10)
        ax.set_title(MODEL_RU[m], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("зазор F1: zero-shot − in-language")
    # легенда на уровне фигуры: внутри панели она перекрывает подписи столбцов
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               frameon=False)
    fig.suptitle("Деградация zero-shot относительно in-language\n"
                 "столбцы — два источника, точки — по одному источнику", fontsize=13)
    fig.text(0.01, 0.01, "Ноль = перенос не хуже обучения на самом целевом языке. "
                         "Выравнивание до 464 отделяет эффект переноса от эффекта объёма.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fewshot_curve(p, path):
    """F1 на целевом языке как функция k, с горизонталью in-language."""
    ms = models_of(p)
    fig, axes = plt.subplots(1, len(ms), figsize=(5.8 * len(ms), 5.2), squeeze=False,
                             sharey=True)
    cmap = {"en": "#4C72B0", "ru": "#DD8452", "zh": "#55A868"}
    for ax, m in zip(axes[0], ms):
        fs = p["models"][m]["by_selection"][HOW]["fewshot"]
        # ось КАТЕГОРИАЛЬНАЯ: при линейной шкале 0/10 слипаются, а 50→100 растянуто
        for pair, d in fs.items():
            src, tgt = pair.split("->")
            ks = sorted(d["points"], key=int)     # после JSON ключи — строки
            xs = [KS_POS[int(k)] for k in ks]
            ax.plot(xs, [d["points"][k] for k in ks], marker="o", color=cmap[tgt],
                    ls="-" if src == "en" else ("--" if src == "ru" else ":"),
                    label=f"{src}→{tgt}", ms=6)
        for tgt in LANGS:
            ref = next((d["in_language"] for pr, d in fs.items()
                        if pr.endswith(f"->{tgt}")), None)
            if ref:
                ax.axhline(ref, color=cmap[tgt], lw=1.1, alpha=0.45, ls="-")
                ax.annotate(f"in-language {tgt}", (3.02, ref), fontsize=7.5,
                            color=cmap[tgt], va="bottom", ha="right")
        ax.set_xticks(list(KS_POS.values()), [str(k) for k in KS_POS])
        ax.set_xlim(-0.25, 3.25)
        ax.set_xlabel("k — примеров целевого языка в обучении (шкала категориальная)")
        ax.set_title(MODEL_RU[m], fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("F1 на целевом языке (gold-test v2)")
    axes[0][0].legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Сколько примеров целевого языка нужно, чтобы догнать in-language\n"
                 "цвет — целевой язык, тип линии — язык-источник", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def script_distance(p, path):
    """Зазор против пары письменностей + проверка асимметрии направления."""
    ms = models_of(p)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    pairs = ["кириллица ↔ латиница", "иероглифы ↔ латиница", "иероглифы ↔ кириллица"]
    x = np.arange(len(pairs))
    w = 0.8 / len(ms)
    for k, m in enumerate(ms):
        bs = p["models"][m]["by_selection"][HOW]["by_script"]
        vals = [bs.get(pr, {}).get("mean_gap", np.nan) for pr in pairs]
        errs = [bs.get(pr, {}).get("std_gap", 0.0) for pr in pairs]
        axes[0].bar(x + (k - (len(ms) - 1) / 2) * w, vals, w, yerr=errs, capsize=3,
                    color=MODEL_COLOR[m], label=MODEL_RU[m], edgecolor="white")
    axes[0].axhline(0, color="#222", lw=1.2)
    axes[0].set_xticks(x, [pr.replace(" ↔ ", "\n↔ ") for pr in pairs], fontsize=9.5)
    axes[0].set_ylabel("средний зазор F1 (zero-shot − in-language)")
    axes[0].set_title("Гипотеза графической дистанции:\nзазор по паре письменностей",
                      fontsize=11)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    # вторая панель: асимметрия направления против размаха между парами
    for k, m in enumerate(ms):
        asym = p["models"][m]["by_selection"][HOW]["asymmetry"]
        keys = [a for a in asym if not a.startswith("_")]
        vals = [asym[a]["abs_asymmetry"] for a in keys]
        axes[1].bar(np.arange(len(keys)) + (k - (len(ms) - 1) / 2) * w, vals, w,
                    color=MODEL_COLOR[m], label=MODEL_RU[m], edgecolor="white")
        spread = [v["mean_gap"] for v in
                  p["models"][m]["by_selection"][HOW]["by_script"].values()]
        axes[1].axhline(max(spread) - min(spread), color=MODEL_COLOR[m], ls="--", lw=1.4)
    axes[1].set_xticks(np.arange(3), ["en ↔ ru", "en ↔ zh", "ru ↔ zh"], fontsize=10)
    axes[1].set_ylabel("|зазор(A→B) − зазор(B→A)|")
    axes[1].set_title("Асимметрия направления\n(пунктир — размах между парами "
                      "письменностей у той же модели)", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)
    fig.text(0.01, 0.01, "Дистанция письменностей симметрична. Если столбцы справа "
                         "сопоставимы с пунктиром или выше, направление переноса важнее "
                         "пары письменностей.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def models_comparison(p, path):
    """Способность к переносу у слоёв 1, 2 и файнтюна одной картинкой."""
    ms = models_of(p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(LANGS))
    w = 0.8 / len(ms)
    for k, m in enumerate(ms):
        mat = p["models"][m]["by_selection"][HOW]["matrix"]
        zs, il = [], []
        for lg in LANGS:
            cells = [mat[r][lg]["f1"]["median"] for r in mat
                     if lg not in mat[r]["_sources"] and r.startswith("single_")]
            zs.append(float(np.mean(cells)) if cells else np.nan)
            il.append(mat[f"single_{lg}"][lg]["f1"]["median"])
        axes[0].bar(x + (k - (len(ms) - 1) / 2) * w, zs, w, color=MODEL_COLOR[m],
                    label=MODEL_RU[m], edgecolor="white")
        axes[1].bar(x + (k - (len(ms) - 1) / 2) * w, np.array(zs) - np.array(il), w,
                    color=MODEL_COLOR[m], label=MODEL_RU[m], edgecolor="white")
    axes[0].set_title("Zero-shot F1 (среднее по двум источникам)", fontsize=11)
    axes[0].set_ylabel("F1 на целевом языке")
    axes[1].set_title("Зазор к in-language", fontsize=11)
    axes[1].set_ylabel("F1(zero-shot) − F1(in-language)")
    axes[1].axhline(0, color="#222", lw=1.2)
    for ax in axes:
        ax.set_xticks(x, [f"цель {lg}" for lg in LANGS], fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("Способность к переносу по слоям", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/crosslingual/results.json")
    ap.add_argument("--out", default="runs/crosslingual/plots")
    a = ap.parse_args()
    p = json.loads(pathlib.Path(a.results).read_text(encoding="utf-8"))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    transfer_matrix(p, out / "transfer_matrix.png")
    transfer_gap(p, out / "transfer_gap.png")
    fewshot_curve(p, out / "fewshot_curve.png")
    script_distance(p, out / "script_distance.png")
    models_comparison(p, out / "models_transfer_comparison.png")
    print(f"графики: {out} — модели {models_of(p)} (dpi={DPI})")


if __name__ == "__main__":
    main()
