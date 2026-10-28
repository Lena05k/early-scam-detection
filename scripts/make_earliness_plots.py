#!/usr/bin/env python3
"""Графики ранней детекции и значимости.

  python scripts/make_earliness_plots.py
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
BIN_COLOR = {"1": "#4C72B0", "2": "#DD8452", "3–4": "#55A868", "5+": "#C44E52"}


def latency_distribution(p, path):
    """Распределение latency: где срабатывает детектор относительно онсета."""
    per = [d for d in p["per_dialog"] if d["latency"] is not None]
    lat = np.array([d["latency"] for d in per])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lo, hi = int(lat.min()), int(lat.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    axes[0].hist(lat, bins=bins, color="#4C72B0", edgecolor="white")
    axes[0].axvline(0, color="#C44E52", lw=2)
    axes[0].annotate("онсет", (0, axes[0].get_ylim()[1] * 0.93), color="#C44E52",
                     fontsize=10, ha="left", xytext=(6, 0), textcoords="offset points")
    axes[0].set_xlabel("latency = позиция срабатывания − t*  (реплик)")
    axes[0].set_ylabel("диалогов")
    axes[0].set_title(f"Распределение latency, порог {p['main_threshold']}\n"
                      f"медиана {p['at_main_threshold']['latency_median']:.1f}, "
                      f"среднее {p['at_main_threshold']['latency_mean']:.2f}", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    # разбивка по бинам t* — там, где эффект и живёт
    for name, col in BIN_COLOR.items():
        v = [d["latency"] for d in per
             if (str(d["onset"]) == name if name in ("1", "2")
                 else (3 <= d["onset"] <= 4 if name == "3–4" else d["onset"] >= 5))]
        if v:
            axes[1].scatter([name] * len(v), v, alpha=0.45, s=45, color=col)
            axes[1].plot([name], [np.median(v)], "_", ms=42, mew=3.5, color=col)
    axes[1].axhline(0, color="#C44E52", lw=2)
    axes[1].set_xlabel("бин истинного онсета t*")
    axes[1].set_ylabel("latency (реплик)")
    axes[1].set_title("Latency по бинам t*: чем позже онсет,\nтем раньше относительно "
                      "него срабатывает детектор", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Через сколько реплик детектор говорит «scam»", fontsize=13)
    fig.text(0.01, 0.01, "Отрицательная latency = сработал ДО реплики, где обман "
                         "становится текстово наблюдаемым. " + p["checkpoint_warning"],
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def earliness_by_bin(p, path):
    """Нормализованная earliness и доля срабатываний по бинам и языкам."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], "by_onset_bin", "по бинам истинного онсета t*"),
                           (axes[1], "by_language", "по языкам")):
        d = p[key]
        names = list(d)
        x = np.arange(len(names))
        ear = [d[k]["earliness_median"] or 0.0 for k in names]
        det = [d[k]["detection_rate"] for k in names]
        ax.bar(x - 0.2, ear, 0.4, color="#55A868", edgecolor="white",
               label="earliness (медиана)")
        ax.bar(x + 0.2, det, 0.4, color="#4C72B0", edgecolor="white",
               label="доля сработавших")
        for xi, (e, dd, k) in enumerate(zip(ear, det, names)):
            ax.text(xi - 0.2, e + 0.03, f"{e:.2f}", ha="center", fontsize=9)
            ax.text(xi + 0.2, dd + 0.03, f"{dd:.2f}", ha="center", fontsize=9)
            ax.text(xi, -0.12, f"n={d[k]['n']}", ha="center", fontsize=8, color="#555")
        ax.axhline(1.0, color="#C44E52", ls="--", lw=1.4)
        ax.set_xticks(x, names, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.18, max(2.6, max(ear) * 1.15))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("значение")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Нормализованная earliness = (T − позиция) / (T − t*)", fontsize=13)
    fig.text(0.01, 0.01,
             "1.0 (красный пунктир) = сработал ровно на онсете. Больше 1.0 означает "
             "срабатывание ДО онсета: у модели ещё нет текстовых улик обмана, "
             "то есть диалог опознаётся по стилю, а не по обманной реплике.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def tradeoff_curve(p, path):
    """Ложные тревоги на benign против latency и детекции на скамах."""
    c = p["tradeoff_curve"]
    fa = [x["false_alarm_rate"] for x in c]
    lat = [x["latency_median"] if x["latency_median"] is not None else np.nan for x in c]
    det = [x["detection_rate"] for x in c]
    thr = [x["threshold"] for x in c]
    bal = p["balance_point"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sc = axes[0].scatter(fa, lat, c=thr, cmap="viridis", s=32)
    axes[0].plot(bal["false_alarm_rate"], bal["latency_median"], "*", ms=22,
                 color="#C44E52", zorder=5)
    axes[0].annotate(f"баланс: порог {bal['threshold']:.2f}",
                     (bal["false_alarm_rate"], bal["latency_median"]),
                     textcoords="offset points", xytext=(12, 10), fontsize=9,
                     color="#C44E52")
    axes[0].set_xlabel("доля ложных тревог на benign-диалогах")
    axes[0].set_ylabel("медианная latency на скамах (реплик)")
    axes[0].set_title("Компромисс: ложные тревоги против запаздывания", fontsize=11)
    fig.colorbar(sc, ax=axes[0], label="порог уверенности")
    axes[0].grid(alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].plot(thr, det, color="#4C72B0", lw=2, label="детекция на скамах")
    axes[1].plot(thr, fa, color="#C44E52", lw=2, label="ложные тревоги на benign")
    axes[1].axvline(bal["threshold"], color="#333", ls="--", lw=1.4)
    axes[1].annotate(f"баланс {bal['threshold']:.2f}", (bal["threshold"], 0.5),
                     textcoords="offset points", xytext=(6, 0), fontsize=9)
    axes[1].set_xlabel("порог уверенности")
    axes[1].set_ylabel("доля диалогов")
    axes[1].set_title("Детекция и ложные тревоги от порога", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Кривая компромисса ранней детекции", fontsize=13)
    fig.text(0.01, 0.01, "Ложная тревога = детектор сработал хотя бы на одном префиксе "
                         "benign-диалога. " + p["checkpoint_warning"],
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def significance_forest(s, path):
    """Лес разниц F1 с доверительными интервалами по всем парам."""
    pairs = s["pairs"]
    fig, ax = plt.subplots(figsize=(11, 1.35 * len(pairs) + 2.6))
    y = np.arange(len(pairs))[::-1]
    for yi, pr in zip(y, pairs):
        for off, tag, col, lab in ((0.16, "test_only", "#4C72B0", "только тест"),
                                   (-0.16, "test_and_seed", "#C44E52", "тест + сиды")):
            lo, hi = pr[tag]["ci95"]
            ax.plot([lo, hi], [yi + off] * 2, color=col, lw=3, solid_capstyle="round")
            ax.plot(pr["observed_diff"], yi + off, "o", ms=8, color=col,
                    label=lab if yi == y[0] else None)
            ax.text(hi + 0.004, yi + off, f"[{lo:+.3f}, {hi:+.3f}]", va="center",
                    fontsize=8, color=col)
    ax.axvline(0, color="#222", lw=1.6)
    ax.set_ylim(min(y) - 0.65, max(y) + 0.8)   # запас сверху под подписи ΔF1
    ax.set_yticks(y, [f"{pr['a_ru']}\nпротив {pr['b_ru']}" for pr in pairs], fontsize=9)
    ax.set_xlabel("разница F1 (положительная = первая модель лучше)")
    ax.set_title("Значимость различий: парный бутстрап, 95% ДИ\n"
                 "ДИ пересекает ноль → разница не показана", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for yi, pr in zip(y, pairs):
        ax.text(ax.get_xlim()[0] + 0.002, yi + 0.42,
                f"ΔF1 {pr['observed_diff']:+.4f}, перестановочный p = "
                f"{pr['permutation']['p_value']:.4f}", fontsize=8, color="#444")
    fig.text(0.01, 0.01,
             f"gold-test v2, n={s['n_test']}, {s['n_boot']} реплик парного бутстрапа. "
             "«Только тест» — предсказания зафиксированы голосованием по сидам; "
             "«тест + сиды» — в каждой реплике заново тянется сид обучения.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--earliness", default="runs/earliness/results.json")
    ap.add_argument("--significance", default="runs/significance/results.json")
    a = ap.parse_args()
    e = pathlib.Path(a.earliness)
    if e.exists():
        p = json.loads(e.read_text(encoding="utf-8"))
        out = e.parent / "plots"
        out.mkdir(parents=True, exist_ok=True)
        latency_distribution(p, out / "latency_distribution.png")
        earliness_by_bin(p, out / "earliness_by_bin.png")
        tradeoff_curve(p, out / "tradeoff_curve.png")
        print(f"графики ранней детекции: {out}")
    s = pathlib.Path(a.significance)
    if s.exists():
        d = json.loads(s.read_text(encoding="utf-8"))
        out = s.parent / "plots"
        out.mkdir(parents=True, exist_ok=True)
        significance_forest(d, out / "significance_forest.png")
        print(f"график значимости: {out}")


if __name__ == "__main__":
    main()
