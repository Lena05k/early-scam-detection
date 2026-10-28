#!/usr/bin/env python3
"""Графики ранней детекции на восстановленных моделях (три сида, среднее ± std).

Отличие от первой версии: основная метрика — знаковая latency, earliness в главные
панели не идёт. Доля срабатываний ДО онсета показана отдельной величиной.

  python scripts/make_earliness_v2_plots.py
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


def _foot(p):
    ms = p["models"]
    return ("Восстановленные веса, три сида: " +
            ", ".join(f"{m['seed']} F1 {m['gold_test_v2_f1']:.4f}" for m in ms) +
            f" (среднее {np.mean([m['gold_test_v2_f1'] for m in ms]):.4f}). "
            "Все величины — среднее ± std по трём сидам.")


def latency_distribution(p, path):
    """Слева — распределение latency по всем сидам, справа — по бинам t*."""
    per = p["per_dialog_by_seed"]
    seeds = sorted(per)
    allv = [d["latency"] for s in seeds for d in per[s] if d["latency"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lo, hi = int(min(allv)), int(max(allv))
    axes[0].hist(allv, bins=np.arange(lo - 0.5, hi + 1.5, 1), color="#4C72B0",
                 edgecolor="white")
    axes[0].axvline(0, color="#C44E52", lw=2)
    axes[0].annotate("онсет", (0, axes[0].get_ylim()[1] * 0.93), color="#C44E52",
                     fontsize=10, xytext=(6, 0), textcoords="offset points")
    m = p["at_main_threshold"]
    axes[0].set_xlabel("latency = позиция срабатывания − t*  (реплик)")
    axes[0].set_ylabel(f"диалогов (сложено по {len(seeds)} сидам)")
    axes[0].set_title(f"Распределение latency, порог {p['main_threshold']}\n"
                      f"медиана {m['latency_median']['mean']:.2f} ± "
                      f"{m['latency_median']['std']:.2f}, среднее "
                      f"{m['latency_mean']['mean']:.3f} ± {m['latency_mean']['std']:.3f}",
                      fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    names = list(p["by_onset_bin"])
    x = np.arange(len(names))
    med = [p["by_onset_bin"][k]["latency_median"]["mean"] for k in names]
    err = [p["by_onset_bin"][k]["latency_median"]["std"] for k in names]
    axes[1].bar(x, med, 0.55, yerr=err, capsize=5,
                color=[BIN_COLOR[k] for k in names], edgecolor="white")
    for xi, (v, e, k) in enumerate(zip(med, err, names)):
        axes[1].text(xi, v - 0.28 if v < 0 else v + 0.12, f"{v:.2f}", ha="center",
                     fontsize=10)
        axes[1].text(xi, 0.42, f"n={p['by_onset_bin'][k]['n']}", ha="center",
                     fontsize=8, color="#555")
    axes[1].axhline(0, color="#C44E52", lw=2)
    # все значения ≤ 0, поэтому потолок оси надо задать вручную — иначе подписи n=
    # уезжают в заголовок
    axes[1].set_ylim(min(m - e for m, e in zip(med, err)) - 0.55, 0.75)
    axes[1].set_xticks(x, names, fontsize=11)
    axes[1].set_xlabel("бин истинного онсета t*")
    axes[1].set_ylabel("медианная latency (реплик)")
    axes[1].set_title("Чем позже онсет, тем раньше относительно него\nсрабатывает "
                      "детектор", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Через сколько реплик детектор говорит «scam»", fontsize=13)
    fig.text(0.01, 0.01, _foot(p), fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def before_onset_by_bin(p, path):
    """Доля срабатываний ДО онсета — метрика взамен earliness."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], "by_onset_bin", "по бинам истинного онсета t*"),
                           (axes[1], "by_language", "по языкам")):
        d = p[key]
        names = list(d)
        x = np.arange(len(names))
        for k, (field, col, lab) in enumerate((
                ("share_before_onset", "#C44E52", "сработал ДО онсета"),
                ("detection_rate", "#4C72B0", "доля сработавших"))):
            mu = [d[n][field]["mean"] for n in names]
            sd = [d[n][field]["std"] for n in names]
            ax.bar(x + (k - 0.5) * 0.38, mu, 0.38, yerr=sd, capsize=4, color=col,
                   label=lab, edgecolor="white")
            for xi, (v, e) in zip(x + (k - 0.5) * 0.38, zip(mu, sd)):
                ax.text(xi, v + e + 0.03, f"{v:.2f}", ha="center", fontsize=8.5)
        for xi, n in enumerate(names):
            ax.text(xi, -0.1, f"n={d[n]['n']}", ha="center", fontsize=8, color="#555")
        ax.set_xticks(x, names, fontsize=11)
        ax.set_ylim(-0.15, 1.18)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("доля диалогов")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Доля срабатываний ДО онсета — вместо earliness", fontsize=13)
    fig.text(0.01, 0.01,
             "Прежняя earliness = (T−позиция)/(T−t*) при срабатывании до онсета уходит "
             "выше 1 и перестаёт читаться как доля окна; она сохранена в results.json "
             "как earliness_median_legacy. " + _foot(p), fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def tradeoff_curve(p, path):
    c = p["tradeoff_curve"]
    thr = [x["threshold"] for x in c]
    fa = np.array([x["false_alarm_rate"]["mean"] for x in c])
    fasd = np.array([x["false_alarm_rate"]["std"] for x in c])
    det = np.array([x["detection_rate"]["mean"] for x in c])
    detsd = np.array([x["detection_rate"]["std"] for x in c])
    lat = [x["latency_median"]["mean"] if x["latency_median"] else np.nan for x in c]
    bal = p["balance_point"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sc = axes[0].scatter(fa, lat, c=thr, cmap="viridis", s=32)
    axes[0].plot(bal["false_alarm_rate"]["mean"], bal["latency_median"]["mean"], "*",
                 ms=22, color="#C44E52", zorder=5)
    axes[0].annotate(f"баланс: порог {bal['threshold']:.2f}",
                     (bal["false_alarm_rate"]["mean"], bal["latency_median"]["mean"]),
                     textcoords="offset points", xytext=(12, 10), fontsize=9,
                     color="#C44E52")
    axes[0].set_xlabel("доля ложных тревог на benign-диалогах")
    axes[0].set_ylabel("медианная latency на скамах (реплик)")
    axes[0].set_title("Компромисс: ложные тревоги против запаздывания", fontsize=11)
    fig.colorbar(sc, ax=axes[0], label="порог уверенности")
    axes[0].grid(alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].plot(thr, det, color="#4C72B0", lw=2, label="детекция на скамах")
    axes[1].fill_between(thr, det - detsd, det + detsd, color="#4C72B0", alpha=0.2)
    axes[1].plot(thr, fa, color="#C44E52", lw=2, label="ложные тревоги на benign")
    axes[1].fill_between(thr, fa - fasd, fa + fasd, color="#C44E52", alpha=0.2)
    axes[1].axvline(bal["threshold"], color="#333", ls="--", lw=1.4)
    axes[1].annotate(f"баланс {bal['threshold']:.2f}", (bal["threshold"], 0.55),
                     textcoords="offset points", xytext=(6, 0), fontsize=9)
    axes[1].set_xlabel("порог уверенности")
    axes[1].set_ylabel("доля диалогов")
    axes[1].set_title("Детекция и ложные тревоги от порога\n(полоса = ± std по сидам)",
                      fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle("Кривая компромисса ранней детекции", fontsize=13)
    fig.text(0.01, 0.01, "Ложная тревога = срабатывание хотя бы на одном префиксе "
                         "benign-диалога. " + _foot(p), fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/earliness_v2/results.json")
    a = ap.parse_args()
    p = json.loads(pathlib.Path(a.results).read_text(encoding="utf-8"))
    out = pathlib.Path(a.results).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    latency_distribution(p, out / "latency_distribution.png")
    before_onset_by_bin(p, out / "before_onset_by_bin.png")
    tradeoff_curve(p, out / "tradeoff_curve.png")
    print(f"графики: {out} (dpi={DPI})")


if __name__ == "__main__":
    main()
