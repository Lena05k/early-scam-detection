"""Publication figures for the PIERE-2026 paper.

English-only labels (Cyrillic is prohibited in the submission), IEEE single-column
width (3.5 in), serif faces, 600 dpi PNG plus vector EPS/PDF.

Numbers are read from the run artifacts; nothing is hard-coded except axis text.
    runs/earliness_v2/results.json -> Fig. 1
    runs/significance/results.json -> Fig. 2

Usage: .venv/bin/python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "paper"
COL = 3.3  # PIERE template column width: (11906-907-907-360)/2 twips = 3.38 in

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

INK = "#1a1a1a"
FILL = "#c9c9c9"
ACCENT = "#4d4d4d"


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 600), ("pdf", 600), ("eps", 600)):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT / stem}.{{png,pdf,eps}}")


def fig1_latency() -> None:
    d = json.loads((ROOT / "runs/earliness_v2/results.json").read_text())
    bins = ["1", "2", "3–4", "5+"]
    labels = ["1", "2", "3-4", "5+"]  # en dash avoided in the figure text
    ns = [d["by_onset_bin"][b]["n"] for b in bins]
    med = [d["by_onset_bin"][b]["latency_median"]["mean"] for b in bins]
    seeds = [d["by_onset_bin"][b]["latency_median"]["values"] for b in bins]
    before = [d["by_onset_bin"][b]["share_before_onset"]["mean"] for b in bins]
    before_sd = [d["by_onset_bin"][b]["share_before_onset"]["std"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL, 3.0), sharex=True)
    x = range(len(bins))

    ax1.axhline(0, color=INK, lw=0.6)
    ax1.bar(x, med, width=0.55, color=FILL, edgecolor=INK, linewidth=0.6)
    for xi, vals in zip(x, seeds):
        ax1.plot([xi] * len(vals), vals, "o", ms=2.5, color=ACCENT, zorder=3)
    ax1.set_ylabel("Median latency (turns)")
    ax1.set_ylim(-4.9, 1.0)
    for xi, v in zip(x, med):
        ax1.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, -9 if v < 0 else 4),
            ha="center",
            fontsize=7,
        )
    ax1.text(0.02, 0.06, "(a)", transform=ax1.transAxes, fontsize=7)

    ax2.bar(
        x,
        before,
        yerr=before_sd,
        width=0.55,
        color=FILL,
        edgecolor=INK,
        linewidth=0.6,
        error_kw={"elinewidth": 0.6, "capsize": 2, "ecolor": INK},
    )
    ax2.set_ylabel("Flagged before onset")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("True onset turn")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax2.text(0.02, 0.86, "(b)", transform=ax2.transAxes, fontsize=7)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.align_ylabels([ax1, ax2])
    save(fig, "fig1_latency_by_onset_bin")


def fig2_significance() -> None:
    d = json.loads((ROOT / "runs/significance/results.json").read_text())
    names = {
        ("xlmr", "keywords"): "Cross-lingual vs. lexicon",
        ("xlmr", "mdeberta"): "Cross-lingual vs. disentangled",
        ("xlmr", "layer2"): "Cross-lingual vs. embeddings",
        ("xlmr", "tfidf"): "Cross-lingual vs. n-grams",
        ("keywords", "tfidf"): "Lexicon vs. n-grams",
    }
    rows = []
    for p in d["pairs"]:
        key = (p["a"], p["b"])
        if key in names:
            rows.append(
                (
                    names[key],
                    p["observed_diff"],
                    p["test_only"]["ci95"],
                    p["test_and_seed"]["ci95"],
                    p["permutation"]["p_value"],
                )
            )
    rows.reverse()

    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ax.axvline(0, color=INK, lw=0.6, ls="--")
    for i, (lab, diff, ci_t, ci_ts, pv) in enumerate(rows):
        ax.plot(ci_ts, [i, i], color=ACCENT, lw=0.9, solid_capstyle="butt")
        ax.plot(ci_t, [i, i], color=INK, lw=2.0, solid_capstyle="butt")
        ax.plot([diff], [i], "o", ms=3.5, color=INK, zorder=3)
        ax.annotate(
            f"p = {pv:.4f}",
            (0.128, i),
            fontsize=7,
            va="center",
            annotation_clip=False,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Difference in F1 score")
    ax.set_xlim(-0.06, 0.125)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], color=INK, lw=2.0, label="95% CI, dialogues")
    ax.plot([], [], color=ACCENT, lw=0.9, label="95% CI, dialogues and seeds")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.1,
    )
    save(fig, "fig2_significance_forest")


if __name__ == "__main__":
    fig1_latency()
    fig2_significance()
