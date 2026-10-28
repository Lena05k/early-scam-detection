"""Фигуры для статьи из результатов IAA-калибровки.

Каждая функция принимает готовые данные (расчёт κ/α — в iaa.harness) и путь,
сохраняет PNG (dpi=200, bbox_inches='tight') и возвращает путь строкой.
Требует matplotlib (extra [viz]); в ядро не импортировать.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# палитра: статусные и sequential-цвета валидированы (CVD ΔE 12.4, контраст >=3:1)
GOOD = "#0ca30c"          # значение >= порога
BAD = "#d03b3b"           # значение < порога
BAD_EDGE = "#7f1d1d"      # штриховка тон-в-тон (вторичное кодирование, не только цвет)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ROUND1, ROUND2 = "#86b6ef", "#256abf"   # ординальная пара из того же синего ряда
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"         # серые оси/тики
GRID = "#e1e0d9"
NEUTRAL = "#f0efec"       # клетка без значения (∅)

LABELS = {
    "is_scam": "is_scam (κ)",
    "phishing": "phishing_link (κ)",
    "money": "money_request (κ)",
    "other": "other (κ)",
    "onset": "onset t* (α)",
}

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9)


def _save(fig, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_iaa_bars(scores, thresholds, out_path):
    """Горизонтальные бары согласия по меткам; цвет — относительно порога своей метрики."""
    names = list(scores)
    vals = [scores[k] for k in names]
    fig, ax = plt.subplots(figsize=(6.0, 0.62 * len(names) + 1.0))
    ys = list(range(len(names)))[::-1]  # первая метрика сверху
    for y, k, v in zip(ys, names, vals):
        thr = thresholds[k]
        ok = v >= thr
        ax.barh(y, v, height=0.55, zorder=3,
                color=GOOD if ok else BAD,
                hatch=None if ok else "///",
                edgecolor="none" if ok else BAD_EDGE, linewidth=0.6)
        ax.plot([thr, thr], [y - 0.38, y + 0.38], color=INK, lw=1.2, zorder=4)
        pad = 0.015
        ax.text(v + pad if v >= 0 else v - pad, y, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    lo = min(0.0, min(vals) - 0.08)
    hi = max(1.0, max(vals)) + 0.10
    ax.set_xlim(lo, hi)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS.get(k, k) for k in names])
    ax.set_xlabel("согласие (κ / α)", color=SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(handles=[
        Patch(facecolor=GOOD, label="выше порога"),
        Patch(facecolor=BAD, hatch="///", edgecolor=BAD_EDGE, label="ниже порога"),
        Line2D([0], [0], color=INK, lw=1.2, label="порог метрики"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    return _save(fig, out_path)


def plot_onset_matrix(onsets, gold, out_path):
    """Тепловая матрица онсетов: строки — скам-диалоги, столбцы — аннотаторы + gold.

    benign (gold t* = ∅) пропускаются; пропуск аннотатора рисуется нейтральной клеткой '∅'.
    """
    items = [i for i in sorted(onsets) if gold.get(i) is not None]
    if not items:
        raise ValueError("нет скам-диалогов с онсетом — нечего рисовать")
    raters = sorted({w for i in items for w in onsets[i]})
    cols = raters + ["gold"]
    arr = np.full((len(items), len(cols)), np.nan)
    for r, it in enumerate(items):
        for c, w in enumerate(raters):
            v = onsets[it].get(w)
            if v is not None:
                arr[r, c] = v
        arr[r, len(raters)] = gold[it]
    vmax = max(2.0, np.nanmax(arr))
    cmap = SEQ_CMAP.with_extremes(bad=NEUTRAL)
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 1.6, 0.5 * len(items) + 0.9))
    ax.imshow(arr, cmap=cmap, vmin=1, vmax=vmax, aspect="equal")
    for r in range(len(items)):
        for c in range(len(cols)):
            v = arr[r, c]
            if np.isnan(v):
                ax.text(c, r, "∅", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                frac = (v - 1) / (vmax - 1)
                ax.text(c, r, f"{int(v)}", ha="center", va="center", fontsize=9,
                        color="white" if frac >= 0.5 else INK)
    for c in range(1, len(cols)):  # 2px-спейсеры между клетками; жирнее — перед gold
        ax.axvline(c - 0.5, color="white", lw=3.0 if c == len(raters) else 1.2)
    for r in range(1, len(items)):
        ax.axhline(r - 0.5, color="white", lw=1.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"Пр.{i}" for i in items])
    ax.tick_params(length=0, labelcolor=SECONDARY, labelsize=9)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_xlabel("онсет t* (номер реплики)", color=SECONDARY, fontsize=9)
    return _save(fig, out_path)


def pairwise_agreement(annotations, a, b):
    """Доля совпавших решений пары аннотаторов по is_scam, подтипам и онсету."""
    num = den = 0
    for it in annotations:
        recs = annotations[it]
        if a not in recs or b not in recs:
            continue
        ra, rb = recs[a], recs[b]
        checks = [ra["is_scam"] == rb["is_scam"], ra["onset"] == rb["onset"]]
        checks += [x == y for x, y in zip(ra["subs"], rb["subs"])]
        num += sum(checks)
        den += len(checks)
    return num / den if den else float("nan")


def plot_agreement_matrix(annotations, out_path):
    """N×N матрица попарного согласия аннотаторов (доля совпавших меток), в процентах."""
    raters = sorted({w for it in annotations for w in annotations[it]})
    n = len(raters)
    arr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            arr[i, j] = arr[j, i] = pairwise_agreement(annotations, raters[i], raters[j])
    fig, ax = plt.subplots(figsize=(0.85 * n + 1.6, 0.85 * n + 1.2))
    ax.imshow(arr, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            txt = "–" if np.isnan(v) else f"{100 * v:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if v >= 0.5 else INK)
    for k in range(1, n):
        ax.axvline(k - 0.5, color="white", lw=1.2)
        ax.axhline(k - 0.5, color="white", lw=1.2)
    ax.set_xticks(range(n)); ax.set_xticklabels(raters)
    ax.set_yticks(range(n)); ax.set_yticklabels(raters)
    ax.tick_params(length=0, labelcolor=SECONDARY, labelsize=9)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    return _save(fig, out_path)


def plot_before_after(scores_before, scores_after, out_path, thresholds=None):
    """Сгруппированные бары «раунд 1 vs раунд 2» по меткам, с линиями порогов."""
    names = list(scores_before)
    vb = [scores_before[k] for k in names]
    va = [scores_after[k] for k in names]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 1.2, 3.4))
    ax.bar(x - w / 2 - 0.015, vb, width=w, color=ROUND1, zorder=3, label="раунд 1")
    ax.bar(x + w / 2 + 0.015, va, width=w, color=ROUND2, zorder=3, label="раунд 2")
    for xi, v in zip(x - w / 2 - 0.015, vb):
        ax.text(xi, v + 0.02 if v >= 0 else v - 0.02, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2 + 0.015, va):
        ax.text(xi, v + 0.02 if v >= 0 else v - 0.02, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    handles_extra = []
    if thresholds:
        for xi, k in zip(x, names):
            ax.plot([xi - 0.48, xi + 0.48], [thresholds[k]] * 2,
                    color=INK, lw=1.2, ls="--", zorder=4)
        handles_extra.append(Line2D([0], [0], color=INK, lw=1.2, ls="--", label="порог метрики"))
    lo = min(0.0, min(vb + va) - 0.08)
    ax.set_ylim(lo, max(1.0, max(vb + va)) + 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(k, k) for k in names], fontsize=9)
    ax.set_ylabel("согласие (κ / α)", color=SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + handles_extra, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
    return _save(fig, out_path)
