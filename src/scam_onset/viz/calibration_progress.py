"""Фигура прогресса калибровки по раундам — из JSON-результатов run_iaa.py --save-json.

Числа не хардкодятся: значения читаются из сохранённых JSON харнесса,
пороги передаются вызывающим кодом из configs/default.yaml (calibration.freeze_thresholds).
Требует matplotlib (extra [viz]); в ядро не импортировать.
"""
import json
import pathlib

import matplotlib.pyplot as plt

from .plots import GRID, INK, MUTED, ROUND1, ROUND2, SECONDARY, _save, _style


def plot_calibration_progress(round_jsons, out_path, thresholds):
    """Линии onset_alpha и is_scam_kappa по раундам калибровки.

    round_jsons: [(label, path_to_json), ...] в порядке раундов;
    thresholds: {"kappa_is_scam": ..., "onset_alpha": ...} из configs/default.yaml.
    """
    if not round_jsons:
        raise ValueError("нет раундов — нечего рисовать")
    labels, kappas, alphas = [], [], []
    for label, path in round_jsons:
        d = json.loads(pathlib.Path(path).read_text())
        labels.append(label)
        kappas.append(d["is_scam_kappa"])
        alphas.append(d["onset_alpha"])
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 2.0, 3.6))
    series = [("is_scam (Fleiss κ)", kappas, ROUND2, "o"),
              ("onset t* (Krippendorff α)", alphas, ROUND1, "s")]
    for name, vals, color, marker in series:
        ax.plot(x, vals, color=color, marker=marker, lw=1.8, ms=6, zorder=3, label=name)
        for xi, v in zip(x, vals):
            if v is None:
                continue
            ax.annotate(f"{round(v, 2)}", (xi, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=INK, zorder=4)
    thr_lines = [("kappa_is_scam", "порог κ"), ("onset_alpha", "порог α")]
    for key, name in thr_lines:
        thr = thresholds[key]
        ax.axhline(thr, color=MUTED, lw=1.1, ls="--", zorder=2)
        ax.annotate(f"{name} = {thr:.2f}", (x[-1] + 0.38, thr), ha="left", va="center",
                    fontsize=8, color=SECONDARY, annotation_clip=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("согласие (κ / α)", color=SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("Согласие аннотаторов по раундам калибровки",
                 color=INK, fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9)
    return _save(fig, out_path)
