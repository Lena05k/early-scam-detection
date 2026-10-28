#!/usr/bin/env python3
"""Фигуры для статьи из IAA-разметки. Расчёт κ/α — только через iaa.harness."""
import sys, argparse, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from scam_onset.iaa.harness import (
    SUBS, load_gold, load_annotations, kappa_for, krippendorff_alpha_interval,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    from scam_onset.viz import plots
except ImportError:
    sys.exit("Нужен matplotlib: pip install -e '.[viz]'")

SUB_KEYS = ["phishing", "money", "other"]  # порядок = SUBS харнесса


def compute_scores(ann):
    """κ по is_scam/подтипам и α по онсету — теми же функциями, что и IAA-отчёт."""
    s = {"is_scam": kappa_for(ann, lambda r: r["is_scam"], categories=[0, 1])}
    for key, idx in zip(SUB_KEYS, range(len(SUBS))):
        s[key] = kappa_for(ann, lambda r, i=idx: r["subs"][i], categories=[0, 1])
    co_scam = [it for it in ann
               if all(ann[it][w]["is_scam"] == 1 and ann[it][w]["onset"] is not None
                      for w in ann[it])]
    s["onset"] = krippendorff_alpha_interval(
        [[ann[it][w]["onset"] for w in ann[it]] for it in co_scam])
    return s


def load_thresholds(config_path):
    ft = yaml.safe_load(pathlib.Path(config_path).read_text())["calibration"]["freeze_thresholds"]
    sub = ft["kappa_subtype_macro"]
    return {"is_scam": ft["kappa_is_scam"], "phishing": sub, "money": sub, "other": sub,
            "onset": ft["onset_alpha"]}


def main():
    ap = argparse.ArgumentParser(description="Фигуры IAA-калибровки для статьи")
    ap.add_argument("--gold", default="data/annotations/calibration_items_gold.csv")
    ap.add_argument("--annotations", default="data/annotations/annotator_demo.csv")
    ap.add_argument("--before", help="разметка раунда 1 (для фигуры before/after)")
    ap.add_argument("--after", help="разметка раунда 2 (для фигуры before/after)")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--out-dir", default="reports/figures")
    a = ap.parse_args()

    gold = load_gold(a.gold)
    ann = load_annotations(a.annotations)
    thresholds = load_thresholds(a.config)
    out = pathlib.Path(a.out_dir)
    saved = [plots.plot_iaa_bars(compute_scores(ann), thresholds, out / "iaa_bars.png")]

    scam_items = [it for it in sorted(gold) if gold[it]["is_scam"] == 1 and it in ann]
    onsets = {it: {w: ann[it][w]["onset"] for w in ann[it]} for it in scam_items}
    gold_onset = {it: gold[it]["onset"] for it in scam_items}
    saved.append(plots.plot_onset_matrix(onsets, gold_onset, out / "onset_matrix.png"))

    saved.append(plots.plot_agreement_matrix(ann, out / "agreement_matrix.png"))

    if a.before and a.after:
        saved.append(plots.plot_before_after(
            compute_scores(load_annotations(a.before)),
            compute_scores(load_annotations(a.after)),
            out / "before_after.png", thresholds=thresholds))
    elif a.before or a.after:
        print("⚠ для before/after нужны оба файла (--before и --after) — фигура пропущена",
              file=sys.stderr)

    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
