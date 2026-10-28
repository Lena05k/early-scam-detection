#!/usr/bin/env python3
"""Насколько шумным становится отбор чекпоинта на языковом подмножестве gold-dev.

Zero-shot требует, чтобы целевой язык не участвовал НИ В ЧЁМ, включая отбор
чекпоинта. Значит отбирать надо по gold-dev, отфильтрованному по языкам-источникам:
33–34 диалога при одном источнике, 66–67 при двух, вместо полных 100.

Считаем на ГОТОВЫХ снимках gold-dev (`runs/epoch_cost/data/xlmr/`, 3 сида × 3 эпохи):
ширину 95% бутстрап-интервала F1 при n = 33/34, 66/67, 100 и долю попаданий в ту же
эпоху, что выбрал бы полный dev. Ничего не обучается.

  python scripts/crosslingual_dev_ci.py
"""
import argparse
import glob
import itertools
import json
import pathlib

import numpy as np
from sklearn.metrics import f1_score

N_BOOT = 2000
BOOT_SEED = 20260810


def f1_of(y, p):
    return float(f1_score(y, p, pos_label=1, zero_division=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/epoch_cost/data/xlmr")
    ap.add_argument("--out", default="runs/crosslingual/dev_ci.json")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data}/gold_dev_per_epoch_seed*.json"))
    if not files:
        raise SystemExit(f"нет снимков gold-dev в {args.data}")

    rng = np.random.default_rng(BOOT_SEED)
    langs_all = ["en", "ru", "zh"]
    subsets = ({"all": langs_all}
               | {lg: [lg] for lg in langs_all}
               | {"+".join(c): list(c) for c in itertools.combinations(langs_all, 2)})

    out = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "subsets": {}}
    print(f"снимков: {len(files)} сидов, бутстрап {N_BOOT} выборок\n")
    print(f"{'подмножество dev':<16}{'n':>5}{'ширина 95% ДИ':>16}{'×к n=100':>10}"
          f"{'совпало с полным dev':>22}")
    print("-" * 69)

    base_width = None
    for name, langs in subsets.items():
        widths, hits, ns = [], [], []
        for f in files:
            d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
            lg = np.array(d["langs"])
            idx = np.flatnonzero(np.isin(lg, langs))
            ns.append(len(idx))
            per_epoch = d["per_epoch"]
            y_full = np.array(per_epoch[0]["y_true"])
            # какую эпоху выбрал бы ПОЛНЫЙ dev
            best_full = int(np.argmax([f1_of(y_full, np.array(pe["y_pred"]))
                                       for pe in per_epoch]))
            y = y_full[idx]
            preds = [np.array(pe["y_pred"])[idx] for pe in per_epoch]
            # ширина ДИ для F1 выбранной эпохи + доля совпадения выбора
            boot_f1, agree = [], 0
            for _ in range(N_BOOT):
                b = rng.integers(0, len(idx), len(idx))
                if len(set(y[b])) < 2:
                    continue
                scores = [f1_of(y[b], p[b]) for p in preds]
                boot_f1.append(scores[best_full])
                agree += int(int(np.argmax(scores)) == best_full)
            lo, hi = np.percentile(boot_f1, [2.5, 97.5])
            widths.append(float(hi - lo))
            hits.append(agree / max(1, len(boot_f1)))
        w, h, n = float(np.mean(widths)), float(np.mean(hits)), int(np.mean(ns))
        if name == "all":
            base_width = w
        out["subsets"][name] = {"n": n, "ci_width": w, "agreement_with_full_dev": h,
                                "ratio_vs_full": w / base_width if base_width else 1.0}
        print(f"{name:<16}{n:>5}{w:>16.4f}{w / base_width:>10.2f}{h:>22.3f}")

    single = [v for k, v in out["subsets"].items() if k in langs_all]
    pairs = [v for k, v in out["subsets"].items() if "+" in k]
    out["summary"] = {
        "single_source_mean_ratio": float(np.mean([v["ratio_vs_full"] for v in single])),
        "two_source_mean_ratio": float(np.mean([v["ratio_vs_full"] for v in pairs])),
        "single_source_mean_agreement": float(np.mean([v["agreement_with_full_dev"]
                                                       for v in single])),
        "two_source_mean_agreement": float(np.mean([v["agreement_with_full_dev"]
                                                    for v in pairs])),
        "sqrt_rule_single": float(np.sqrt(100 / 33)),
        "sqrt_rule_two": float(np.sqrt(100 / 66)),
    }
    s = out["summary"]
    print(f"\nодин источник (n≈33): ДИ шире в {s['single_source_mean_ratio']:.2f}× "
          f"(правило 1/√n дало бы {s['sqrt_rule_single']:.2f}×), "
          f"выбор совпадает с полным dev в {s['single_source_mean_agreement']:.1%}")
    print(f"два источника (n≈66): ДИ шире в {s['two_source_mean_ratio']:.2f}× "
          f"(правило 1/√n: {s['sqrt_rule_two']:.2f}×), "
          f"совпадение {s['two_source_mean_agreement']:.1%}")

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {p}")


if __name__ == "__main__":
    main()
