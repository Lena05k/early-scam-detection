#!/usr/bin/env python3
"""Метрики ранней детекции на ВОССТАНОВЛЕННЫХ моделях (три сида, отбор по gold-dev).

Заменяет первую версию раздела, посчитанную на чужом чекпоинте (эпоха 3 первого
baseline, отбор по silver-val, gold-test F1 0.8680). Здесь используются веса,
восстановленные `notebooks/retrain_save_kaggle.ipynb`, и всё отчитывается средним ±
std по трём сидам.

Изменение метрики (задача 1e): **основная величина — знаковая latency**. Прежняя
earliness = (T − позиция)/(T − t*) при срабатывании ДО онсета уходит выше 1 и
перестаёт читаться как «доля окна»; она сохраняется в results.json, но в основную
таблицу не идёт. Вместо неё — доля срабатываний до онсета, отдельно по бинам.

  python scripts/earliness_v2.py
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import load_all  # noqa: E402

BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
        ("3–4", lambda t: 3 <= t <= 4), ("5+", lambda t: t >= 5)]
MAIN_THR = 0.5


def first_fire(probs, thr):
    for i, p in enumerate(probs, 1):
        if p >= thr:
            return i
    return None


def stats_one_seed(rows, thr):
    """Метрики одного сида на наборе скам-диалогов."""
    lat, ear, per = [], [], []
    for r in rows:
        pos = first_fire(r["probs"], thr)
        if pos is None:
            per.append({"id": r["id"], "lang": r["lang"], "onset": r["onset"],
                        "T": r["T"], "pos": None, "latency": None, "earliness": None})
            continue
        lt = pos - r["onset"]
        den = r["T"] - r["onset"]
        er = (r["T"] - pos) / den if den > 0 else None
        lat.append(lt)
        if er is not None:
            ear.append(er)
        per.append({"id": r["id"], "lang": r["lang"], "onset": r["onset"], "T": r["T"],
                    "pos": pos, "latency": lt, "earliness": er})
    a = np.array(lat, dtype=float)
    return {
        "n": len(rows), "n_fired": len(lat),
        "detection_rate": len(lat) / len(rows) if rows else 0.0,
        "latency_median": float(np.median(a)) if len(a) else None,
        "latency_mean": float(a.mean()) if len(a) else None,
        "latency_p25": float(np.percentile(a, 25)) if len(a) else None,
        "latency_p75": float(np.percentile(a, 75)) if len(a) else None,
        "share_before_onset": float((a < 0).mean()) if len(a) else None,
        "share_at_onset": float((a == 0).mean()) if len(a) else None,
        "share_after_onset": float((a > 0).mean()) if len(a) else None,
        # старая метрика: оставлена для сравнимости с первой версией раздела
        "earliness_median_legacy": float(np.median(ear)) if ear else None,
        "per_dialog": per,
    }


def agg_seeds(per_seed, keys):
    """Среднее ± std по сидам для набора скалярных ключей."""
    out = {}
    for k in keys:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "values": [float(v) for v in vals]}
    return out


SCALARS = ["detection_rate", "latency_median", "latency_mean", "latency_p25",
           "latency_p75", "share_before_onset", "share_at_onset", "share_after_onset",
           "earliness_median_legacy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", default="runs/earliness_v2/prefix_probs_by_seed.json")
    ap.add_argument("--retrain", default="runs/earliness_v2/retrain_results.json")
    ap.add_argument("--out-dir", default="runs/earliness_v2")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_by_seed = json.loads(pathlib.Path(args.probs).read_text(encoding="utf-8"))
    retrain = json.loads(pathlib.Path(args.retrain).read_text(encoding="utf-8"))
    seeds = sorted(probs_by_seed)
    print(f"сидов: {seeds}")
    for r in retrain["runs"]:
        print(f"  сид {r['seed']}: эпоха {r['best_epoch']}, gold-test v2 F1 "
              f"{r['test_metrics']['f1']:.4f}, эталон v1 {r['reference_v1']:.4f}, "
              f"|Δ| {r['abs_diff_vs_reference']:.4f} → "
              f"{'воспроизвелось' if r['reproduced'] else 'РАСХОЖДЕНИЕ'}")
    if not all(r["reproduced"] for r in retrain["runs"]):
        print("\n⚠ не все сиды воспроизвелись — см. числа выше")

    data = load_all(verbose=False)
    test = data["test"]
    gold_onset = {}
    with open("data/gold/gold_test_v2.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            o = (row.get("onset") or "").strip()
            gold_onset[row["dialog_id"]] = int(o) if o else None

    def rows_for(seed, only_scam=True):
        pr = probs_by_seed[seed]
        out = []
        for r in test:
            is_scam = r["gold_is_scam"] == 1
            if only_scam and not is_scam:
                continue
            if not only_scam and is_scam:
                continue
            out.append({"id": r["id"], "lang": r["lang"], "T": len(r["turns"]),
                        "probs": pr[r["id"]], "onset": gold_onset.get(r["id"])})
        return out

    scam = {s: [r for r in rows_for(s) if r["onset"]] for s in seeds}
    benign = {s: rows_for(s, only_scam=False) for s in seeds}
    print(f"\nскам с онсетом {len(scam[seeds[0]])}, benign {len(benign[seeds[0]])}")

    # ── основная таблица ──
    per_seed = [stats_one_seed(scam[s], MAIN_THR) for s in seeds]
    main = agg_seeds(per_seed, SCALARS)
    print(f"\nпри пороге {MAIN_THR}, среднее ± std по {len(seeds)} сидам:")
    for k in ("detection_rate", "latency_median", "latency_mean", "share_before_onset",
              "share_at_onset", "share_after_onset"):
        v = main[k]
        print(f"  {k:<20} {v['mean']:>8.4f} ± {v['std']:.4f}   значения "
              f"{[round(x, 3) for x in v['values']]}")

    # ── по бинам t* и по языкам ──
    by_bin = {}
    for name, pred in BINS:
        ps = [stats_one_seed([r for r in scam[s] if pred(r["onset"])], MAIN_THR)
              for s in seeds]
        if ps[0]["n"]:
            by_bin[name] = {"n": ps[0]["n"], **agg_seeds(ps, SCALARS)}
    by_lang = {}
    for lg in ("en", "ru", "zh"):
        ps = [stats_one_seed([r for r in scam[s] if r["lang"] == lg], MAIN_THR)
              for s in seeds]
        if ps[0]["n"]:
            by_lang[lg] = {"n": ps[0]["n"], **agg_seeds(ps, SCALARS)}

    print("\nпо бинам t* (порог 0.5):")
    print(f"  {'бин':<5}{'n':>5}{'детекция':>18}{'latency медиана':>22}"
          f"{'сработал ДО онсета':>24}")
    for k, v in by_bin.items():
        print(f"  {k:<5}{v['n']:>5}"
              f"{v['detection_rate']['mean']:>11.3f}±{v['detection_rate']['std']:.3f}"
              f"{v['latency_median']['mean']:>15.2f}±{v['latency_median']['std']:.2f}"
              f"{v['share_before_onset']['mean']:>17.3f}±{v['share_before_onset']['std']:.3f}")
    print("по языкам:")
    for k, v in by_lang.items():
        print(f"  {k:<5}{v['n']:>5}"
              f"{v['detection_rate']['mean']:>11.3f}±{v['detection_rate']['std']:.3f}"
              f"{v['latency_median']['mean']:>15.2f}±{v['latency_median']['std']:.2f}"
              f"{v['share_before_onset']['mean']:>17.3f}±{v['share_before_onset']['std']:.3f}")

    # ── кривая компромисса ──
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    curve = []
    for thr in grid:
        ps = [stats_one_seed(scam[s], thr) for s in seeds]
        fa = [float(np.mean([first_fire(r["probs"], thr) is not None
                             for r in benign[s]])) for s in seeds]
        curve.append({
            "threshold": float(thr),
            "false_alarm_rate": {"mean": float(np.mean(fa)),
                                 "std": float(np.std(fa, ddof=1))},
            "detection_rate": agg_seeds(ps, ["detection_rate"])["detection_rate"],
            "latency_median": agg_seeds(ps, ["latency_median"]).get("latency_median"),
        })
    bal = max(curve, key=lambda c: c["detection_rate"]["mean"] - c["false_alarm_rate"]["mean"])
    print(f"\nточка баланса: порог {bal['threshold']:.2f}, детекция "
          f"{bal['detection_rate']['mean']:.3f}, ложные тревоги "
          f"{bal['false_alarm_rate']['mean']:.3f}, latency медиана "
          f"{bal['latency_median']['mean'] if bal['latency_median'] else None}")

    payload = {
        "source": "восстановленные веса, notebooks/retrain_save_kaggle.ipynb",
        "models": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                    "gold_test_v2_f1": r["test_metrics"]["f1"],
                    "reference_v1": r["reference_v1"],
                    "reproduced": r["reproduced"]} for r in retrain["runs"]],
        "gold": "data/gold/gold_test_v2.csv", "seeds": seeds,
        "main_threshold": MAIN_THR,
        "n_scam_with_onset": len(scam[seeds[0]]), "n_benign": len(benign[seeds[0]]),
        "metric_change": ("основная метрика — знаковая latency; earliness сохранена как "
                          "earliness_median_legacy, но в основную таблицу не идёт: при "
                          "срабатывании до онсета она уходит выше 1"),
        "at_main_threshold": main,
        "by_onset_bin": by_bin, "by_language": by_lang,
        "tradeoff_curve": curve, "balance_point": bal,
        "per_dialog_by_seed": {s: stats_one_seed(scam[s], MAIN_THR)["per_dialog"]
                               for s in seeds},
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
