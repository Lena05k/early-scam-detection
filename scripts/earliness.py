#!/usr/bin/env python3
"""Метрики ранней детекции (tech_report §6): latency, earliness, кривая компромисса.

OAD измеряет, НАСКОЛЬКО модель промахнулась мимо онсета. Он ничего не говорит о том,
через сколько реплик детектор реально сработает, если подавать диалог по одной
реплике — а это и есть операционная величина ранней детекции.

Здесь бинарный классификатор прогоняется по ПРЕФИКСАМ каждого диалога (1, 2, … T
реплик) и фиксируется первая позиция, где P(scam) превышает порог.
  latency   = позиция срабатывания − t*   (отрицательная = сработал до онсета)
  earliness = (T − позиция) / (T − t*)    — доля оставшегося окна, 1.0 = мгновенно

⚠ ЧЕКПОИНТ. В репозитории сохранён единственный чекпоинт — `runs/binary_xlmr/best/`,
и это эпоха 3 ПЕРВОГО baseline, отобранная по silver-val (gold-test F1 0.8680 на 303).
Модель с 0.9344 обучалась с `save_total_limit=1`, а поздние прогоны — с
`save_strategy="no"`, её веса не сохранились. Поэтому все числа этого раздела
относятся к более слабой модели и НЕ переносятся на headline-результат.

  python scripts/earliness.py --smoke      # 12 диалогов, проверить скорость
  python scripts/earliness.py
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import load_all  # noqa: E402

CKPT = "runs/binary_xlmr/best"
MAX_LENGTH = 512
BATCH = 32
BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
        ("3–4", lambda t: 3 <= t <= 4), ("5+", lambda t: t >= 5)]


def prefix_text(rec, upto):
    """Первые upto реплик в том же формате, что при обучении."""
    return "\n".join(f"[{t['i']}|{t['speaker']}] {t['text']}"
                     for t in rec["turns"][:upto])


def run_inference(recs, ckpt=CKPT, batch=BATCH, device=None):
    """P(scam) для каждого префикса каждого диалога → {id: [p1, …, pT]}."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    jobs = [(r["id"], k, prefix_text(r, k))
            for r in recs for k in range(1, len(r["turns"]) + 1)]
    # сортировка по длине — динамический паддинг тогда режет вычисления втрое
    jobs.sort(key=lambda j: len(j[2]))
    print(f"префиксов: {len(jobs)} на {len(recs)} диалогов, устройство {device}")

    out, t0 = {}, time.time()
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            enc = tok([c[2] for c in chunk], truncation=True, max_length=MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].float().cpu().numpy()
            for (did, k, _), p in zip(chunk, probs):
                out.setdefault(did, {})[k] = float(p)
            if (i // batch) % 20 == 0:
                done = i + len(chunk)
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el:.0f} c  "
                      f"осталось ~{el / max(1, done) * (len(jobs) - done):.0f} c", flush=True)
    print(f"инференс занял {(time.time() - t0) / 60:.1f} мин")
    return {did: [v[k] for k in sorted(v)] for did, v in out.items()}


# ─────────────────────────── метрики ранней детекции ───────────────────────────
def first_fire(probs, thr):
    """Первая позиция (1-индексная), где P(scam) ≥ thr; None — не сработал."""
    for i, p in enumerate(probs, 1):
        if p >= thr:
            return i
    return None


def latency_stats(rows, thr):
    """rows: [{probs, onset, T, lang}] по скам-диалогам. Latency и earliness."""
    lat, ear, fired = [], [], 0
    per = []
    for r in rows:
        pos = first_fire(r["probs"], thr)
        if pos is None:
            per.append({**{k: r[k] for k in ("id", "lang", "onset", "T")},
                        "pos": None, "latency": None, "earliness": None})
            continue
        fired += 1
        lt = pos - r["onset"]
        denom = r["T"] - r["onset"]
        # earliness не определена, когда онсет на последней реплике: окна нет
        er = (r["T"] - pos) / denom if denom > 0 else None
        lat.append(lt)
        if er is not None:
            ear.append(er)
        per.append({**{k: r[k] for k in ("id", "lang", "onset", "T")},
                    "pos": pos, "latency": lt, "earliness": er})
    n = len(rows)
    return {"n": n, "n_fired": fired, "detection_rate": fired / n if n else 0.0,
            "latency_median": float(np.median(lat)) if lat else None,
            "latency_mean": float(np.mean(lat)) if lat else None,
            "latency_p25": float(np.percentile(lat, 25)) if lat else None,
            "latency_p75": float(np.percentile(lat, 75)) if lat else None,
            "share_before_onset": float(np.mean(np.array(lat) < 0)) if lat else None,
            "share_at_onset": float(np.mean(np.array(lat) == 0)) if lat else None,
            "earliness_median": float(np.median(ear)) if ear else None,
            "earliness_mean": float(np.mean(ear)) if ear else None,
            "per_dialog": per}


def false_alarm_rate(benign_rows, thr):
    """Доля benign-диалогов, где детектор сработал хотя бы на одном префиксе."""
    if not benign_rows:
        return 0.0
    return float(np.mean([first_fire(r["probs"], thr) is not None for r in benign_rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/earliness")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--smoke", action="store_true", help="12 диалогов — замерить скорость")
    ap.add_argument("--recompute", action="store_true", help="игнорировать кэш вероятностей")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    test = data["test"]
    if args.smoke:
        test = test[:12]

    cache = out_dir / ("prefix_probs_smoke.json" if args.smoke else "prefix_probs.json")
    if cache.exists() and not args.recompute:
        probs = json.loads(cache.read_text(encoding="utf-8"))
        print(f"вероятности из кэша: {cache} ({len(probs)} диалогов)")
    else:
        probs = run_inference(test, ckpt=args.ckpt)
        cache.write_text(json.dumps(probs, ensure_ascii=False), encoding="utf-8")
        print(f"сохранено: {cache}")
    if args.smoke:
        n_pref = sum(len(v) for v in probs.values())
        print(f"смоук: {len(probs)} диалогов, {n_pref} префиксов")
        return

    # ── подготовка строк ──
    gold_onset = {}
    import csv
    with open("data/gold/gold_test_v2.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            o = (row.get("onset") or "").strip()
            gold_onset[row["dialog_id"]] = int(o) if o else None
    scam, benign = [], []
    for r in test:
        row = {"id": r["id"], "lang": r["lang"], "T": len(r["turns"]),
               "probs": probs[r["id"]], "onset": gold_onset.get(r["id"])}
        (scam if r["gold_is_scam"] == 1 else benign).append(row)
    scam_with_onset = [r for r in scam if r["onset"]]
    print(f"\nскам {len(scam)} (с онсетом {len(scam_with_onset)}), benign {len(benign)}")

    # ── кривая компромисса ──
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    curve = []
    for thr in grid:
        st = latency_stats(scam_with_onset, thr)
        curve.append({"threshold": float(thr), "false_alarm_rate":
                      false_alarm_rate(benign, thr),
                      "detection_rate": st["detection_rate"],
                      "latency_median": st["latency_median"],
                      "earliness_median": st["earliness_median"]})
    # точка баланса: максимум (детекция на скамах − ложные тревоги на benign)
    bal = max(curve, key=lambda c: c["detection_rate"] - c["false_alarm_rate"])
    print(f"\nточка баланса: порог {bal['threshold']:.2f}, детекция "
          f"{bal['detection_rate']:.3f}, ложные тревоги {bal['false_alarm_rate']:.3f}, "
          f"медианная latency {bal['latency_median']}")

    main_thr = 0.5
    base = latency_stats(scam_with_onset, main_thr)
    print(f"\nпри пороге {main_thr}: детекция {base['detection_rate']:.3f}, "
          f"latency медиана {base['latency_median']}, среднее "
          f"{base['latency_mean']:.3f}, earliness медиана {base['earliness_median']:.3f}")
    print(f"  сработал ДО онсета: {base['share_before_onset']:.3f}, "
          f"ровно на онсете: {base['share_at_onset']:.3f}")

    by_bin, by_lang = {}, {}
    for name, pred in BINS:
        rows = [r for r in scam_with_onset if pred(r["onset"])]
        if rows:
            by_bin[name] = {k: v for k, v in latency_stats(rows, main_thr).items()
                            if k != "per_dialog"}
    for lg in ("en", "ru", "zh"):
        rows = [r for r in scam_with_onset if r["lang"] == lg]
        if rows:
            by_lang[lg] = {k: v for k, v in latency_stats(rows, main_thr).items()
                           if k != "per_dialog"}
    print(f"\nпо бинам t* (порог {main_thr}):")
    for k, v in by_bin.items():
        print(f"  t*={k:<4} n={v['n']:>3} детекция {v['detection_rate']:.3f}  "
              f"latency медиана {v['latency_median']}  earliness "
              f"{v['earliness_median'] if v['earliness_median'] is None else round(v['earliness_median'], 3)}")
    print("по языкам:")
    for k, v in by_lang.items():
        print(f"  {k}: n={v['n']:>3} детекция {v['detection_rate']:.3f}  "
              f"latency медиана {v['latency_median']}  earliness "
              f"{v['earliness_median'] if v['earliness_median'] is None else round(v['earliness_median'], 3)}")

    payload = {
        "checkpoint": args.ckpt,
        "checkpoint_warning": ("эпоха 3 первого baseline, отбор по silver-val, gold-test "
                               "F1 0.8680 на 303 — НЕ модель с 0.9344, её веса не "
                               "сохранялись"),
        "gold": "data/gold/gold_test_v2.csv",
        "n_scam": len(scam), "n_scam_with_onset": len(scam_with_onset),
        "n_benign": len(benign), "main_threshold": main_thr,
        "definitions": {
            "latency": "позиция первого срабатывания − t* (отрицательная = до онсета)",
            "earliness": "(T − позиция) / (T − t*); не определена при t* = T",
            "false_alarm": "доля benign, где детектор сработал хотя бы на одном префиксе"},
        "at_main_threshold": {k: v for k, v in base.items() if k != "per_dialog"},
        "per_dialog": base["per_dialog"],
        "by_onset_bin": by_bin, "by_language": by_lang,
        "tradeoff_curve": curve, "balance_point": bal,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
