#!/usr/bin/env python3
"""Три уточнения по готовым снимкам. Ничего не обучается.

1. Смещение / OAD / exact по бинам истинного t*: {1}, {2}, {3–4}, {5+}.
   Среднее смещение по всем скамам нечитаемо: 69% из них имеют t*=1, где ошибка
   структурно неотрицательна (раньше первой реплики предсказать нельзя).
2. Сравнение модели с silver на ОБЩИХ диалогах, где silver вообще даёт онсет.
3. Бинарные метрики xlm-r без 40 спорных диалогов (hard_negative=1 при gold scam=1).

  python scripts/onset_refinements.py
"""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
        ("3–4", lambda t: (t >= 3) & (t <= 4)), ("5+", lambda t: t >= 5)]


def onset_metrics(pred, true):
    if len(pred) == 0:
        return {"n": 0}
    d = np.abs(pred - true)
    return {"n": int(len(d)), "oad": float(d.mean()), "exact": float((d == 0).mean()),
            "bias": float((pred - true).mean())}


def bin_metrics(pred, true):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(true, pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(true, pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(true, pred)), "n": int(len(true))}


def mstd(vals):
    return [float(np.mean(vals)), float(np.std(vals, ddof=1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onset-data", default="runs/onset/data")
    ap.add_argument("--binary-data", default="runs/epoch_cost/data/xlmr")
    ap.add_argument("--out", default="runs/onset")
    args = ap.parse_args()

    od = pathlib.Path(args.onset_data)
    raw = json.loads((od / "results_raw.json").read_text(encoding="utf-8"))
    chosen = {r["seed"]: r["best_epoch"] for r in raw["runs"]}
    runs = []
    for tp in sorted(od.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        i = t["epochs"].index(chosen[t["seed"]])
        runs.append({"seed": t["seed"], "meta": t["meta"],
                     "pred": np.array(t["per_epoch"][i]["pred"]),
                     "true": np.array(t["per_epoch"][i]["true"])})
    meta = runs[0]["meta"]
    true = runs[0]["true"]
    disputed = np.array([m["disputed"] for m in meta], dtype=bool)
    silver = np.array([m["t_star_silver"] if m["t_star_silver"] else -1 for m in meta])
    VARIANTS = [("а) все скамы", np.ones(len(true), dtype=bool)),
                ("б) без спорных", ~disputed)]
    out = {}

    # ───────── 1. по бинам t* ─────────
    print("1) СМЕЩЕНИЕ ПО БИНАМ ИСТИННОГО t*\n")
    out["by_bin"] = {}
    for vname, vmask in VARIANTS:
        print(f"  {vname}")
        print(f"    {'бин t*':>8} {'n':>4} {'OAD':>16} {'exact':>16} {'смещение':>18}")
        out["by_bin"][vname] = {}
        for bname, sel in BINS:
            m = vmask & sel(true)
            if not m.any():
                continue
            per = [onset_metrics(r["pred"][m], r["true"][m]) for r in runs]
            row = {k: mstd([p[k] for p in per]) for k in ("oad", "exact", "bias")}
            row["n"] = int(m.sum())
            out["by_bin"][vname][bname] = row
            print(f"    {bname:>8} {row['n']:>4} "
                  f"{row['oad'][0]:>8.4f}±{row['oad'][1]:.4f} "
                  f"{row['exact'][0]:>8.4f}±{row['exact'][1]:.4f} "
                  f"{row['bias'][0]:>+9.4f}±{row['bias'][1]:.4f}")
        print()

    # ───────── 2. сравнение с silver на общих данных ─────────
    print("2) МОДЕЛЬ ПРОТИВ SILVER НА ОБЩИХ ДИАЛОГАХ\n")
    out["vs_silver_common"] = {}
    for vname, vmask in VARIANTS:
        m = vmask & (silver > 0)
        if not m.any():
            continue
        per = [onset_metrics(r["pred"][m], r["true"][m]) for r in runs]
        model = {k: mstd([p[k] for p in per]) for k in ("oad", "exact", "bias")}
        sil = onset_metrics(silver[m], true[m])
        out["vs_silver_common"][vname] = {"n": int(m.sum()), "model": model, "silver": sil,
                                          "delta_oad": float(model["oad"][0] - sil["oad"]),
                                          "delta_exact": float(model["exact"][0] - sil["exact"])}
        print(f"  {vname}, общих диалогов n={int(m.sum())}")
        print(f"    модель: OAD {model['oad'][0]:.4f}±{model['oad'][1]:.4f}  "
              f"exact {model['exact'][0]:.4f}±{model['exact'][1]:.4f}  "
              f"смещение {model['bias'][0]:+.4f}")
        print(f"    silver: OAD {sil['oad']:.4f}  exact {sil['exact']:.4f}  "
              f"смещение {sil['bias']:+.4f}")
        print(f"    разница: OAD {model['oad'][0] - sil['oad']:+.4f}, "
              f"exact {model['exact'][0] - sil['exact']:+.4f}  "
              f"→ {'модель лучше' if model['oad'][0] < sil['oad'] else 'silver лучше'}\n")

    # ───────── 3. бинарные метрики без спорных ─────────
    print("3) БИНАРНЫЙ РЕЗУЛЬТАТ БЕЗ СПОРНЫХ\n")
    gold = {}
    with open("data/gold/gold_test_v1.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            gold[r["dialog_id"].strip()] = int(r["is_scam"])
    hard = {}
    with open("data/corpus/corpus_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            hard[x["id"]] = int(x.get("hard_negative", 0))

    bd = pathlib.Path(args.binary_data)
    braw = json.loads((bd / "results.json").read_text(encoding="utf-8"))
    bchosen = {r["seed"]: r["best_epoch"] for r in braw["runs"]}
    bruns = []
    for tp in sorted(bd.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        i = t["epochs"].index(bchosen[t["seed"]])
        bruns.append({"seed": t["seed"], "ids": t["dialog_ids"],
                      "y_true": np.array(t["per_epoch"][i]["y_true"]),
                      "y_pred": np.array(t["per_epoch"][i]["y_pred"])})
    ids = bruns[0]["ids"]
    disp_bin = np.array([hard.get(i, 0) == 1 and gold.get(i, 0) == 1 for i in ids])
    print(f"  диалогов в тесте {len(ids)}, спорных {int(disp_bin.sum())}, "
          f"остаётся {int((~disp_bin).sum())}")
    out["binary_without_disputed"] = {}
    for vname, m in (("а) все 303", np.ones(len(ids), dtype=bool)),
                     ("б) без спорных", ~disp_bin)):
        per = [bin_metrics(r["y_pred"][m], r["y_true"][m]) for r in bruns]
        row = {k: mstd([p[k] for p in per]) for k in
               ("accuracy", "precision", "recall", "f1", "cohen_kappa")}
        row["n"] = per[0]["n"]
        out["binary_without_disputed"][vname] = row
        print(f"  {vname} (n={row['n']}): " +
              "  ".join(f"{k} {row[k][0]:.4f}±{row[k][1]:.4f}"
                        for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa")))
    # Финальный predict (та же величина, что документирована как 0.9375 ± 0.0086):
    # y_pred лежит в results.json прогона. Считаем его же без спорных — это
    # сравнение «как с как», в отличие от пер-эпохных снимков.
    fin = []
    for r in braw["runs"]:
        fin.append({"seed": r["seed"], "y_pred": np.array(r["y_pred"])})
    y_true_fin = bruns[0]["y_true"]
    out["binary_final_predict"] = {}
    for vname, m in (("а) все 303", np.ones(len(ids), dtype=bool)),
                     ("б) без спорных", ~disp_bin)):
        per = [bin_metrics(f["y_pred"][m], y_true_fin[m]) for f in fin]
        row = {k: mstd([q[k] for q in per]) for k in
               ("accuracy", "precision", "recall", "f1", "cohen_kappa")}
        row["n"] = per[0]["n"]
        out["binary_final_predict"][vname] = row
    fa = out["binary_final_predict"]["а) все 303"]
    fb = out["binary_final_predict"]["б) без спорных"]
    print("\n  ФИНАЛЬНЫЙ PREDICT (документированная база 0.9375 ± 0.0086):")
    for vname, row in out["binary_final_predict"].items():
        print(f"    {vname} (n={row['n']}): " +
              "  ".join(f"{k} {row[k][0]:.4f}±{row[k][1]:.4f}"
                        for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa")))
    out["binary_final_delta"] = {}
    print("    дельта (б) − (а):")
    for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa"):
        d = fb[k][0] - fa[k][0]
        rel = d / fa[k][0] if fa[k][0] else float("nan")
        out["binary_final_delta"][k] = {"abs": float(d), "rel": float(rel)}
        print(f"      {k:>12}: {fa[k][0]:.4f} → {fb[k][0]:.4f}  ({d:+.4f}, {rel:+.1%})"
              + ("  ⚠ >3%" if abs(rel) > 0.03 else ""))
    out["binary_final_f1_drop_over_3pct"] = bool(abs(out["binary_final_delta"]["f1"]["rel"]) > 0.03)

    a = out["binary_without_disputed"]["а) все 303"]
    b = out["binary_without_disputed"]["б) без спорных"]
    print("\n  дельта (б) − (а):")
    out["binary_delta"] = {}
    for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa"):
        d = b[k][0] - a[k][0]
        rel = d / a[k][0] if a[k][0] else float("nan")
        out["binary_delta"][k] = {"abs": float(d), "rel": float(rel)}
        print(f"    {k:>12}: {a[k][0]:.4f} → {b[k][0]:.4f}  ({d:+.4f}, {rel:+.1%})"
              + ("  ⚠ >3%" if abs(rel) > 0.03 else ""))
    out["binary_f1_drop_over_3pct"] = bool(abs(out["binary_delta"]["f1"]["rel"]) > 0.03)

    # ───────── график ─────────
    plt.style.use("seaborn-v0_8-whitegrid")
    plots = pathlib.Path(args.out) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    names = [b for b, _ in BINS if b in out["by_bin"]["а) все скамы"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, (vname, c) in enumerate((("а) все скамы", "#4C72B0"),
                                    ("б) без спорных", "#937860"))):
        mu = [out["by_bin"][vname][n]["bias"][0] for n in names]
        sd = [out["by_bin"][vname][n]["bias"][1] for n in names]
        ax.bar(x + (k - 0.5) * 0.38, mu, 0.38, yerr=sd, capsize=4, color=c, label=vname)
        for xi, n in zip(x, names):
            ax.text(xi + (k - 0.5) * 0.38, 0.12 if k == 0 else -0.35,
                    f"n={out['by_bin'][vname][n]['n']}", ha="center", fontsize=9, color=c)
    ax.axhline(0, color="#333", linewidth=1.6)
    ax.set_xticks(x, names)
    ax.set_xlabel("бин истинного t*")
    ax.set_ylabel("смещение t̂* − t*  (<0 — модель спешит)")
    ax.set_title("Смещение онсета по бинам истинного t*, gold-test\n"
                 "усы — стандартное отклонение по 3 сидам; при t*=1 ошибка "
                 "структурно неотрицательна")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "bias_by_onset_bin.png", dpi=DPI)
    plt.close(fig)

    (pathlib.Path(args.out) / "refinements.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nграфик:    {(plots / 'bias_by_onset_bin.png').resolve()}")
    print(f"результат: {(pathlib.Path(args.out) / 'refinements.json').resolve()}")


if __name__ == "__main__":
    main()
