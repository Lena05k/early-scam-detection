#!/usr/bin/env python3
"""Применение ручного разбора 51 спорного hard negative и оценка последствий.

Спорные — диалоги с корпусным `hard_negative=1` при gold `is_scam=1`. Решения
человека лежат в data/gold/hardneg_review_v1.csv (dialog_id, is_scam_final).

Строит data/gold/gold_test_v2.csv (v1 + решения; при is_scam_final=0 обнуляются
subtypes и onset) и пересчитывает по ГОТОВЫМ снимкам метрики трёх задач против
v1 и против v2. Ничего не обучается.

gold_test_v1.csv и hardneg_review_v1.csv только читаются.

  python scripts/apply_hardneg_review.py
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
SUB = ["phishing_link", "money_request", "other"]
BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
        ("3–4", lambda t: (t >= 3) & (t <= 4)), ("5+", lambda t: t >= 5)]


def mstd(v):
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]


def bin_metrics(pred, true):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(true, pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(true, pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(true, pred))}


def ml_metrics(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score, f1_score, hamming_loss
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_kappa": float(np.mean([cohen_kappa_score(y_true[:, i], y_pred[:, i])
                                          for i in range(y_true.shape[1])])),
            "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
            "hamming_loss": float(hamming_loss(y_true, y_pred))}


def onset_metrics(pred, true):
    d = np.abs(pred - true)
    return {"oad": float(d.mean()), "exact": float((d == 0).mean()),
            "within1": float((d <= 1).mean()), "bias": float((pred - true).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", default="data/gold/hardneg_review_v1.csv")
    ap.add_argument("--gold-v1", default="data/gold/gold_test_v1.csv")
    ap.add_argument("--gold-v2", default="data/gold/gold_test_v2.csv")
    ap.add_argument("--out", default="runs/gold_v2")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    imp = {}

    # ── 1. проверка файла решений ──
    rev = {}
    with open(args.review, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rev[r["dialog_id"].strip()] = int(r["is_scam_final"])
    disputed_ref = set()
    with open("reports/gold_vs_draft_test_disagreements.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("llm_hard_negative") == "1" and r.get("human_is_scam") == "1":
                disputed_ref.add(r["dialog_id"])
    if len(rev) != 51 or set(rev) != disputed_ref:
        print(f"❌ расхождение: в решениях {len(rev)}, в списке спорных {len(disputed_ref)}")
        print("  только в решениях:", sorted(set(rev) - disputed_ref)[:10])
        print("  только в списке:  ", sorted(disputed_ref - set(rev))[:10])
        sys.exit(1)
    print(f"1) решения: {len(rev)} строк, id уникальны, множество совпадает со списком спорных ✅")

    # ── 2. разбивка решений ──
    corp = {}
    with open("data/corpus/corpus_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            corp[x["id"]] = x
    assign = json.loads(pathlib.Path("data/gold/gold_dev_split.json").read_text(
        encoding="utf-8"))["assignment"]

    keep = sum(1 for v in rev.values() if v == 1)
    flip = sum(1 for v in rev.values() if v == 0)
    print(f"\n2) подтверждено scam: {keep}, изменено на benign: {flip}")
    by = {"lang": collections.Counter(), "scam_type": collections.Counter(),
          "part": collections.Counter()}
    flips = []
    for did, v in rev.items():
        c = corp[did]
        by["lang"][(c["lang"], v)] += 1
        by["scam_type"][(c["scam_type"], v)] += 1
        by["part"][(assign[did], v)] += 1
        if v == 0:
            flips.append({"id": did, "lang": c["lang"], "scam_type": c["scam_type"],
                          "part": assign[did]})
    for key, title in (("lang", "по языкам"), ("scam_type", "по сценариям"),
                       ("part", "по частям (dev/test)")):
        print(f"  {title}:")
        names = sorted({k for k, _ in by[key]})
        for nm in names:
            k1, k0 = by[key][(nm, 1)], by[key][(nm, 0)]
            print(f"    {nm:22s} подтверждено {k1:3d}, изменено {k0:2d}")
    print("  изменённые на benign:")
    for f_ in flips:
        print(f"    {f_['id']} ({f_['lang']}, {f_['scam_type']}, {f_['part']})")
    imp["decisions"] = {
        "n": len(rev), "confirmed_scam": keep, "flipped_to_benign": flip,
        "by_lang": {f"{k[0]}|{k[1]}": v for k, v in by["lang"].items()},
        "by_scam_type": {f"{k[0]}|{k[1]}": v for k, v in by["scam_type"].items()},
        "by_part": {f"{k[0]}|{k[1]}": v for k, v in by["part"].items()},
        "flipped": flips}

    # ── 3. gold_test_v2.csv ──
    with open(args.gold_v1, encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames)
        rows = list(rdr)
    cleared = 0
    for r in rows:
        did = r["dialog_id"].strip()
        if did in rev:
            r["label_source"] = "review"
            new = rev[did]
            if new == 0 and int(r["is_scam"]) == 1:
                if r.get("subtypes") or r.get("onset"):
                    cleared += 1
                r["subtypes"] = ""
                r["onset"] = ""
            r["is_scam"] = str(new)
        else:
            r["label_source"] = "v1"
    with open(args.gold_v2, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["label_source"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n3) {args.gold_v2}: {len(rows)} строк, "
          f"label_source=review у {sum(1 for r in rows if r['label_source'] == 'review')}, "
          f"обнулены subtypes/onset у {cleared} диалогов")
    imp["gold_v2"] = {"n_rows": len(rows), "n_review": len(rev), "cleared_subtypes_onset": cleared}

    def labels(ver):
        src = args.gold_v1 if ver == "v1" else args.gold_v2
        out = {}
        with open(src, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                o = (r.get("onset") or "").strip()
                out[r["dialog_id"].strip()] = {
                    "is_scam": int(r["is_scam"]),
                    "subs": [s for s in (r.get("subtypes") or "").split("|") if s],
                    "onset": int(o) if o else None}
        return out
    L = {"v1": labels("v1"), "v2": labels("v2")}

    # ── 4. пересчёт по снимкам ──
    print("\n4) ПЕРЕСЧЁТ ПО ГОТОВЫМ СНИМКАМ (без обучения)\n")
    imp["metrics"] = {}

    # 4a. бинарная
    bd = pathlib.Path("runs/epoch_cost/data/xlmr")
    braw = json.loads((bd / "results.json").read_text(encoding="utf-8"))
    bpred = {r["seed"]: np.array(r["y_pred"]) for r in braw["runs"]}
    ids_bin = json.loads((bd / "gold_test_per_epoch_seed20260721.json").read_text(
        encoding="utf-8"))["dialog_ids"]
    res_b = {}
    for ver in ("v1", "v2"):
        yt = np.array([L[ver][i]["is_scam"] for i in ids_bin])
        per = [bin_metrics(bpred[s], yt) for s in sorted(bpred)]
        res_b[ver] = {k: mstd([p[k] for p in per]) for k in per[0]}
    imp["metrics"]["binary"] = res_b
    print("  бинарная (xlm-r, финальный predict, n=303):")
    for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa"):
        a, b = res_b["v1"][k][0], res_b["v2"][k][0]
        print(f"    {k:>12}: {a:.4f}±{res_b['v1'][k][1]:.4f} → "
              f"{b:.4f}±{res_b['v2'][k][1]:.4f}  ({b - a:+.4f}, {(b - a) / a:+.2%})")

    # 4b. подтипы
    sd = pathlib.Path("runs/subtypes/data")
    sraw = json.loads((sd / "results_raw.json").read_text(encoding="utf-8"))
    schosen = {r["seed"]: r["best_epoch"] for r in sraw["runs"]}
    res_s = {"v1": [], "v2": []}
    ids_sub = None
    for tp in sorted(sd.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        ids_sub = t["dialog_ids"]
        i = t["epochs"].index(schosen[t["seed"]])
        yp = np.array(t["per_epoch"][i]["y_pred"])
        for ver in ("v1", "v2"):
            yt = np.array([[1 if s in L[ver][d]["subs"] else 0 for s in SUB] for d in ids_sub])
            res_s[ver].append(ml_metrics(yt, yp))
    res_s = {ver: {k: mstd([p[k] for p in v]) for k in v[0]} for ver, v in res_s.items()}
    imp["metrics"]["subtypes"] = res_s
    print("\n  подтипы (n=303):")
    for k in ("macro_f1", "micro_f1", "macro_kappa", "subset_accuracy", "hamming_loss"):
        a, b = res_s["v1"][k][0], res_s["v2"][k][0]
        print(f"    {k:>16}: {a:.4f}±{res_s['v1'][k][1]:.4f} → "
              f"{b:.4f}±{res_s['v2'][k][1]:.4f}  ({b - a:+.4f}, {(b - a) / a:+.2%})")

    # 4c. онсет — состав скамов меняется
    od = pathlib.Path("runs/onset/data")
    oraw = json.loads((od / "results_raw.json").read_text(encoding="utf-8"))
    ochosen = {r["seed"]: r["best_epoch"] for r in oraw["runs"]}
    oruns = []
    for tp in sorted(od.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        i = t["epochs"].index(ochosen[t["seed"]])
        oruns.append({"seed": t["seed"], "ids": [m["id"] for m in t["meta"]],
                      "pred": np.array(t["per_epoch"][i]["pred"]),
                      "true": np.array(t["per_epoch"][i]["true"])})
    ids_on = oruns[0]["ids"]
    res_o, masks = {}, {}
    for ver in ("v1", "v2"):
        m = np.array([L[ver][d]["is_scam"] == 1 for d in ids_on])
        masks[ver] = m
        per = [onset_metrics(r["pred"][m], r["true"][m]) for r in oruns]
        res_o[ver] = {k: mstd([p[k] for p in per]) for k in per[0]} | {"n": int(m.sum())}
    imp["metrics"]["onset"] = res_o
    print(f"\n  онсет (скамов v1 {res_o['v1']['n']} → v2 {res_o['v2']['n']}):")
    for k in ("oad", "exact", "within1", "bias"):
        a, b = res_o["v1"][k][0], res_o["v2"][k][0]
        print(f"    {k:>10}: {a:.4f}±{res_o['v1'][k][1]:.4f} → "
              f"{b:.4f}±{res_o['v2'][k][1]:.4f}  ({b - a:+.4f}, {(b - a) / abs(a):+.2%})")

    # смещение по бинам
    print("\n  смещение по бинам t*:")
    true_all = oruns[0]["true"]
    imp["onset_bins"] = {}
    for bname, sel in BINS:
        row = {}
        for ver in ("v1", "v2"):
            m = masks[ver] & sel(true_all)
            if not m.any():
                row[ver] = {"n": 0}
                continue
            per = [onset_metrics(r["pred"][m], r["true"][m]) for r in oruns]
            row[ver] = {k: mstd([p[k] for p in per]) for k in per[0]} | {"n": int(m.sum())}
        imp["onset_bins"][bname] = row
        if row["v1"].get("n"):
            print(f"    {bname:>5}: n {row['v1']['n']} → {row['v2']['n']}, "
                  f"смещение {row['v1']['bias'][0]:+.4f} → {row['v2']['bias'][0]:+.4f}")

    # ── 5. состав скамов и распределение t* ──
    d1 = collections.Counter(int(true_all[k]) for k in range(len(ids_on)) if masks["v1"][k])
    d2 = collections.Counter(int(true_all[k]) for k in range(len(ids_on)) if masks["v2"][k])
    removed = [ids_on[k] for k in range(len(ids_on)) if masks["v1"][k] and not masks["v2"][k]]
    imp["onset_composition"] = {
        "n_v1": int(masks["v1"].sum()), "n_v2": int(masks["v2"].sum()),
        "removed": removed,
        "t_star_dist_v1": {str(k): v for k, v in sorted(d1.items())},
        "t_star_dist_v2": {str(k): v for k, v in sorted(d2.items())}}
    print(f"\n5) состав онсет-теста: {masks['v1'].sum()} → {masks['v2'].sum()} "
          f"(убыло {len(removed)}: {removed})")
    print(f"   распределение t* v1: {dict(sorted(d1.items()))}")
    print(f"   распределение t* v2: {dict(sorted(d2.items()))}")

    # ── графики ──
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    specs = [(axes[0], "бинарная (xlm-r)", res_b,
              ["accuracy", "precision", "recall", "f1", "cohen_kappa"],
              ["acc", "prec", "recall", "F1", "κ"]),
             (axes[1], "подтипы", res_s,
              ["macro_f1", "micro_f1", "macro_kappa", "subset_accuracy"],
              ["macro-F1", "micro-F1", "macro-κ", "subset acc"]),
             (axes[2], f"онсет (n {res_o['v1']['n']} → {res_o['v2']['n']})", res_o,
              ["oad", "exact", "within1"], ["OAD", "exact", "±1"])]
    for ax, title, res, keys, labs in specs:
        x = np.arange(len(keys))
        for k, (ver, c) in enumerate((("v1", "#4C72B0"), ("v2", "#937860"))):
            ax.bar(x + (k - 0.5) * 0.38, [res[ver][m][0] for m in keys], 0.38,
                   yerr=[res[ver][m][1] for m in keys], capsize=4, color=c, label=ver)
        ax.set_xticks(x, labs, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("значение метрики")
    fig.suptitle("Влияние разбора спорных hard negatives: gold v1 против v2\n"
                 "усы — стандартное отклонение по 3 сидам, снимки те же")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "v1_vs_v2_metrics.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [b for b, _ in BINS]
    x = np.arange(len(names))
    for k, (ver, c) in enumerate((("v1", "#4C72B0"), ("v2", "#937860"))):
        ax.bar(x + (k - 0.5) * 0.38,
               [imp["onset_bins"][b][ver].get("n", 0) for b in names], 0.38,
               color=c, label=f"gold {ver}")
        for xi, b in enumerate(names):
            n = imp["onset_bins"][b][ver].get("n", 0)
            ax.text(xi + (k - 0.5) * 0.38, n + 1.5, str(n), ha="center", fontsize=10)
    ax.set_xticks(x, names)
    ax.set_xlabel("бин истинного t*")
    ax.set_ylabel("число скам-диалогов в тесте")
    ax.set_title("Состав онсет-теста по бинам t*: gold v1 против v2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "onset_bins_v1_vs_v2.png", dpi=DPI)
    plt.close(fig)

    (out_dir / "impact.json").write_text(json.dumps(imp, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\nграфики: {(out_dir / 'plots').resolve()}")
    print(f"результат: {(out_dir / 'impact.json').resolve()}")


if __name__ == "__main__":
    main()
