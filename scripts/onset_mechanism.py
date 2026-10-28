#!/usr/bin/env python3
"""Механика спешки онсет-модели: видит ли она правильное место вообще.

При t*>=5 модель называет онсет в среднем на 3.73 реплики раньше, причём все
предсказания ранние. Две гипотезы: (1) выучен приор t*=1 из train; (2) модель
реагирует на ранние поверхностные сигналы, не дожидаясь ложного утверждения.

Различить их можно по профилю вероятностей: если истинный t* стабильно даёт
локальный пик и входит в top-2, модель место видит и проблема в argmax; если
ранг истинного t* близок к случайному — медленные скамы она не видит вовсе.

Ничего не обучается: читаются снимки runs/onset/data/.

  python scripts/onset_mechanism.py
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
        ("3–4", lambda t: (t >= 3) & (t <= 4)), ("5+", lambda t: t >= 5)]


def mstd(v):
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/onset/data")
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--out", default="runs/onset")
    args = ap.parse_args()

    d = pathlib.Path(args.data)
    raw = json.loads((d / "results_raw.json").read_text(encoding="utf-8"))
    chosen = {r["seed"]: r["best_epoch"] for r in raw["runs"]}
    runs = []
    for tp in sorted(d.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        i = t["epochs"].index(chosen[t["seed"]])
        runs.append({"seed": t["seed"], "meta": t["meta"],
                     "probs": t["per_epoch"][i]["probs"],
                     "pred": np.array(t["per_epoch"][i]["pred"]),
                     "true": np.array(t["per_epoch"][i]["true"])})
    true = runs[0]["true"]
    out = {}

    def per_dialog(run, j):
        """Профиль вероятностей одного диалога + производные величины."""
        pr = np.array(run["probs"][j], dtype=float)
        t = int(run["true"][j])
        if len(pr) == 0 or t < 1 or t > len(pr):
            return None
        k = t - 1                                    # индекс истинного t*
        order = np.argsort(-pr)                      # по убыванию вероятности
        rank = int(np.where(order == k)[0][0]) + 1   # 1 = самая вероятная
        left = pr[k - 1] if k - 1 >= 0 else -np.inf
        right = pr[k + 1] if k + 1 < len(pr) else -np.inf
        return {"n_turns": int(len(pr)), "rank": rank,
                "local_max": bool(pr[k] >= left and pr[k] >= right),
                "p_true": float(pr[k]), "p_pred": float(pr.max()),
                "top2": bool(rank <= 2), "top3": bool(rank <= 3)}

    # ── 1. диалоги с t* >= 3 ──
    idx3 = [j for j in range(len(true)) if true[j] >= 3]
    print(f"1) ДИАЛОГИ С t* >= 3 (n={len(idx3)})\n")
    per_seed = []
    for run in runs:
        rows = [per_dialog(run, j) for j in idx3]
        rows = [r for r in rows if r]
        per_seed.append({
            "n": len(rows),
            "local_max_share": float(np.mean([r["local_max"] for r in rows])),
            "rank_mean": float(np.mean([r["rank"] for r in rows])),
            "rank_median": float(np.median([r["rank"] for r in rows])),
            "p_true_mean": float(np.mean([r["p_true"] for r in rows])),
            "p_pred_mean": float(np.mean([r["p_pred"] for r in rows])),
            "n_turns_mean": float(np.mean([r["n_turns"] for r in rows])),
        })
    agg = {k: mstd([p[k] for p in per_seed]) for k in per_seed[0] if k != "n"}
    agg["n"] = per_seed[0]["n"]
    out["t_star_ge3"] = agg
    print(f"  доля с локальным максимумом в истинном t*: "
          f"{agg['local_max_share'][0]:.4f} ± {agg['local_max_share'][1]:.4f}")
    print(f"  ранг истинного t*: среднее {agg['rank_mean'][0]:.3f} ± {agg['rank_mean'][1]:.3f}, "
          f"медиана {agg['rank_median'][0]:.2f}")
    print(f"  реплик в окне (среднее): {agg['n_turns_mean'][0]:.2f}  →  "
          f"случайный ранг был бы ≈ {(agg['n_turns_mean'][0] + 1) / 2:.2f}")
    print(f"  вероятность в истинном t*: {agg['p_true_mean'][0]:.4f}, "
          f"в предсказанном (максимум): {agg['p_pred_mean'][0]:.4f}, "
          f"отношение {agg['p_true_mean'][0] / agg['p_pred_mean'][0]:.3f}")
    out["random_rank_expected"] = float((agg["n_turns_mean"][0] + 1) / 2)

    # ── 2. top-k по бинам ──
    print("\n2) ПОПАДАНИЕ ИСТИННОГО t* В TOP-K\n")
    print(f"  {'бин':>6} {'n':>4} {'top-1':>16} {'top-2':>16} {'top-3':>16} "
          f"{'случайный top-2':>16}")
    out["topk_by_bin"] = {}
    for bname, sel in BINS:
        jj = [j for j in range(len(true)) if sel(true[j])]
        if not jj:
            continue
        vals = {"top1": [], "top2": [], "top3": [], "nt": []}
        for run in runs:
            rows = [per_dialog(run, j) for j in jj]
            rows = [r for r in rows if r]
            vals["top1"].append(float(np.mean([r["rank"] == 1 for r in rows])))
            vals["top2"].append(float(np.mean([r["top2"] for r in rows])))
            vals["top3"].append(float(np.mean([r["top3"] for r in rows])))
            vals["nt"].append(float(np.mean([r["n_turns"] for r in rows])))
        row = {k: mstd(v) for k, v in vals.items()}
        row["n"] = len(jj)
        row["random_top2"] = float(2 / row["nt"][0])
        row["random_top3"] = float(3 / row["nt"][0])
        # ранг и локальный максимум внутри бина: усреднение по всем t*>=3
        # маскирует различие между бинами 3–4 и 5+
        rk, lm, pt, pp = [], [], [], []
        for run in runs:
            rows = [per_dialog(run, j) for j in jj]
            rows = [r for r in rows if r]
            rk.append(float(np.mean([r["rank"] for r in rows])))
            lm.append(float(np.mean([r["local_max"] for r in rows])))
            pt.append(float(np.mean([r["p_true"] for r in rows])))
            pp.append(float(np.mean([r["p_pred"] for r in rows])))
        row["rank_mean"] = mstd(rk)
        row["local_max_share"] = mstd(lm)
        row["p_true_mean"] = mstd(pt)
        row["p_pred_mean"] = mstd(pp)
        row["random_rank"] = float((row["nt"][0] + 1) / 2)
        out["topk_by_bin"][bname] = row
        print(f"  {bname:>6} {row['n']:>4} "
              f"{row['top1'][0]:>8.4f}±{row['top1'][1]:.4f} "
              f"{row['top2'][0]:>8.4f}±{row['top2'][1]:.4f} "
              f"{row['top3'][0]:>8.4f}±{row['top3'][1]:.4f} "
              f"{row['random_top2']:>16.4f}")
        print(f"         ранг {row['rank_mean'][0]:.2f} (случайный {row['random_rank']:.2f}), "
              f"локальный максимум в t* {row['local_max_share'][0]:.4f}, "
              f"P(t*) {row['p_true_mean'][0]:.4f} против P(пик) {row['p_pred_mean'][0]:.4f}")

    # ── 3. распределение t* в train (silver) ──
    print("\n3) РАСПРЕДЕЛЕНИЕ t* В TRAIN (silver)\n")
    import collections
    cnt = collections.Counter()
    n_tr = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("split") == "train" and x.get("is_scam") == 1 and x.get("onset"):
                cnt[int(x["onset"])] += 1
                n_tr += 1
    share1_train = cnt[1] / n_tr
    gold_cnt = collections.Counter(int(t) for t in true)
    share1_gold = gold_cnt[1] / len(true)
    out["prior"] = {"train_n": n_tr, "train_share_t1": float(share1_train),
                    "train_distribution": {str(k): v for k, v in sorted(cnt.items())},
                    "gold_test_n": int(len(true)), "gold_test_share_t1": float(share1_gold),
                    "gold_test_distribution": {str(k): int(v) for k, v in sorted(gold_cnt.items())}}
    print(f"  train (silver): n={n_tr}, доля t*=1 = {share1_train:.4f}")
    print(f"  gold-test:      n={len(true)}, доля t*=1 = {share1_gold:.4f}")
    print(f"  распределение train: {dict(sorted(cnt.items()))}")
    print(f"  ВЫВОД: приор t*=1 в train {'СОВПАДАЕТ' if abs(share1_train - share1_gold) < 0.05 else 'НЕ СОВПАДАЕТ'} "
          f"с долей в gold ({share1_train:.1%} против {share1_gold:.1%})")

    # ── 4. профили для t* >= 5 ──
    idx5 = [j for j in range(len(true)) if true[j] >= 5]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plt.style.use("seaborn-v0_8-whitegrid")
    cmap = plt.get_cmap("tab20")
    for c, j in enumerate(idx5):
        prof = np.mean([np.array(r["probs"][j], dtype=float) for r in runs], axis=0)
        x = np.arange(1, len(prof) + 1)
        ax.plot(x, prof, marker="o", markersize=3.5, linewidth=1.4,
                color=cmap(c % 20), alpha=0.85)
        ax.axvline(true[j], color=cmap(c % 20), linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("номер реплики")
    ax.set_ylabel("P(здесь онсет)")
    ax.set_title(f"Профили вероятности онсета для диалогов с t* ≥ 5 (n={len(idx5)})\n"
                 "линия — диалог, пунктир того же цвета — его истинный t*; "
                 "усреднено по 3 сидам")
    ax.set_ylim(-0.02, 1.02)
    plots = pathlib.Path(args.out) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plots / "onset_prob_profiles.png", dpi=DPI)
    plt.close(fig)

    (pathlib.Path(args.out) / "mechanism.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nграфик:    {(plots / 'onset_prob_profiles.png').resolve()}")
    print(f"результат: {(pathlib.Path(args.out) / 'mechanism.json').resolve()}")


if __name__ == "__main__":
    main()
