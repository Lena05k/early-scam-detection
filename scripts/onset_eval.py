#!/usr/bin/env python3
"""Разбор онсет-головы: метрики на gold-test в двух вариантах.

Вариант (а) — все скамы теста (191). Вариант (б) — без 51 спорного диалога
(корпусный hard_negative=1 при gold is_scam=1, в тесте таких 40). Вопрос об их
скамовости НЕ решается: разница между вариантами сама по себе результат.

Точки отсчёта: silver-онсеты DeepSeek против gold и тривиальный предсказатель
«всегда t*=1» — без них OAD не читается, потому что 69% золотых онсетов равны 1.

Ничего не обучается: читаются снимки предсказаний по эпохам.

  python scripts/onset_eval.py
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
STYLE = "seaborn-v0_8-whitegrid"
SEED_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
LANG_COLORS = {"en": "#4C72B0", "ru": "#DD8452", "zh": "#55A868"}


def onset_metrics(pred, true):
    """OAD и производные. pred/true — массивы индексов реплик."""
    if len(pred) == 0:
        return {"n": 0}
    d = np.abs(np.asarray(pred) - np.asarray(true))
    return {"n": int(len(d)), "oad": float(d.mean()),
            "exact": float((d == 0).mean()), "within1": float((d <= 1).mean()),
            "within2": float((d <= 2).mean()), "median_abs": float(np.median(d)),
            "bias": float((np.asarray(pred) - np.asarray(true)).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/onset/data")
    ap.add_argument("--out", default="runs/onset")
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    raw = json.loads((data / "results_raw.json").read_text(encoding="utf-8"))
    chosen = {r["seed"]: r["best_epoch"] for r in raw["runs"]}
    out_dir = pathlib.Path(args.out)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE)

    runs = []
    for tp in sorted(data.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        seed = t["seed"]
        dv = json.loads((data / f"gold_dev_per_epoch_seed{seed}.json").read_text(encoding="utf-8"))
        meta = t["meta"]
        i = t["epochs"].index(chosen[seed])
        runs.append({
            "seed": seed, "epochs": t["epochs"], "chosen_epoch": chosen[seed],
            "meta": meta,
            "pred": np.array(t["per_epoch"][i]["pred"]),
            "true": np.array(t["per_epoch"][i]["true"]),
            "probs": t["per_epoch"][i]["probs"],
            "oad_by_epoch_test": [onset_metrics(np.array(pe["pred"]), np.array(pe["true"]))["oad"]
                                  for pe in t["per_epoch"]],
            "oad_by_epoch_dev": [onset_metrics(np.array(pe["pred"]), np.array(pe["true"]))["oad"]
                                 for pe in dv["per_epoch"]],
        })

    meta = runs[0]["meta"]
    true = runs[0]["true"]
    disputed = np.array([m["disputed"] for m in meta], dtype=bool)
    langs = np.array([m["lang"] for m in meta])
    silver = np.array([m["t_star_silver"] if m["t_star_silver"] else -1 for m in meta])
    n_all, n_clean = len(true), int((~disputed).sum())

    VARIANTS = [("а) все скамы", np.ones(n_all, dtype=bool)),
                ("б) без спорных", ~disputed)]

    print(f"gold-test: скамов {n_all}, спорных {int(disputed.sum())}, "
          f"без спорных {n_clean}")
    print(f"t* за пределами 512 токенов: {raw['counts']['test_t_star_out_of_window']} диалог(ов)")
    print(f"лучшие эпохи {raw['best_epochs']}, отработано {raw['epochs_ran']} из "
          f"{raw['config']['epochs']}\n")

    def agg(mask, key="pred"):
        """Среднее ± std по сидам для каждой метрики."""
        per = [onset_metrics(r[key][mask], r["true"][mask]) for r in runs]
        keys = [k for k in per[0] if k != "n"]
        return {k: (float(np.mean([p[k] for p in per])),
                    float(np.std([p[k] for p in per], ddof=1))) for k in keys} | {"n": per[0]["n"]}

    results = {"counts": raw["counts"], "config": raw["config"],
               "best_epochs": raw["best_epochs"], "epochs_ran": raw["epochs_ran"],
               "variants": {}, "reference": {}, "per_language": {}}

    MET = ["oad", "exact", "within1", "within2", "median_abs", "bias"]
    print(f"{'вариант':>18} {'n':>4} " + " ".join(f"{m:>15}" for m in MET))
    for name, mask in VARIANTS:
        a = agg(mask)
        results["variants"][name] = a
        print(f"{name:>18} {a['n']:>4} " +
              " ".join(f"{a[m][0]:>8.4f}±{a[m][1]:.4f}" for m in MET))

    # ── точки отсчёта ──
    print("\nточки отсчёта:")
    for name, mask in VARIANTS:
        ok = mask & (silver > 0)
        s = onset_metrics(silver[ok], true[ok])
        results["reference"][f"silver {name}"] = s
        print(f"  silver DeepSeek, {name:>16}: n={s['n']:>3} OAD {s['oad']:.4f} "
              f"exact {s['exact']:.4f} ±1 {s['within1']:.4f} смещение {s['bias']:+.4f}")
    for name, mask in VARIANTS:
        c = onset_metrics(np.ones(int(mask.sum()), dtype=int), true[mask])
        results["reference"][f"всегда t*=1 {name}"] = c
        print(f"  всегда t*=1,    {name:>16}: n={c['n']:>3} OAD {c['oad']:.4f} "
              f"exact {c['exact']:.4f} ±1 {c['within1']:.4f} смещение {c['bias']:+.4f}")

    # ── разница между вариантами ──
    a, b = results["variants"]["а) все скамы"], results["variants"]["б) без спорных"]
    print("\nразница (б) − (а):")
    big = []
    for m in MET:
        d = b[m][0] - a[m][0]
        rel = abs(d) / abs(a[m][0]) if a[m][0] else float("inf")
        results.setdefault("variant_delta", {})[m] = {"abs": float(d), "rel": float(rel)}
        flag = "  ⚠ >10%" if rel > 0.10 else ""
        big.append(rel > 0.10)
        print(f"  {m:>12}: {a[m][0]:.4f} → {b[m][0]:.4f}  ({d:+.4f}, {rel:+.1%}){flag}")
    results["variant_delta_over_10pct"] = bool(any(big))

    # ── по языкам ──
    print("\nпо языкам (вариант а / вариант б):")
    for lg in sorted(set(langs.tolist())):
        row = {}
        for name, mask in VARIANTS:
            mm = mask & (langs == lg)
            row[name] = agg(mm)
        results["per_language"][lg] = row
        print(f"  {lg}: OAD {row['а) все скамы']['oad'][0]:.4f} / "
              f"{row['б) без спорных']['oad'][0]:.4f}   "
              f"exact {row['а) все скамы']['exact'][0]:.4f} / "
              f"{row['б) без спорных']['exact'][0]:.4f}   "
              f"(n {row['а) все скамы']['n']} / {row['б) без спорных']['n']})")

    # ─────────── графики ───────────
    pred_mean = runs[0]["pred"]

    # 1. onset_scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(20260809)
    for lg in sorted(set(langs.tolist())):
        for dis, mk in ((False, "o"), (True, "o")):
            m = (langs == lg) & (disputed == dis)
            if not m.any():
                continue
            jx = true[m] + rng.uniform(-0.18, 0.18, m.sum())
            jy = pred_mean[m] + rng.uniform(-0.18, 0.18, m.sum())
            ax.scatter(jx, jy, s=60, marker=mk,
                       facecolors="none" if dis else LANG_COLORS.get(lg, "#888"),
                       edgecolors=LANG_COLORS.get(lg, "#888"), linewidths=1.6 if dis else 0,
                       label=f"{lg}{' (спорные)' if dis else ''}", alpha=0.85)
    lim = max(true.max(), pred_mean.max()) + 1
    ax.plot([0, lim], [0, lim], color="#666", linestyle="--", linewidth=1.6, label="идеал")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("истинный t* (gold)")
    ax.set_ylabel("предсказанный t̂*")
    ax.set_title(f"Предсказанный онсет против истинного, gold-test (n={n_all}, сид "
                 f"{runs[0]['seed']})\nполые точки — спорные (hard_negative=1), "
                 "координаты дрожат для читаемости")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "onset_scatter.png", dpi=DPI)
    plt.close(fig)

    # 2. error_distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    err = np.concatenate([r["pred"] - r["true"] for r in runs])
    bins = np.arange(err.min() - 0.5, err.max() + 1.5, 1)
    ax.hist(err, bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="#666", linestyle="--", linewidth=1.8, label="ноль")
    ax.axvline(err.mean(), color="#C44E52", linestyle="-", linewidth=2.2,
               label=f"среднее смещение {err.mean():+.3f}")
    ax.set_xlabel("t̂* − t*  (>0 — модель опаздывает)")
    ax.set_ylabel("число диалогов (все сиды)")
    ax.set_title("Распределение ошибки онсета на gold-test (вариант а, все скамы)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "error_distribution.png", dpi=DPI)
    plt.close(fig)

    # 3. oad_by_epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, c in zip(runs, SEED_COLORS):
        ax.plot(r["epochs"], r["oad_by_epoch_dev"], marker="s", linestyle="--", color=c,
                linewidth=2, label=f"gold-dev, сид {r['seed']}")
        ax.plot(r["epochs"], r["oad_by_epoch_test"], marker="o", linestyle="-", color=c,
                linewidth=2, label=f"gold-test, сид {r['seed']}")
        i = r["epochs"].index(r["chosen_epoch"])
        ax.scatter([r["chosen_epoch"]], [r["oad_by_epoch_dev"][i]], s=230,
                   facecolors="none", edgecolors=c, linewidths=2.6, zorder=5)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("OAD (меньше — лучше)")
    ax.set_title("OAD по эпохам: gold-dev (пунктир) и gold-test (сплошная)\n"
                 "кружок — эпоха, выбранная по минимуму OAD на dev")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "oad_by_epoch.png", dpi=DPI)
    plt.close(fig)

    # 4. onset_distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    mx = int(max(true.max(), max(r["pred"].max() for r in runs)))
    bins = np.arange(0.5, mx + 1.5, 1)
    ax.hist(true, bins=bins, alpha=0.6, color="#55A868", label=f"gold t* (n={n_all})")
    ax.hist(np.concatenate([r["pred"] for r in runs]), bins=bins, alpha=0.6,
            color="#C44E52", weights=np.full(n_all * len(runs), 1 / len(runs)),
            label="предсказанный t̂* (среднее по сидам)")
    ax.set_xlabel("номер реплики")
    ax.set_ylabel("число диалогов")
    ax.set_title("Распределение онсета: истинное и предсказанное, gold-test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "onset_distribution.png", dpi=DPI)
    plt.close(fig)

    # 5. accuracy_by_tolerance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    tols = np.arange(0, 6)
    for ax_, (name, mask) in zip(axes, VARIANTS):
        for r, c in zip(runs, SEED_COLORS):
            d = np.abs(r["pred"][mask] - r["true"][mask])
            ax_.plot(tols, [(d <= t).mean() for t in tols], marker="o", color=c,
                     linewidth=2, label=f"сид {r['seed']}")
        ax_.set_title(f"{name} (n={int(mask.sum())})")
        ax_.set_xlabel("допуск, реплик")
        ax_.set_xticks(tols)
        ax_.set_ylim(0, 1.02)
        ax_.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("доля диалогов в пределах допуска")
    fig.suptitle("Точность онсета как функция допуска")
    fig.tight_layout()
    fig.savefig(plots / "accuracy_by_tolerance.png", dpi=DPI)
    plt.close(fig)

    # 6. per_language_onset
    lgs = sorted(set(langs.tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(lgs))
    for k, (name, _) in enumerate(VARIANTS):
        axes[0].bar(x + (k - 0.5) * 0.38,
                    [results["per_language"][lg][name]["oad"][0] for lg in lgs], 0.38,
                    yerr=[results["per_language"][lg][name]["oad"][1] for lg in lgs],
                    capsize=4, color=["#4C72B0", "#937860"][k], label=name)
        axes[1].bar(x + (k - 0.5) * 0.38,
                    [results["per_language"][lg][name]["exact"][0] for lg in lgs], 0.38,
                    yerr=[results["per_language"][lg][name]["exact"][1] for lg in lgs],
                    capsize=4, color=["#4C72B0", "#937860"][k], label=name)
    for ax_, ttl, yl in ((axes[0], "OAD по языкам (меньше — лучше)", "OAD"),
                         (axes[1], "Точное попадание по языкам", "доля exact")):
        ax_.set_xticks(x, lgs)
        ax_.set_xlabel("язык")
        ax_.set_ylabel(yl)
        ax_.set_title(ttl)
        ax_.legend(fontsize=9)
    fig.suptitle("Онсет по языкам, gold-test, среднее ± std по 3 сидам")
    fig.tight_layout()
    fig.savefig(plots / "per_language_onset.png", dpi=DPI)
    plt.close(fig)

    # 7. turn_prob_heatmap — 20 случайных скам-диалогов
    rng = np.random.default_rng(20260809)
    idx = sorted(rng.choice(n_all, size=min(20, n_all), replace=False).tolist())
    probs = runs[0]["probs"]
    mx_t = max(len(probs[i]) for i in idx)
    mat = np.full((len(idx), mx_t), np.nan)
    for k, i in enumerate(idx):
        mat[k, :len(probs[i])] = probs[i]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    for k, i in enumerate(idx):
        ax.plot([true[i] - 1], [k], marker="|", markersize=18, color="#1f3b73",
                markeredgewidth=3)
    ax.set_xticks(range(mx_t), [str(t + 1) for t in range(mx_t)])
    ax.set_yticks(range(len(idx)),
                  [f"{meta[i]['id'][-18:]} ({meta[i]['lang']})" for i in idx], fontsize=8)
    ax.set_xlabel("номер реплики")
    ax.set_title(f"Вероятность онсета по репликам, {len(idx)} случайных скам-диалогов "
                 f"теста (сид {runs[0]['seed']})\nсиняя черта — истинный t*")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(здесь онсет)")
    fig.tight_layout()
    fig.savefig(plots / "turn_prob_heatmap.png", dpi=DPI)
    plt.close(fig)

    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\nграфики:")
    for f in sorted(plots.glob("*.png")):
        print("  ", f.resolve())
    print("  ", (out_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
