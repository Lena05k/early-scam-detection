#!/usr/bin/env python3
"""Цена неверного выбора эпохи: метрики gold-test на КАЖДОЙ эпохе.

Бутстрап показал, что отбор чекпоинта по gold-dev неустойчив (попаданий 62–66%
у xlm-r). Здесь измеряется, во что этот промах обходится: для каждой эпохи каждого
сида считаются метрики на gold-test 303 и сравниваются с выбранной эпохой.

Цена неверного выбора = F1(выбранная эпоха) − F1(лучшая по тесту эпоха) ≤ 0.

Ничего не обучается: читаются gold_test_per_epoch_seed*.json и
gold_dev_per_epoch_seed*.json, снятые во время прогонов.

  python scripts/epoch_cost.py
"""
import argparse
import json
import pathlib

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap_selection import bootstrap_seed  # noqa: E402

DPI = 150
STYLE = "seaborn-v0_8-whitegrid"
SEED_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MODEL_COLORS = {"xlm-roberta-base": "#4C72B0", "mDeBERTa-v3-base": "#C44E52"}


def metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}


def is_degenerate(m, y_pred):
    """Вырожденная эпоха: модель почти не различает классы.

    (а) recall > 0.98 при precision < 0.75 — гонит почти всё в scam;
    (б) доля предсказаний одного класса > 0.9 — почти константный ответ.
    Пример: mDeBERTa сид 20260721 эпоха 1, матрица [[9,103],[0,191]].
    """
    if m["recall"] > 0.98 and m["precision"] < 0.75:
        return "recall>0.98 при precision<0.75"
    share = max(float((y_pred == c).mean()) for c in (0, 1))
    if share > 0.9:
        return f"доля одного класса {share:.3f} > 0.9"
    return None


def load_model(d, name):
    """Каталог прогона → структура с пер-эпохными метриками теста и dev."""
    d = pathlib.Path(d)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    chosen = {r["seed"]: r["best_epoch"] for r in res["runs"]}
    out = {"name": name, "versions": res["config"].get("versions"),
           "config": res["config"], "seeds": [], "n_test": res["runs"][0]["n_test"]}
    for tp in sorted(d.glob("gold_test_per_epoch_seed*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        seed = t["seed"]
        dp = d / f"gold_dev_per_epoch_seed{seed}.json"
        dv = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else None
        y_true = np.array(t["per_epoch"][0]["y_true"])
        langs = np.array(t["langs"])
        eps = []
        for i, ep in enumerate(t["epochs"]):
            yp = np.array(t["per_epoch"][i]["y_pred"])
            m = metrics(y_true, yp)
            m["epoch"] = ep
            m["prob_scam"] = t["per_epoch"][i]["prob_scam"]
            m["confusion"] = [[int(((yp == c) & (y_true == g)).sum()) for c in (0, 1)]
                              for g in (0, 1)]
            m["per_language"] = {lg: metrics(y_true[langs == lg], yp[langs == lg])
                                 for lg in sorted(set(langs.tolist()))}
            m["degenerate"] = is_degenerate(m, yp)
            if dv:
                dy = np.array(dv["per_epoch"][0]["y_true"])
                dpred = np.array(dv["per_epoch"][i]["y_pred"])
                m["gold_dev_f1"] = metrics(dy, dpred)["f1"]
            eps.append(m)
        f1s = [e["f1"] for e in eps]
        best_by_test = t["epochs"][int(np.argmax(f1s))]
        sel = chosen[seed]
        out["seeds"].append({
            "seed": seed, "epochs": t["epochs"], "per_epoch": eps,
            "chosen_epoch": sel, "best_by_test_epoch": best_by_test,
            "f1_chosen": eps[t["epochs"].index(sel)]["f1"],
            "f1_best_by_test": max(f1s),
            "cost": eps[t["epochs"].index(sel)]["f1"] - max(f1s),
            "f1_spread": float(max(f1s) - min(f1s)),
            "f1_by_epoch": {int(e): float(f) for e, f in zip(t["epochs"], f1s)},
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlmr-dir", default="runs/epoch_cost/data/xlmr")
    ap.add_argument("--mdeberta-dir", default="runs/epoch_cost/data/mdeberta")
    ap.add_argument("--out", default="runs/epoch_cost")
    args = ap.parse_args()

    models = [load_model(args.xlmr_dir, "xlm-roberta-base"),
              load_model(args.mdeberta_dir, "mDeBERTa-v3-base")]
    out_dir = pathlib.Path(args.out)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE)
    n_test = models[0]["n_test"]

    # ── главная таблица ──
    print(f"gold-test, n={n_test}: метрики на КАЖДОЙ эпохе\n")
    hdr = (f"{'модель':>18} {'сид':>10} {'эпоха':>6} {'F1':>8} {'prec':>8} "
           f"{'recall':>8} {'kappa':>8} {'dev F1':>8}  ")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        for s in m["seeds"]:
            for e in s["per_epoch"]:
                mark = ""
                if e["epoch"] == s["chosen_epoch"]:
                    mark += " ВЫБРАНА"
                if e["epoch"] == s["best_by_test_epoch"]:
                    mark += " ЛУЧШАЯ-ПО-ТЕСТУ"
                dv = f"{e.get('gold_dev_f1', float('nan')):8.4f}"
                print(f"{m['name']:>18} {s['seed']:>10} {e['epoch']:>6} {e['f1']:>8.4f} "
                      f"{e['precision']:>8.4f} {e['recall']:>8.4f} {e['cohen_kappa']:>8.4f} "
                      f"{dv}{mark}")
            print("-" * len(hdr))

    # ── цена ──
    print(f"\n{'модель':>18} {'сид':>10} {'выбрана':>8} {'лучшая':>8} "
          f"{'F1 выбр.':>9} {'F1 лучш.':>9} {'ЦЕНА':>8} {'размах F1':>10}")
    all_costs = {}
    for m in models:
        costs = []
        for s in m["seeds"]:
            costs.append(s["cost"])
            print(f"{m['name']:>18} {s['seed']:>10} {s['chosen_epoch']:>8} "
                  f"{s['best_by_test_epoch']:>8} {s['f1_chosen']:>9.4f} "
                  f"{s['f1_best_by_test']:>9.4f} {s['cost']:>8.4f} {s['f1_spread']:>10.4f}")
        all_costs[m["name"]] = costs

    print("\nсводка по цене неверного выбора (F1 выбранной − F1 лучшей по тесту):")
    for name, c in all_costs.items():
        print(f"  {name:>18}: средняя {np.mean(c):+.4f}, максимальная {min(c):+.4f}, "
              f"нулевая у {sum(1 for x in c if x == 0)}/{len(c)} сидов")
    pooled = [c for v in all_costs.values() for c in v]
    print(f"  {'обе модели':>18}: средняя {np.mean(pooled):+.4f}, максимальная {min(pooled):+.4f}")

    # ── корреляция dev ↔ test ──
    corr = {}
    for m in models:
        xs = [e["gold_dev_f1"] for s in m["seeds"] for e in s["per_epoch"] if "gold_dev_f1" in e]
        ys = [e["f1"] for s in m["seeds"] for e in s["per_epoch"] if "gold_dev_f1" in e]
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
        corr[m["name"]] = {"pearson_r": r, "n_points": len(xs), "dev": xs, "test": ys}
    xs_all = [x for m in models for x in corr[m["name"]]["dev"]]
    ys_all = [y for m in models for y in corr[m["name"]]["test"]]
    r_all = float(np.corrcoef(xs_all, ys_all)[0, 1])
    corr["pooled"] = {"pearson_r": r_all, "n_points": len(xs_all)}
    # ФАКТИЧЕСКАЯ цена (определение из задания) равна нулю, если отбор угадал.
    # Поэтому дополнительно считаем ОЖИДАЕМУЮ цену: распределение выбора эпохи из
    # бутстрапа по gold-dev ЭТОГО ЖЕ прогона, свёрнутое с ценой каждой эпохи.
    for m in models:
        d = pathlib.Path(args.xlmr_dir if "xlm" in m["name"] else args.mdeberta_dir)
        for s in m["seeds"]:
            snap_p = d / f"gold_dev_per_epoch_seed{s['seed']}.json"
            if not snap_p.exists():
                continue
            bs = bootstrap_seed(json.loads(snap_p.read_text(encoding="utf-8")), 1000,
                                np.random.default_rng(20260808))
            dist = bs["selection_distribution"]
            best = s["f1_best_by_test"]
            s["selection_distribution"] = dist
            s["expected_cost"] = float(sum(
                p * (s["f1_by_epoch"][e] - best) for e, p in dist.items()))
            s["hit_rate"] = bs["hit_rate"]

    print("\nожидаемая цена с учётом неустойчивости отбора")
    print("(распределение выбора по бутстрапу gold-dev × цена каждой эпохи):")
    for m in models:
        ec = [s["expected_cost"] for s in m["seeds"] if "expected_cost" in s]
        if ec:
            print(f"  {m['name']:>18}: средняя ожидаемая цена {np.mean(ec):+.4f}, "
                  f"худшая {min(ec):+.4f}")
            for s in m["seeds"]:
                if "expected_cost" in s:
                    print(f"      сид {s['seed']}: попаданий {s['hit_rate']:.1%}, "
                          f"фактическая {s['cost']:+.4f}, ожидаемая {s['expected_cost']:+.4f}")

    # 1a: те же корреляции БЕЗ вырожденных эпох
    corr_clean = {}
    for m in models:
        pts = [(e["gold_dev_f1"], e["f1"]) for s in m["seeds"] for e in s["per_epoch"]
               if "gold_dev_f1" in e and not e["degenerate"]]
        n_drop = sum(1 for s in m["seeds"] for e in s["per_epoch"] if e["degenerate"])
        r = float(np.corrcoef([x for x, _ in pts], [y for _, y in pts])[0, 1]) if len(pts) > 2 else float("nan")
        corr_clean[m["name"]] = {"pearson_r": r, "n_points": len(pts), "n_dropped": n_drop}
    pts_all = [(e["gold_dev_f1"], e["f1"]) for m in models for s in m["seeds"]
               for e in s["per_epoch"] if "gold_dev_f1" in e and not e["degenerate"]]
    corr_clean["pooled"] = {
        "pearson_r": float(np.corrcoef([x for x, _ in pts_all], [y for _, y in pts_all])[0, 1]),
        "n_points": len(pts_all),
        "n_dropped": sum(v["n_dropped"] for k, v in corr_clean.items() if k != "pooled")}

    print("\nкорреляция Пирсона gold_dev F1 ↔ gold_test F1 (точка = эпоха одного сида):")
    print(f"{'':>18}  {'со всеми':>22}   {'без вырожденных':>26}")
    for k in corr:
        c1, c2 = corr[k], corr_clean[k]
        print(f"  {k:>18}: r={c1['pearson_r']:+.4f} (n={c1['n_points']:>2})"
              f"      r={c2['pearson_r']:+.4f} (n={c2['n_points']:>2}, "
              f"выброшено {c2['n_dropped']})")
    print("\n  вырожденные эпохи:")
    for m in models:
        for s in m["seeds"]:
            for e in s["per_epoch"]:
                if e["degenerate"]:
                    cm = e["confusion"]
                    print(f"    {m['name']}, сид {s['seed']}, эпоха {e['epoch']}: "
                          f"{e['degenerate']}, матрица {cm}")

    # 1b: знаковое согласие на СОСЕДНИХ эпохах — разрешающая способность dev
    sign_stats = {}
    for m in models:
        pairs = []
        for s in m["seeds"]:
            pe = [e for e in s["per_epoch"] if "gold_dev_f1" in e]
            for a, b in zip(pe, pe[1:]):
                d_dev = b["gold_dev_f1"] - a["gold_dev_f1"]
                d_test = b["f1"] - a["f1"]
                pairs.append({"seed": s["seed"], "from": a["epoch"], "to": b["epoch"],
                              "d_dev": float(d_dev), "d_test": float(d_test),
                              "agree": bool(np.sign(d_dev) == np.sign(d_test)),
                              "dev_tie": bool(d_dev == 0)})
        n = len(pairs)
        nz = [p for p in pairs if not p["dev_tie"]]
        sign_stats[m["name"]] = {
            "n_pairs": n, "n_agree": sum(p["agree"] for p in pairs),
            "agreement": float(np.mean([p["agree"] for p in pairs])) if n else float("nan"),
            "n_pairs_no_tie": len(nz),
            "agreement_no_tie": float(np.mean([p["agree"] for p in nz])) if nz else float("nan"),
            "pairs": pairs}
    all_pairs = [p for v in sign_stats.values() for p in v["pairs"]]
    all_nz = [p for p in all_pairs if not p["dev_tie"]]
    sign_stats["pooled"] = {"n_pairs": len(all_pairs),
                            "n_agree": sum(p["agree"] for p in all_pairs),
                            "agreement": float(np.mean([p["agree"] for p in all_pairs])),
                            "n_pairs_no_tie": len(all_nz),
                            "agreement_no_tie": float(np.mean([p["agree"] for p in all_nz]))}
    print("\nразрешающая способность dev на соседних эпохах")
    print("(совпал ли ЗНАК разницы dev со знаком разницы на тесте):")
    for k, v in sign_stats.items():
        print(f"  {k:>18}: {v['n_agree']}/{v['n_pairs']} = {v['agreement']:.1%}"
              f"   без ничьих на dev: {v['agreement_no_tie']:.1%} (n={v['n_pairs_no_tie']})")

    # ───────────────────────── графики ─────────────────────────
    def panels(title):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
        fig.suptitle(title)
        return fig, axes

    # 1. test_by_epoch
    fig, axes = panels(f"F1 на золотом тесте по эпохам (n={n_test})\n"
                       "крупный маркер — эпоха, выбранная по gold-dev")
    for ax, m in zip(axes, models):
        for s, c in zip(m["seeds"], SEED_COLORS):
            ax.plot(s["epochs"], [e["f1"] for e in s["per_epoch"]], marker="o",
                    color=c, linewidth=2, label=f"сид {s['seed']}")
            i = s["epochs"].index(s["chosen_epoch"])
            ax.scatter([s["chosen_epoch"]], [s["per_epoch"][i]["f1"]], s=260,
                       facecolors="none", edgecolors=c, linewidths=3, zorder=5)
        ax.set_title(m["name"])
        ax.set_xlabel("эпоха")
        ax.set_xticks(sorted({e for s in m["seeds"] for e in s["epochs"]}))
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("F1 на gold-test")
    fig.tight_layout()
    fig.savefig(plots / "test_by_epoch.png", dpi=DPI)
    plt.close(fig)

    # 2. dev_vs_test
    fig, axes = panels("gold-dev (пунктир) против gold-test (сплошная) по эпохам\n"
                       "следует ли dev за тестом")
    for ax, m in zip(axes, models):
        for s, c in zip(m["seeds"], SEED_COLORS):
            ax.plot(s["epochs"], [e["f1"] for e in s["per_epoch"]], marker="o",
                    color=c, linewidth=2, linestyle="-", label=f"test, сид {s['seed']}")
            if "gold_dev_f1" in s["per_epoch"][0]:
                ax.plot(s["epochs"], [e["gold_dev_f1"] for e in s["per_epoch"]], marker="s",
                        color=c, linewidth=2, linestyle="--", label=f"dev, сид {s['seed']}")
        ax.set_title(m["name"])
        ax.set_xlabel("эпоха")
        ax.set_xticks(sorted({e for s in m["seeds"] for e in s["epochs"]}))
        ax.legend(fontsize=8, ncol=2, loc="lower right")
    axes[0].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(plots / "dev_vs_test.png", dpi=DPI)
    plt.close(fig)

    # 3. dev_test_scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        c = MODEL_COLORS[m["name"]]
        for s in m["seeds"]:
            for e in s["per_epoch"]:
                if "gold_dev_f1" not in e:
                    continue
                sel = e["epoch"] == s["chosen_epoch"]
                deg = bool(e["degenerate"])
                ax.scatter(e["gold_dev_f1"], e["f1"], s=170 if sel else 70,
                           marker="*" if sel else "o",
                           color="none" if deg else c,
                           edgecolors=c if deg else ("black" if sel else "none"),
                           linewidths=1.8 if deg else (1.2 if sel else 0),
                           zorder=4 if sel else 3)
        ax.scatter([], [], color=c, label=f"{m['name']} (r = {corr[m['name']]['pearson_r']:+.3f})")
    xc = [x for x, _ in pts_all]
    yc = [y for _, y in pts_all]
    b, a = np.polyfit(xc, yc, 1)
    xr = np.linspace(min(xs_all), max(xs_all), 50)
    ax.plot(xr, a + b * xr, color="#666", linestyle="--", linewidth=1.8,
            label=f"регрессия по невырожденным, r = {corr_clean['pooled']['pearson_r']:+.3f}"
                  f"  (со всеми {r_all:+.3f})")
    ax.scatter([], [], facecolors="none", edgecolors="#666", s=90,
               label=f"полые — вырожденные эпохи ({corr_clean['pooled']['n_dropped']} шт., "
                     f"в регрессию не входят)")
    ax.scatter([], [], marker="*", s=170, color="none", edgecolors="black",
               label="звезда — эпоха, выбранная по dev")
    ax.set_xlabel("F1 на gold-dev (100 диалогов)")
    ax.set_ylabel(f"F1 на gold-test ({n_test} диалогов)")
    ax.set_title("Предсказывает ли gold-dev результат на тесте\n"
                 "точка = одна эпоха одного сида")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "dev_test_scatter.png", dpi=DPI)
    plt.close(fig)

    # 4. cost_distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    per_epoch_costs = {}
    for m in models:
        cc = []
        for s in m["seeds"]:
            best = s["f1_best_by_test"]
            cc += [e["f1"] - best for e in s["per_epoch"]]
        per_epoch_costs[m["name"]] = cc
        ax.hist(cc, bins=np.linspace(min(min(v) for v in per_epoch_costs.values()) - 0.01, 0.005, 22),
                alpha=0.6, color=MODEL_COLORS[m["name"]],
                label=f"{m['name']} (средняя {np.mean(cc):+.4f})")
    ax.set_xlabel("цена: F1 эпохи − F1 лучшей по тесту эпохи")
    ax.set_ylabel("число эпох")
    ax.set_title("Распределение цены выбора эпохи по всем эпохам и сидам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "cost_distribution.png", dpi=DPI)
    plt.close(fig)

    # 5. metric_tradeoff_by_epoch
    fig, axes = panels("Precision и recall на золотом тесте по эпохам\n"
                       "что именно меняется с обучением")
    for ax, m in zip(axes, models):
        for s, c in zip(m["seeds"], SEED_COLORS):
            ax.plot(s["epochs"], [e["precision"] for e in s["per_epoch"]], marker="o",
                    color=c, linewidth=2, linestyle="-", label=f"precision, сид {s['seed']}")
            ax.plot(s["epochs"], [e["recall"] for e in s["per_epoch"]], marker="v",
                    color=c, linewidth=2, linestyle=":", label=f"recall, сид {s['seed']}")
        ax.set_title(m["name"])
        ax.set_xlabel("эпоха")
        ax.set_xticks(sorted({e for s in m["seeds"] for e in s["epochs"]}))
        ax.legend(fontsize=8, ncol=2, loc="lower left")
    axes[0].set_ylabel("значение метрики")
    fig.tight_layout()
    fig.savefig(plots / "metric_tradeoff_by_epoch.png", dpi=DPI)
    plt.close(fig)

    # 6. confusion_by_epoch — отдельный файл на модель
    for m in models:
        eps = sorted({e for s in m["seeds"] for e in s["epochs"]})
        fig, axes = plt.subplots(len(eps), len(m["seeds"]),
                                 figsize=(3.1 * len(m["seeds"]), 2.7 * len(eps)),
                                 squeeze=False)
        for i, ep in enumerate(eps):
            for j, s in enumerate(m["seeds"]):
                ax = axes[i][j]
                rec = next((e for e in s["per_epoch"] if e["epoch"] == ep), None)
                if rec is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "не дошла\n(ранняя остановка)", ha="center",
                            va="center", fontsize=9, color="#999")
                    continue
                cm = np.array(rec["confusion"])
                ax.imshow(cm, cmap="Blues")
                for a in range(2):
                    for b_ in range(2):
                        ax.text(b_, a, str(cm[a, b_]), ha="center", va="center",
                                fontsize=11,
                                color="white" if cm[a, b_] > cm.max() / 2 else "#222")
                ax.set_xticks([0, 1], ["b", "s"], fontsize=8)
                ax.set_yticks([0, 1], ["b", "s"], fontsize=8)
                ax.grid(False)
                if i == 0:
                    ax.set_title(f"сид {s['seed']}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"эпоха {ep}\ngold", fontsize=9)
                if i == len(eps) - 1:
                    ax.set_xlabel("предсказание", fontsize=9)
        fig.suptitle(f"{m['name']}: матрицы ошибок на gold-test (n={n_test})\n"
                     "строки — эпохи, столбцы — сиды; b = benign, s = scam")
        fig.tight_layout()
        tag = "xlmr" if "xlm" in m["name"] else "mdeberta"
        fig.savefig(plots / f"confusion_by_epoch_{tag}.png", dpi=DPI)
        plt.close(fig)

    # 7. prob_drift
    fig, axes = panels("Распределение предсказанной вероятности класса scam\n"
                       "на gold-test по эпохам (объединено по сидам)")
    for ax, m in zip(axes, models):
        eps = sorted({e for s in m["seeds"] for e in s["epochs"]})
        data = [[p for s in m["seeds"] for e in s["per_epoch"] if e["epoch"] == ep
                 for p in e["prob_scam"]] for ep in eps]
        parts = ax.violinplot(data, positions=range(len(eps)), showmedians=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor(MODEL_COLORS[m["name"]])
            pc.set_alpha(0.55)
        ax.set_xticks(range(len(eps)), [str(e) for e in eps])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(m["name"])
        ax.set_xlabel("эпоха")
    axes[0].set_ylabel("вероятность класса scam")
    fig.tight_layout()
    fig.savefig(plots / "prob_drift.png", dpi=DPI)
    plt.close(fig)

    # 8. per_language_by_epoch
    langs = sorted(models[0]["seeds"][0]["per_epoch"][0]["per_language"])
    lang_colors = dict(zip(langs, ["#4C72B0", "#DD8452", "#55A868"]))
    fig, axes = panels(f"F1 по языкам на золотом тесте по эпохам (усреднено по сидам, "
                       f"n={n_test // 3} на язык)")
    for ax, m in zip(axes, models):
        eps = sorted({e for s in m["seeds"] for e in s["epochs"]})
        for lg in langs:
            ys = []
            for ep in eps:
                v = [e["per_language"][lg]["f1"] for s in m["seeds"]
                     for e in s["per_epoch"] if e["epoch"] == ep]
                ys.append(float(np.mean(v)) if v else np.nan)
            ax.plot(eps, ys, marker="o", linewidth=2, color=lang_colors[lg], label=lg)
        ax.set_title(m["name"])
        ax.set_xlabel("эпоха")
        ax.set_xticks(eps)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("F1 (класс scam)")
    fig.tight_layout()
    fig.savefig(plots / "per_language_by_epoch.png", dpi=DPI)
    plt.close(fig)

    # ── сохранение ──
    results = {
        "n_test": n_test,
        "cost_definition": "F1(выбранная эпоха) - F1(лучшая по тесту эпоха), <= 0",
        "degenerate_rule": "recall>0.98 и precision<0.75, либо доля одного класса > 0.9",
        "correlation_dev_test": {k: {"pearson_r": v["pearson_r"], "n_points": v["n_points"]}
                                 for k, v in corr.items()},
        "correlation_dev_test_no_degenerate": corr_clean,
        "adjacent_epoch_sign_agreement": {k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                                          for k, v in sign_stats.items()},
        "models": {},
    }
    for m in models:
        results["models"][m["name"]] = {
            "versions": m["versions"],
            "config": {k: v for k, v in m["config"].items() if k != "versions"},
            "cost_mean": float(np.mean(all_costs[m["name"]])),
            "cost_worst": float(min(all_costs[m["name"]])),
            "cost_per_epoch_mean": float(np.mean(per_epoch_costs[m["name"]])),
            "cost_per_epoch_worst": float(min(per_epoch_costs[m["name"]])),
            "expected_cost_mean": float(np.mean([s["expected_cost"] for s in m["seeds"]
                                                 if "expected_cost" in s])),
            "expected_cost_worst": float(min([s["expected_cost"] for s in m["seeds"]
                                              if "expected_cost" in s])),
            "seeds": [{k: v for k, v in s.items()
                       if k not in ("per_epoch",)} |
                      {"per_epoch": [{k: v for k, v in e.items() if k != "prob_scam"}
                                     for e in s["per_epoch"]]}
                      for s in m["seeds"]],
        }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nверсии окружения:")
    for m in models:
        print(f"  {m['name']}: {m['versions']}")
    print("\nграфики:")
    for f in sorted(plots.glob("*.png")):
        print("  ", f.resolve())
    print("  ", (out_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
