#!/usr/bin/env python3
"""Бутстрап устойчивости отбора чекпоинта по gold-dev.

Вопрос: у xlm-r лучшая эпоха [1,1,1], у mDeBERTa [2,4,2] — это свойство кривых
обучения или шум gold-dev из 100 диалогов?

Метод: 1000 раз пересэмплируем 100 диалогов gold-dev С ВОЗВРАЩЕНИЕМ, на каждой
выборке пересчитываем F1 по ВСЕМ эпохам и берём argmax. Ключевой момент — одна и
та же выборка индексов применяется ко всем эпохам сразу: это сохраняет корреляцию
между эпохами (они отличаются несколькими диалогами), а независимый бутстрап по
эпохам её ломает и завышает нестабильность.

Ничего не обучается: читаются готовые gold_dev_per_epoch_seed*.json, где на каждую
эпоху лежат пер-диалоговые y_true / y_pred / prob_scam по 100 диалогам.

  python scripts/bootstrap_selection.py
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

N_BOOT = 1000
BOOT_SEED = 20260808          # фиксирован, чтобы бутстрап воспроизводился
DPI = 150
SEED_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def f1_on(y_true, y_pred, idx):
    """F1 класса scam на подвыборке индексов (с учётом кратности)."""
    yt, yp = y_true[idx], y_pred[idx]
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def bootstrap_seed(snap, n_boot, rng):
    """Один сид → распределение выбранной эпохи и интервалы F1 по эпохам."""
    epochs = snap["epochs"]
    y_true = np.array(snap["per_epoch"][0]["y_true"])
    preds = [np.array(pe["y_pred"]) for pe in snap["per_epoch"]]
    n = len(y_true)
    for pe in snap["per_epoch"]:                    # y_true обязан совпадать
        assert pe["y_true"] == snap["per_epoch"][0]["y_true"], "разные метки между эпохами"

    full_f1 = [f1_on(y_true, p, np.arange(n)) for p in preds]
    orig_best = epochs[int(np.argmax(full_f1))]     # argmax → при равенстве ранняя эпоха

    chosen = []
    boot_f1 = np.empty((n_boot, len(epochs)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)            # 100 диалогов с возвращением
        fs = [f1_on(y_true, p, idx) for p in preds]
        boot_f1[b] = fs
        chosen.append(epochs[int(np.argmax(fs))])

    chosen = np.array(chosen)
    dist = {int(e): float((chosen == e).mean()) for e in epochs}
    ci = {int(e): [float(np.percentile(boot_f1[:, i], 2.5)),
                   float(np.percentile(boot_f1[:, i], 97.5))]
          for i, e in enumerate(epochs)}
    return {"seed": snap["seed"], "epochs": epochs,
            "f1_full": [float(x) for x in full_f1],
            "orig_best_epoch": int(orig_best),
            "selection_distribution": dist,
            "hit_rate": float((chosen == orig_best).mean()),
            "f1_ci95": ci,
            "f1_ci95_orig_best": ci[int(orig_best)],
            "n_dev": n, "n_boot": n_boot}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlmr-dir", default="runs/bootstrap_selection/data/xlmr")
    ap.add_argument("--mdeberta-dir", default="runs/bootstrap_selection/data/mdeberta")
    ap.add_argument("--out", default="runs/bootstrap_selection")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--results-name", default="results.json",
                   help="имя файла результатов внутри --out; отдельное имя нужно, "
                        "когда каталог уже занят другим results.json")
    ap.add_argument("--epoch-cost", default=None,
                   help="runs/epoch_cost/results.json — добавит третью панель "
                        "со средней ценой неверного выбора")
    args = ap.parse_args()

    models = [("xlm-roberta-base", pathlib.Path(args.xlmr_dir)),
              ("mDeBERTa-v3-base", pathlib.Path(args.mdeberta_dir))]
    out_dir = pathlib.Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    results = {"n_boot": args.n_boot, "bootstrap_seed": BOOT_SEED, "models": {}}
    for name, d in models:
        snaps = sorted(d.glob("gold_dev_per_epoch_seed*.json"))
        if not snaps:
            raise SystemExit(f"❌ нет снимков gold-dev в {d}")
        # ГСЧ пересоздаётся для каждого сида и каждой модели, поэтому набор
        # бутстрап-выборок один и тот же везде: сравнение сидов и моделей парное,
        # различия не смешиваются с различиями в самих выборках.
        per_seed = [bootstrap_seed(json.loads(p.read_text(encoding="utf-8")),
                                   args.n_boot, np.random.default_rng(BOOT_SEED))
                    for p in snaps]
        results["models"][name] = {"per_seed": per_seed,
                                   "orig_best_epochs": [s["orig_best_epoch"] for s in per_seed],
                                   "hit_rate_mean": float(np.mean([s["hit_rate"] for s in per_seed]))}

    # ── печать ──
    for name, m in results["models"].items():
        print("=" * 78)
        print(f"{name}   исходно выбранные эпохи: {m['orig_best_epochs']}")
        print("=" * 78)
        for s in m["per_seed"]:
            dist = ", ".join(f"эп.{e}: {v:.1%}" for e, v in sorted(s["selection_distribution"].items())
                             if v > 0)
            lo, hi = s["f1_ci95_orig_best"]
            print(f"  сид {s['seed']}: исходная эпоха {s['orig_best_epoch']}, "
                  f"попаданий {s['hit_rate']:.1%}")
            print(f"      распределение выбора: {dist}")
            print(f"      F1 на gold-dev, эпоха {s['orig_best_epoch']}: "
                  f"{s['f1_full'][s['epochs'].index(s['orig_best_epoch'])]:.4f} "
                  f"[95% ДИ {lo:.4f}–{hi:.4f}]")
        print(f"  средняя доля попаданий: {m['hit_rate_mean']:.1%}")
        verdict = ("УСТОЙЧИВ" if m["hit_rate_mean"] >= 0.90 else
                   "НЕУСТОЙЧИВ — отбор существенно зависит от состава gold-dev")
        print(f"  вывод: {verdict}\n")

    # ── график ──
    cost = None
    if args.epoch_cost and pathlib.Path(args.epoch_cost).exists():
        cost = json.loads(pathlib.Path(args.epoch_cost).read_text(encoding="utf-8"))
    n_panels = 3 if cost else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.4))
    for ax, (name, m) in zip(axes[:2], results["models"].items()):
        all_eps = sorted({e for s in m["per_seed"] for e in s["epochs"]})
        w = 0.8 / len(m["per_seed"])
        for k, (s, c) in enumerate(zip(m["per_seed"], SEED_COLORS)):
            vals = [s["selection_distribution"].get(e, 0.0) * 100 for e in all_eps]
            x = np.arange(len(all_eps)) + (k - (len(m["per_seed"]) - 1) / 2) * w
            ax.bar(x, vals, w, color=c,
                   label=f"сид {s['seed']} (исходно эп.{s['orig_best_epoch']}, "
                         f"попаданий {s['hit_rate']:.0%})")
        ax.set_xticks(np.arange(len(all_eps)), [str(e) for e in all_eps])
        ax.set_ylim(0, 105)
        ax.axhline(90, color="#666", linestyle="--", linewidth=1.4)
        ax.text(len(all_eps) - 0.5, 91.5, "порог устойчивости 90%", fontsize=9,
                ha="right", color="#666")
        ax.set_xlabel("выбранная эпоха")
        ax.set_ylabel("доля бутстрап-выборок, %")
        ax.set_title(f"{name}\nисходно: {m['orig_best_epochs']}")
        ax.legend(fontsize=9, loc="upper right")
    if cost:
        ax = axes[2]
        names = list(results["models"])
        x = np.arange(len(names))
        hits = [results["models"][n]["hit_rate_mean"] * 100 for n in names]
        costs = [abs(cost["models"][n]["cost_mean"]) * 100 if n in cost["models"] else 0.0
                 for n in names]
        ax.bar(x - 0.2, hits, 0.4, color="#4C72B0", label="доля попаданий, %")
        ax.bar(x + 0.2, costs, 0.4, color="#C44E52",
               label="средняя цена промаха, п.п. F1")
        for xi, (h, c_) in enumerate(zip(hits, costs)):
            ax.text(xi - 0.2, h + 1.5, f"{h:.0f}%", ha="center", fontsize=10)
            ax.text(xi + 0.2, c_ + 1.5, f"{c_:.2f}", ha="center", fontsize=10)
        ax.axhline(90, color="#666", linestyle="--", linewidth=1.4)
        ax.set_xticks(x, [n.replace("-base", "") for n in names])
        ax.set_ylim(0, 105)
        ax.set_ylabel("проценты / пункты F1")
        ax.set_title("Устойчивость отбора против его цены\n"
                     "промах частый, но дешёвый — или наоборот")
        ax.legend(fontsize=9)
    fig.suptitle(f"Устойчивость отбора чекпоинта: {args.n_boot} бутстрап-выборок "
                 f"gold-dev (100 диалогов с возвращением)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "bootstrap_selection.png", dpi=DPI)
    plt.close(fig)

    (out_dir / args.results_name).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"график:   {(out_dir / 'plots' / 'bootstrap_selection.png').resolve()}")
    print(f"результат: {(out_dir / args.results_name).resolve()}")


if __name__ == "__main__":
    main()
