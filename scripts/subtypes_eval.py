#!/usr/bin/env python3
"""Разбор multilabel-головы подтипов: метрики на gold-test и сравнение с silver.

Ничего не обучается: читаются снимки предсказаний по эпохам, снятые на Kaggle.
Чекпоинт выбран по macro-F1 подтипов на gold-dev; здесь считаются метрики этого
чекпоинта на gold-test 303 и то же самое против silver-меток DeepSeek — чтобы
понять, к чьей разметке модель ближе.

  python scripts/subtypes_eval.py
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
LABEL_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def label_metrics(y_true, y_pred, subtypes):
    """Полный набор multilabel-метрик. y_* — бинарные матрицы (n, K)."""
    from sklearn.metrics import (cohen_kappa_score, f1_score, hamming_loss,
                                 precision_recall_fscore_support)
    out = {"per_label": {}}
    for i, s in enumerate(subtypes):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", pos_label=1, zero_division=0)
        out["per_label"][s] = {
            "precision": float(p), "recall": float(r), "f1": float(f),
            "support": int(y_true[:, i].sum()), "predicted": int(y_pred[:, i].sum()),
            "cohen_kappa": float(cohen_kappa_score(y_true[:, i], y_pred[:, i])),
            "confusion": [[int(((y_pred[:, i] == c) & (y_true[:, i] == g)).sum())
                           for c in (0, 1)] for g in (0, 1)],
        }
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_kappa"] = float(np.mean([out["per_label"][s]["cohen_kappa"] for s in subtypes]))
    out["subset_accuracy"] = float((y_true == y_pred).all(axis=1).mean())
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/subtypes/data")
    ap.add_argument("--out", default="runs/subtypes")
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    raw = json.loads((data / "results_raw.json").read_text(encoding="utf-8"))
    SUB = raw["subtypes"]
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
        y_gold = np.array(t["per_epoch"][0]["y_true"])
        y_silver = np.array(t["y_silver"])
        sel = chosen[seed]
        i = t["epochs"].index(sel)
        y_pred = np.array(t["per_epoch"][i]["y_pred"])
        probs = np.array(t["per_epoch"][i]["probs"])
        runs.append({
            "seed": seed, "epochs": t["epochs"], "chosen_epoch": sel,
            "y_gold": y_gold, "y_silver": y_silver, "y_pred": y_pred, "probs": probs,
            "y_dev_gold": np.array(dv["per_epoch"][0]["y_true"]),
            "dev_probs": np.array(dv["per_epoch"][dv["epochs"].index(sel)]["probs"]),
            "vs_gold": label_metrics(y_gold, y_pred, SUB),
            "vs_silver": label_metrics(y_silver, y_pred, SUB),
            "macro_by_epoch_test": [label_metrics(y_gold, np.array(pe["y_pred"]), SUB)["macro_f1"]
                                    for pe in t["per_epoch"]],
            "macro_by_epoch_dev": [label_metrics(np.array(dv["per_epoch"][0]["y_true"]),
                                                 np.array(pe["y_pred"]), SUB)["macro_f1"]
                                   for pe in dv["per_epoch"]],
        })
    n_test = len(runs[0]["y_gold"])
    y_gold = runs[0]["y_gold"]
    y_silver = runs[0]["y_silver"]

    # ── согласованность самой разметки ──
    from sklearn.metrics import f1_score
    gold_scam_no_sub = int(((y_gold.sum(1) == 0)).sum())
    silver_vs_gold = label_metrics(y_gold, y_silver, SUB)

    print(f"gold-test, n={n_test}. Отбор чекпоинта по macro-F1 подтипов на gold-dev.")
    print(f"лучшие эпохи: {raw['best_epochs']}, отработано {raw['epochs_ran']} из "
          f"{raw['config']['epochs']}\n")

    print("СОГЛАСОВАННОСТЬ РАЗМЕТКИ (не модель):")
    print(f"  диалогов gold БЕЗ единого подтипа: {gold_scam_no_sub} из {n_test}")
    print(f"  silver против gold на этих же 303: macro-F1 {silver_vs_gold['macro_f1']:.4f}, "
          f"macro-κ {silver_vs_gold['macro_kappa']:.4f}, "
          f"полное совпадение множества {silver_vs_gold['subset_accuracy']:.4f}")
    for s in SUB:
        m = silver_vs_gold["per_label"][s]
        print(f"    {s:15s} F1 {m['f1']:.4f}  κ {m['cohen_kappa']:.4f}  "
              f"(gold {m['support']}, silver {m['predicted']})")

    # ── метрики модели ──
    def agg(key, path):
        vals = [r[key] for r in runs]
        arr = [v[path[0]][path[1]][path[2]] if len(path) == 3 else v[path[0]] for v in vals]
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    print(f"\nМОДЕЛЬ ПРОТИВ GOLD (n={n_test}, среднее ± std по 3 сидам):")
    print(f"{'класс':>15} {'precision':>16} {'recall':>16} {'F1':>16} {'κ':>16} {'support':>8}")
    per_label_agg = {}
    for s in SUB:
        row = {}
        for met in ("precision", "recall", "f1", "cohen_kappa"):
            row[met] = agg("vs_gold", ("per_label", s, met))
        row["support"] = runs[0]["vs_gold"]["per_label"][s]["support"]
        row["predicted"] = agg("vs_gold", ("per_label", s, "predicted"))
        per_label_agg[s] = row
        print(f"{s:>15} " + " ".join(f"{row[m][0]:>9.4f}±{row[m][1]:.4f}"
                                     for m in ("precision", "recall", "f1", "cohen_kappa")) +
              f" {row['support']:>8}")

    overall = {}
    for met in ("macro_f1", "micro_f1", "macro_kappa", "subset_accuracy", "hamming_loss"):
        overall[met] = agg("vs_gold", (met,))
        print(f"  {met:>18}: {overall[met][0]:.4f} ± {overall[met][1]:.4f}")

    print("\nМОДЕЛЬ ПРОТИВ SILVER (та же выборка):")
    overall_s = {}
    for met in ("macro_f1", "micro_f1", "macro_kappa", "subset_accuracy", "hamming_loss"):
        overall_s[met] = agg("vs_silver", (met,))
        print(f"  {met:>18}: {overall_s[met][0]:.4f} ± {overall_s[met][1]:.4f}")
    print(f"  {'ближе к':>18}: "
          f"{'SILVER' if overall_s['macro_f1'][0] > overall['macro_f1'][0] else 'GOLD'} "
          f"(macro-F1 {overall_s['macro_f1'][0]:.4f} против {overall['macro_f1'][0]:.4f})")

    # ── порог по классам ──
    ths = np.linspace(0.02, 0.98, 97)
    thr = {}
    for i, s in enumerate(SUB):
        curve = []
        for t in ths:
            f = np.mean([f1_score(r["y_gold"][:, i], (r["probs"][:, i] >= t).astype(int),
                                  zero_division=0) for r in runs])
            curve.append(float(f))
        bi = int(np.argmax(curve))
        f05 = float(np.mean([f1_score(r["y_gold"][:, i], (r["probs"][:, i] >= 0.5).astype(int),
                                      zero_division=0) for r in runs]))
        # Порог, подобранный по gold-dev — ЕДИНСТВЕННЫЙ легитимный (правило: тест не
        # участвует в решениях). Кривая по тесту ниже — только диагностика потолка.
        dev_curve = [float(np.mean([f1_score(r["y_dev_gold"][:, i],
                                             (r["dev_probs"][:, i] >= t).astype(int),
                                             zero_division=0) for r in runs])) for t in ths]
        t_dev = float(ths[int(np.argmax(dev_curve))])
        f_test_at_dev = float(np.mean([f1_score(r["y_gold"][:, i],
                                                (r["probs"][:, i] >= t_dev).astype(int),
                                                zero_division=0) for r in runs]))
        def at(t):
            """precision / recall / доля положительных на тесте при пороге t."""
            from sklearn.metrics import precision_score, recall_score
            pr = [(precision_score(r["y_gold"][:, i], (r["probs"][:, i] >= t).astype(int),
                                   zero_division=0),
                   recall_score(r["y_gold"][:, i], (r["probs"][:, i] >= t).astype(int),
                                zero_division=0),
                   float((r["probs"][:, i] >= t).mean())) for r in runs]
            return tuple(float(np.mean([x[k] for x in pr])) for k in range(3))

        # 1c: разумный порог = максимум F1 при доле положительных не выше
        # доли класса в gold-dev x 1.5. Без ограничения порог 0.03 просто
        # объявляет «да» почти всем и F1 растёт за счёт recall.
        dev_rate = float(np.mean([r["y_dev_gold"][:, i].mean() for r in runs]))
        cap = dev_rate * 1.5
        ok = [k for k, t in enumerate(ths)
              if float(np.mean([(r["dev_probs"][:, i] >= t).mean() for r in runs])) <= cap]
        t_con = float(ths[max(ok, key=lambda k: dev_curve[k])]) if ok else 0.5
        f_test_con = float(np.mean([f1_score(r["y_gold"][:, i],
                                             (r["probs"][:, i] >= t_con).astype(int),
                                             zero_division=0) for r in runs]))
        p_dev, r_dev, pos_dev = at(t_dev)
        p_con, r_con, pos_con = at(t_con)
        p05, r05, pos05 = at(0.5)
        thr[s] = {"curve": curve, "best_threshold_on_test": float(ths[bi]),
                  "best_f1_on_test": curve[bi], "f1_at_0.5": f05,
                  "precision_at_0.5": p05, "recall_at_0.5": r05, "pos_rate_at_0.5": pos05,
                  "threshold_from_dev": t_dev, "test_f1_at_dev_threshold": f_test_at_dev,
                  "precision_at_dev_thr": p_dev, "recall_at_dev_thr": r_dev,
                  "pos_rate_at_dev_thr": pos_dev,
                  "dev_class_rate": dev_rate, "pos_rate_cap": cap,
                  "degenerate_dev_thr": bool(pos_dev > 0.8),
                  "threshold_constrained": t_con, "test_f1_constrained": f_test_con,
                  "precision_constrained": p_con, "recall_constrained": r_con,
                  "pos_rate_constrained": pos_con,
                  "degenerate_constrained": bool(pos_con > 0.8)}
        flag = "  ⚠ ВЫРОЖДЕН" if pos_dev > 0.8 else ""
        print(f"  {s:15s}")
        print(f"      порог 0.5      : F1 {f05:.4f}  P {p05:.4f}  R {r05:.4f}  "
              f"доля «да» {pos05:.3f}")
        print(f"      порог с dev {t_dev:.2f}: F1 {f_test_at_dev:.4f}  P {p_dev:.4f}  "
              f"R {r_dev:.4f}  доля «да» {pos_dev:.3f}{flag}")
        print(f"      с ограничением (доля «да» ≤ {cap:.3f} = доля класса в dev "
              f"{dev_rate:.3f} × 1.5), порог {t_con:.2f}:")
        print(f"                       F1 {f_test_con:.4f}  P {p_con:.4f}  R {r_con:.4f}  "
              f"доля «да» {pos_con:.3f}"
              + ("  ⚠ ВЫРОЖДЕН" if pos_con > 0.8 else ""))

    # ─────────────── графики ───────────────
    x = np.arange(len(SUB))

    # 1. per_label_metrics
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, (met, lab, c) in enumerate((("precision", "precision", "#55A868"),
                                       ("recall", "recall", "#C44E52"),
                                       ("f1", "F1", "#4C72B0"))):
        mu = [per_label_agg[s][met][0] for s in SUB]
        sd = [per_label_agg[s][met][1] for s in SUB]
        ax.bar(x + (k - 1) * 0.26, mu, 0.26, yerr=sd, capsize=4, color=c, label=lab)
    for xi, s in zip(x, SUB):
        ax.text(xi, 1.02, f"gold n={per_label_agg[s]['support']}", ha="center", fontsize=10)
    ax.set_xticks(x, SUB)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("подтип")
    ax.set_ylabel("значение метрики")
    ax.set_title(f"Подтипы на золотом тесте (n={n_test}): precision, recall, F1\n"
                 "усы — стандартное отклонение по 3 сидам")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots / "per_label_metrics.png", dpi=DPI)
    plt.close(fig)

    # 2. label_confusion
    fig, axes = plt.subplots(1, len(SUB), figsize=(4.3 * len(SUB), 4.4))
    for ax, s in zip(axes, SUB):
        cm = np.mean([np.array(r["vs_gold"]["per_label"][s]["confusion"]) for r in runs], axis=0)
        im = ax.imshow(cm, cmap="Blues")
        rs = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}\n{cm[i, j] / rs[i, 0]:.0%}", ha="center",
                        va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "#222")
        ax.set_xticks([0, 1], ["нет", "есть"])
        ax.set_yticks([0, 1], ["нет", "есть"])
        ax.set_xlabel("предсказание")
        ax.set_title(f"{s}\nF1 {per_label_agg[s]['f1'][0]:.4f}")
        ax.grid(False)
    axes[0].set_ylabel("gold")
    fig.suptitle(f"Матрицы ошибок по подтипам на золотом тесте (n={n_test}), среднее по 3 сидам")
    fig.tight_layout()
    fig.savefig(plots / "label_confusion.png", dpi=DPI)
    plt.close(fig)

    # 3. label_cooccurrence
    def cooc(y):
        return np.array([[int(((y[:, i] == 1) & (y[:, j] == 1)).sum())
                          for j in range(len(SUB))] for i in range(len(SUB))])
    co_gold = cooc(y_gold)
    co_pred = np.mean([cooc(r["y_pred"]) for r in runs], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, mat, title in ((axes[0], co_gold, "gold"),
                           (axes[1], co_pred, "предсказано моделью (среднее по сидам)")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(co_gold.max(), co_pred.max()))
        for i in range(len(SUB)):
            for j in range(len(SUB)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=13,
                        color="white" if mat[i, j] > mat.max() / 2 else "#222")
        ax.set_xticks(range(len(SUB)), SUB, rotation=20, ha="right")
        ax.set_yticks(range(len(SUB)), SUB)
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle(f"Совместная встречаемость подтипов на золотом тесте (n={n_test})\n"
                 "по диагонали — сколько раз класс встретился всего")
    fig.colorbar(im, ax=axes, shrink=0.8, label="диалогов")
    fig.savefig(plots / "label_cooccurrence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # 4. threshold_per_label
    fig, ax = plt.subplots(figsize=(11, 6))
    for s, c in zip(SUB, LABEL_COLORS):
        ax.plot(ths, thr[s]["curve"], color=c, linewidth=2.2, label=s)
        ax.axvline(thr[s]["best_threshold_on_test"], color=c, linestyle=":", linewidth=1.8)
    ax.axvline(0.5, color="#666", linestyle="--", linewidth=1.6, label="порог 0.5")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("порог сигмоиды")
    ax.set_ylabel("F1 класса на золотом тесте")
    for s, c in zip(SUB, LABEL_COLORS):
        ax.axvline(thr[s]["threshold_from_dev"], color=c, linestyle="-.", linewidth=1.4)
    ax.set_title("F1 каждого подтипа на ТЕСТЕ как функция порога (среднее по 3 сидам)\n"
                 "точки — порог, максимизирующий тест (⚠ диагностика, не результат); "
                 "штрихпунктир — порог, выбранный по gold-dev: "
                 + ", ".join(f"{s} {thr[s]['threshold_from_dev']:.2f}" for s in SUB))
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(plots / "threshold_per_label.png", dpi=DPI)
    plt.close(fig)

    # 5. dev_vs_test_macro
    fig, ax = plt.subplots(figsize=(11, 6))
    for r, c in zip(runs, SEED_COLORS):
        ax.plot(r["epochs"], r["macro_by_epoch_dev"], marker="s", linestyle="--",
                color=c, linewidth=2, label=f"gold-dev, сид {r['seed']}")
        ax.plot(r["epochs"], r["macro_by_epoch_test"], marker="o", linestyle="-",
                color=c, linewidth=2, label=f"gold-test, сид {r['seed']}")
        i = r["epochs"].index(r["chosen_epoch"])
        ax.scatter([r["chosen_epoch"]], [r["macro_by_epoch_dev"][i]], s=230,
                   facecolors="none", edgecolors=c, linewidths=2.6, zorder=5)
    ax.set_xticks(runs[0]["epochs"])
    ax.set_xlabel("эпоха")
    ax.set_ylabel("macro-F1 подтипов")
    ax.set_title("Macro-F1 подтипов по эпохам: gold-dev (пунктир) и gold-test (сплошная)\n"
                 "кружок — эпоха, выбранная по dev")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots / "dev_vs_test_macro.png", dpi=DPI)
    plt.close(fig)

    # 6. gold_vs_silver_agreement
    fig, ax = plt.subplots(figsize=(11, 6))
    g = [np.mean([r["vs_gold"]["per_label"][s]["f1"] for r in runs]) for s in SUB]
    sv = [np.mean([r["vs_silver"]["per_label"][s]["f1"] for r in runs]) for s in SUB]
    gs = [np.std([r["vs_gold"]["per_label"][s]["f1"] for r in runs], ddof=1) for s in SUB]
    ss = [np.std([r["vs_silver"]["per_label"][s]["f1"] for r in runs], ddof=1) for s in SUB]
    ax.bar(x - 0.2, g, 0.4, yerr=gs, capsize=4, color="#4C72B0",
           label="согласие с gold (человек)")
    ax.bar(x + 0.2, sv, 0.4, yerr=ss, capsize=4, color="#937860",
           label="согласие с silver (DeepSeek)")
    for xi, (a, b) in enumerate(zip(g, sv)):
        ax.text(xi - 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=10)
        ax.text(xi + 0.2, b + 0.02, f"{b:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, SUB)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("подтип")
    ax.set_ylabel("F1 согласия")
    ax.set_title("К чьей разметке ближе модель: к человеку или к DeepSeek\n"
                 f"золотой тест, n={n_test}, среднее ± std по 3 сидам")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots / "gold_vs_silver_agreement.png", dpi=DPI)
    plt.close(fig)

    # ── сохранение ──
    res = {
        "n_test": n_test, "subtypes": SUB,
        "config": raw["config"], "split": raw["split"],
        "best_epochs": raw["best_epochs"], "epochs_ran": raw["epochs_ran"],
        "label_consistency": {"gold_dialogs_without_any_subtype": gold_scam_no_sub,
                              "silver_vs_gold": silver_vs_gold},
        "vs_gold": {"per_label": {s: {k: list(v) if isinstance(v, tuple) else v
                                      for k, v in per_label_agg[s].items()} for s in SUB},
                    "overall": {k: list(v) for k, v in overall.items()}},
        "vs_silver": {"overall": {k: list(v) for k, v in overall_s.items()},
                      "per_label_f1": {s: [float(np.mean([r["vs_silver"]["per_label"][s]["f1"]
                                                          for r in runs])),
                                           float(np.std([r["vs_silver"]["per_label"][s]["f1"]
                                                         for r in runs], ddof=1))] for s in SUB}},
        "thresholds": {s: {k: v for k, v in thr[s].items() if k != "curve"} for s in SUB},
        "per_seed": [{"seed": r["seed"], "chosen_epoch": r["chosen_epoch"],
                      "macro_by_epoch_dev": r["macro_by_epoch_dev"],
                      "macro_by_epoch_test": r["macro_by_epoch_test"],
                      "vs_gold": r["vs_gold"], "vs_silver": r["vs_silver"]} for r in runs],
    }
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\nграфики:")
    for f in sorted(plots.glob("*.png")):
        print("  ", f.resolve())
    print("  ", (out_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
