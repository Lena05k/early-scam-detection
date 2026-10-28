#!/usr/bin/env python3
"""Графики обучения из runs/binary_xlmr/history.json и test_predictions.csv.

Все точки — только из реальных артефактов прогона. Ничего не сглаживается сверх
явно подписанного скользящего среднего и ничего не дорисовывается. Если данных
для графика нет — график НЕ строится, вместо него печатается, какого поля не хватило.

Перед построением проверяется, что прогон полноценный: >= 2 эпох валидации в
history.json и >= 400 строк в test_predictions.csv. Иначе графики не строятся.

  python scripts/make_train_plots.py --run-dir runs/binary_xlmr
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
FIGSIZE = (10, 6)
STYLE = "seaborn-v0_8-whitegrid"

MIN_EVAL_EPOCHS = 2
MIN_TEST_ROWS = 400


# ─────────────────────────── загрузка ───────────────────────────

def load_history(run_dir):
    p = run_dir / "history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_predictions(run_dir):
    p = run_dir / "test_predictions.csv"
    if not p.exists():
        return None
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rolling_mean(xs, window):
    """Скользящее среднее по `window` точкам; хвост короче окна усредняется как есть."""
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def epoch_train_loss(tr_pts, ev_pts):
    """Средний train loss внутри каждой эпохи — для сопоставления с val loss."""
    out = {}
    for e in ev_pts:
        ep = e["epoch"]
        vals = [x["loss"] for x in tr_pts if ep - 1 < x["epoch"] <= ep]
        if vals:
            out[ep] = sum(vals) / len(vals)
    return out


# ─────────────────────────── основное ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/binary_xlmr")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    hist = load_history(run_dir)
    preds = load_predictions(run_dir)

    ev = [h for h in (hist or []) if "eval_loss" in h]
    n_rows = len(preds or [])

    # ── обязательная проверка: это не дымовой тест ──
    if len(ev) < MIN_EVAL_EPOCHS or n_rows < MIN_TEST_ROWS:
        print("это дымовой тест, графики строить нечего")
        print(f"  эпох валидации в history.json: {len(ev)} (нужно >= {MIN_EVAL_EPOCHS})")
        print(f"  строк в test_predictions.csv:  {n_rows} (нужно >= {MIN_TEST_ROWS})")
        return

    plt.style.use(STYLE)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    tr = [h for h in hist if "loss" in h and "eval_loss" not in h]
    lrpts = [h for h in hist if "learning_rate" in h]
    best = max(ev, key=lambda h: h["eval_f1"])
    best_ep, best_f1 = best["epoch"], best["eval_f1"]
    steps_per_epoch = tr[-1]["step"] / tr[-1]["epoch"] if tr and tr[-1].get("epoch") else None

    y_true = np.array([int(r["gold_is_scam"]) for r in preds])
    y_pred = np.array([int(r["pred_is_scam"]) for r in preds])
    probs = np.array([float(r["prob_scam"]) for r in preds])

    created, skipped = [], []

    def save(fig, name):
        fig.tight_layout()
        f = plots / name
        fig.savefig(f, dpi=DPI)
        plt.close(fig)
        created.append(f)

    def vline_best(ax, as_step=False):
        x = best_ep * steps_per_epoch if (as_step and steps_per_epoch) else best_ep
        ax.axvline(x, color="#C44E52", linestyle="--", linewidth=1.8,
                   label=f"лучший чекпоинт: эпоха {best_ep:g} (val F1 = {best_f1:.4f})")

    # ── a) learning_curve ──
    if tr:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        steps = [h["step"] for h in tr]
        losses = [h["loss"] for h in tr]
        ax.plot(steps, losses, color="#4C72B0", linewidth=0.9, alpha=0.35,
                label="train loss (по шагам)")
        ax.plot(steps, rolling_mean(losses, 10), color="#4C72B0", linewidth=2.6,
                label="train loss, скользящее среднее по 10 точкам")
        ax.plot([h["step"] for h in ev], [h["eval_loss"] for h in ev],
                color="#DD8452", marker="o", markersize=7, linewidth=2,
                label="val loss (по эпохам)")
        vline_best(ax, as_step=True)
        ax.set_xlabel("шаг обучения")
        ax.set_ylabel("функция потерь")
        ax.set_title("Кривая обучения: потери на train (по шагам) и на val (по эпохам)")
        ax.legend()
        if steps_per_epoch:
            sec = ax.secondary_xaxis("top",
                                     functions=(lambda s: s / steps_per_epoch,
                                                lambda e: e * steps_per_epoch))
            sec.set_xlabel("эпоха")
            sec.set_xticks([h["epoch"] for h in ev])
        save(fig, "learning_curve.png")
    else:
        skipped.append("learning_curve.png — в history.json нет точек с полем 'loss'")

    # ── b) val_metrics ──
    keys = [("eval_accuracy", "accuracy", "#4C72B0"),
            ("eval_f1", "F1", "#DD8452"),
            ("eval_precision", "precision", "#55A868"),
            ("eval_recall", "recall", "#C44E52"),
            ("eval_cohen_kappa", "Cohen κ", "#8172B2")]
    have = [(k, lab, c) for k, lab, c in keys if any(k in h for h in ev)]
    missing = [lab for k, lab, _ in keys if not any(k in h for h in ev)]
    if have:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for k, lab, c in have:
            pts = [(h["epoch"], h[k]) for h in ev if k in h]
            ax.plot([x for x, _ in pts], [y for _, y in pts],
                    marker="o", markersize=6, color=c, linewidth=2, label=lab)
        vline_best(ax)
        ax.set_ylim(0, 1)
        ax.set_xticks([h["epoch"] for h in ev])
        ax.set_xlabel("эпоха")
        ax.set_ylabel("значение метрики")
        ax.set_title("Метрики на валидации по эпохам")
        ax.legend(loc="lower right")
        save(fig, "val_metrics.png")
        if missing:
            skipped.append(f"val_metrics.png построен без: {', '.join(missing)} — нет в history.json")
    else:
        skipped.append("val_metrics.png — в history.json нет ни одной метрики eval_*")

    # ── c) overfit_gap ──
    etl = epoch_train_loss(tr, ev)
    if etl:
        eps = sorted(etl)
        gap = [etl[e] - next(h["eval_loss"] for h in ev if h["epoch"] == e) for e in eps]
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(eps, gap, marker="o", markersize=7, color="#937860", linewidth=2.2,
                label="train loss − val loss")
        ax.axhline(0, color="#666", linewidth=1.2, linestyle="-")
        vline_best(ax)
        ax.set_xticks(eps)
        ax.set_xlabel("эпоха")
        ax.set_ylabel("разрыв потерь (train − val)")
        ax.set_title("Разрыв между train и val loss по эпохам: момент расхождения\n"
                     "(train loss усреднён внутри эпохи; выше нуля — val лучше train)")
        ax.legend()
        save(fig, "overfit_gap.png")
    else:
        skipped.append("overfit_gap.png — не удалось сопоставить train loss с эпохами")

    # ── d) confusion_matrix ──
    cm = np.zeros((2, 2), dtype=int)
    for t, p_ in zip(y_true, y_pred):
        cm[t, p_] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.1%} строки", ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xticks([0, 1], ["benign (0)", "scam (1)"])
    ax.set_yticks([0, 1], ["benign (0)", "scam (1)"])
    ax.set_xlabel("предсказание")
    ax.set_ylabel("gold")
    ax.set_title(f"Матрица ошибок на золотом тесте (n={len(y_true)}): "
                 f"абсолютное число и доля от строки")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="число диалогов")
    save(fig, "confusion_matrix.png")

    # ── e) roc_pr_curves ──
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    if len(set(y_true.tolist())) < 2:
        skipped.append("roc_pr_curves.png — в gold-метках только один класс, ROC/PR неопределены")
    else:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        pr, rc, _ = precision_recall_curve(y_true, probs)
        apr = average_precision_score(y_true, probs)
        base = y_true.mean()
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
        a0, a1 = axes
        a0.plot(fpr, tpr, color="#4C72B0", linewidth=2.4, label=f"ROC (AUC = {auc:.4f})")
        a0.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1.4,
                label="случайный классификатор")
        a0.set_xlabel("доля ложных тревог (FPR)")
        a0.set_ylabel("доля верно найденных (TPR)")
        a0.set_title("ROC-кривая")
        a0.legend(loc="lower right")
        a1.plot(rc, pr, color="#DD8452", linewidth=2.4, label=f"PR (AP = {apr:.4f})")
        a1.axhline(base, color="#999", linestyle="--", linewidth=1.4,
                   label=f"случайный классификатор ({base:.3f})")
        a1.set_xlabel("recall")
        a1.set_ylabel("precision")
        a1.set_title("Precision-recall кривая")
        a1.legend(loc="lower left")
        fig.suptitle(f"Качество ранжирования на золотом тесте (n={len(y_true)})")
        save(fig, "roc_pr_curves.png")

    # ── f) threshold_sweep ──
    ths = np.linspace(0, 1, 201)
    P, R, F = [], [], []
    for t in ths:
        yp = (probs >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        P.append(p_)
        R.append(r_)
        F.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    bi = int(np.argmax(F))
    f1_at_05 = F[int(np.argmin(np.abs(ths - 0.5)))]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ths, P, color="#55A868", linewidth=2, label="precision")
    ax.plot(ths, R, color="#C44E52", linewidth=2, label="recall")
    ax.plot(ths, F, color="#4C72B0", linewidth=2.6, label="F1")
    ax.axvline(0.5, color="#666", linestyle="--", linewidth=1.6,
               label=f"порог 0.5 (F1 = {f1_at_05:.4f})")
    ax.axvline(ths[bi], color="#DD8452", linestyle=":", linewidth=2.2,
               label=f"лучший порог {ths[bi]:.3f} (F1 = {F[bi]:.4f})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("порог отсечения по вероятности класса scam")
    ax.set_ylabel("значение метрики")
    ax.set_title("Зависимость precision, recall и F1 от порога отсечения (золотой тест)")
    ax.legend(loc="lower left")
    save(fig, "threshold_sweep.png")

    # ── g) calibration ──
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum():
            xs.append(probs[m].mean())
            ys.append(y_true[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1.6,
            label="идеальная калибровка")
    ax.plot(xs, ys, marker="o", markersize=8, color="#4C72B0", linewidth=2.2,
            label="фактическая доля класса scam")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("средняя предсказанная вероятность класса scam в бине")
    ax.set_ylabel("фактическая доля класса scam в бине")
    ax.set_title(f"Калибровочная кривая на золотом тесте, 10 бинов "
                 f"(заполнено {len(xs)} из 10)")
    ax.legend(loc="upper left")
    save(fig, "calibration.png")

    # ── h) confidence_hist ──
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins20 = np.linspace(0, 1, 21)
    ax.hist(probs[correct], bins=bins20, alpha=0.55, color="#55A868",
            label=f"верно классифицированы (n={int(correct.sum())})")
    ax.hist(probs[~correct], bins=bins20, alpha=0.55, color="#C44E52",
            label=f"ошибки (n={int((~correct).sum())})")
    ax.set_xlabel("предсказанная вероятность класса scam")
    ax.set_ylabel("число диалогов")
    ax.set_title("Уверенность модели на золотом тесте: верные предсказания против ошибок")
    ax.legend()
    save(fig, "confidence_hist.png")

    # ── i) per_language ──
    if any(r.get("lang") for r in preds):
        from sklearn.metrics import precision_recall_fscore_support
        langs = sorted({r["lang"] for r in preds if r.get("lang")})
        vals = {"F1": [], "precision": [], "recall": []}
        ns = []
        for lg in langs:
            m = np.array([r.get("lang") == lg for r in preds])
            p_, r_, f_, _ = precision_recall_fscore_support(
                y_true[m], y_pred[m], average="binary", pos_label=1, zero_division=0)
            vals["F1"].append(f_)
            vals["precision"].append(p_)
            vals["recall"].append(r_)
            ns.append(int(m.sum()))
        x = np.arange(len(langs))
        w = 0.26
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for k, (lab, c) in enumerate(zip(["F1", "precision", "recall"],
                                         ["#4C72B0", "#55A868", "#C44E52"])):
            ax.bar(x + (k - 1) * w, vals[lab], width=w, color=c, label=lab)
        for xi, n in zip(x, ns):
            ax.text(xi, 1.03, f"n={n}", ha="center", fontsize=11)
        ax.set_xticks(x, langs)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("язык")
        ax.set_ylabel("значение метрики (класс scam)")
        ax.set_title("Качество по языкам на золотом тесте (над группой — число примеров)")
        ax.legend(loc="lower right")
        save(fig, "per_language.png")
    else:
        skipped.append("per_language.png — в test_predictions.csv колонка 'lang' пуста")

    # ── j) lr_schedule ──
    if lrpts:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["step"] for h in lrpts], [h["learning_rate"] for h in lrpts],
                color="#8172B2", linewidth=2.4)
        vline_best(ax, as_step=True)
        ax.set_xlabel("шаг обучения")
        ax.set_ylabel("learning rate")
        ax.set_title("Фактический learning rate по шагам (из истории обучения)")
        ax.legend()
        save(fig, "lr_schedule.png")
    else:
        skipped.append("lr_schedule.png — в history.json нет поля 'learning_rate'")

    print(f"создано {len(created)} графиков:")
    for f in created:
        print(f"  {f.resolve()}")
    if skipped:
        print("\nне построено / неполно:")
        for s in skipped:
            print(f"  ⚠ {s}")


if __name__ == "__main__":
    main()
