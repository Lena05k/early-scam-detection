#!/usr/bin/env python3
"""Бейзлайн is_scam: TF-IDF + логистическая регрессия, отбор по gold-dev.

Та же схема, что у трансформерного прогона: обучение на train (silver-метки),
ОТБОР модели по gold-dev (человеческие метки), финальный замер на gold-test.
Разбиение gold берётся готовым из gold_dev_split.json и не пересоздаётся.

Аналог «эпохи» здесь — сила регуляризации C: перебирается сетка, лучшее
значение выбирается по gold-dev F1. Для сравнения считается и то, что выбрал бы
silver_val — это прямой аналог расхождения метрик, найденного у трансформера.

⚠ Методичка §2a: язык-специфичный TF-IDF слаб для переноса и включается ТОЛЬКО
как in-language бейзлайн. Разбивку по языкам нельзя читать как кросс-язычный
трансфер — модель видела все три языка в обучении.

  python scripts/train_tfidf.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import build_text, read_corpus, read_gold_csv  # noqa: E402

C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def metrics_from(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--gold", default="data/gold/gold_test_v1.csv")
    p.add_argument("--gold-dev", default="data/gold/gold_dev_split.json")
    p.add_argument("--out-dir", default="runs/tfidf_gold_dev")
    p.add_argument("--seeds", default="20260721,20260722,20260723")
    args = p.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, roc_auc_score)

    seeds = [int(s) for s in args.seeds.split(",")]
    records = read_corpus(args.corpus)
    gold = read_gold_csv(args.gold)

    train_recs = [r for r in records if r.get("split") == "train"]
    val_recs = [r for r in records if r.get("split") == "val"]

    all_test = []
    for r in records:
        if r.get("split") != "test":
            continue
        r2 = {k: v for k, v in r.items() if k != "is_scam"}   # silver-метку выбрасываем
        if r2["id"] in gold:
            r2["gold_is_scam"] = gold[r2["id"]]
            all_test.append(r2)
    if len(all_test) != len(gold):
        sys.exit(f"❌ джойн неполный: {len(all_test)} из {len(gold)}")

    sm = json.loads(pathlib.Path(args.gold_dev).read_text(encoding="utf-8"))
    assign = sm["assignment"]
    dev_recs = [r for r in all_test if assign.get(r["id"]) == "dev"]
    test_recs = [r for r in all_test if assign.get(r["id"]) != "dev"]
    assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"], "разбиение не совпало"
    gd = {r.get("scenario_group") for r in dev_recs}
    gt = {r.get("scenario_group") for r in test_recs}
    assert not (gd & gt), "группы scenario_group разорваны между dev и test"

    print(f"разбиение gold: seed={sm['seed']}  dev={len(dev_recs)}  test={len(test_recs)}  "
          f"(групп {len(gd)}/{len(gt)}, пересечение 0)")
    print(f"train {len(train_recs)} (silver) / val {len(val_recs)} (silver) / "
          f"gold-dev {len(dev_recs)} / gold-test {len(test_recs)}")

    X_train_txt = [build_text(r) for r in train_recs]
    y_train = np.array([int(r["is_scam"]) for r in train_recs])
    X_val_txt = [build_text(r) for r in val_recs]
    y_val = np.array([int(r["is_scam"]) for r in val_recs])
    X_dev_txt = [build_text(r) for r in dev_recs]
    y_dev = np.array([r["gold_is_scam"] for r in dev_recs])
    X_test_txt = [build_text(r) for r in test_recs]
    y_test = np.array([r["gold_is_scam"] for r in test_recs])

    runs = []
    for seed in seeds:
        # char_wb n-граммы — как в scenario_groups.py; для zh пословная токенизация
        # не работает (нет пробелов), поэтому символьные n-граммы обязательны
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True)
        Xtr = vec.fit_transform(X_train_txt)          # IDF считается ТОЛЬКО по train
        Xva, Xdev, Xte = (vec.transform(t) for t in (X_val_txt, X_dev_txt, X_test_txt))

        grid = []
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
            clf.fit(Xtr, y_train)
            grid.append({
                "C": C,
                "silver_val_f1": float(f1_score(y_val, clf.predict(Xva), pos_label=1,
                                                zero_division=0)),
                "gold_dev_f1": float(f1_score(y_dev, clf.predict(Xdev), pos_label=1,
                                              zero_division=0)),
            })

        best_gold = max(grid, key=lambda g: g["gold_dev_f1"])
        best_silver = max(grid, key=lambda g: g["silver_val_f1"])

        def eval_at(C):
            clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
            clf.fit(Xtr, y_train)
            yp = clf.predict(Xte)
            pr = clf.predict_proba(Xte)[:, 1]
            m = metrics_from(y_test, yp)
            m["roc_auc"] = float(roc_auc_score(y_test, pr))
            m["average_precision"] = float(average_precision_score(y_test, pr))
            per_lang = {}
            for lg in sorted({r.get("lang") for r in test_recs if r.get("lang")}):
                idx = [i for i, r in enumerate(test_recs) if r.get("lang") == lg]
                per_lang[lg] = metrics_from(y_test[idx], yp[idx]) | {"n": len(idx)}
            silver = {r["id"]: int(r["is_scam"]) for r in records}
            agree = sum(1 for r, q in zip(test_recs, yp) if silver[r["id"]] == int(q))
            return (m, confusion_matrix(y_test, yp, labels=[0, 1]).tolist(), per_lang,
                    agree / len(test_recs), vec.transform(X_test_txt).shape[1])

        m_g, cm_g, pl_g, ag_g, nfeat = eval_at(best_gold["C"])
        m_s, cm_s, _, ag_s, _ = eval_at(best_silver["C"])

        runs.append({"seed": seed, "grid": grid, "n_features": int(Xtr.shape[1]),
                     "selected_by_gold_dev": {"C": best_gold["C"],
                                              "gold_dev_f1": best_gold["gold_dev_f1"],
                                              "test_metrics": m_g, "confusion_matrix": cm_g,
                                              "per_language": pl_g, "silver_agreement": ag_g},
                     "selected_by_silver_val": {"C": best_silver["C"],
                                                "silver_val_f1": best_silver["silver_val_f1"],
                                                "test_metrics": m_s, "confusion_matrix": cm_s,
                                                "silver_agreement": ag_s}})
        print(f"[seed {seed}] признаков {Xtr.shape[1]} | "
              f"C по gold-dev {best_gold['C']} → gold-test F1 {m_g['f1']:.4f} | "
              f"C по silver-val {best_silver['C']} → gold-test F1 {m_s['f1']:.4f}")

    # ── таблица перебора ──
    print(f"\n{'C':>7} " + " ".join(f"{'s'+str(s)[-4:]+' silver':>13} {'gold_dev':>9}" for s in seeds))
    for i, C in enumerate(C_GRID):
        row = f"{C:>7} "
        for r in runs:
            g = r["grid"][i]
            row += f"{g['silver_val_f1']:>13.4f} {g['gold_dev_f1']:>9.4f}"
        print(row)

    KEYS = ["accuracy", "precision", "recall", "f1", "cohen_kappa", "roc_auc",
            "average_precision"]
    summary = {}
    print(f"\ngold-test (n={len(test_recs)}), отбор по gold-dev:")
    print(f"{'метрика':>20} {'среднее':>10} {'± std':>9}")
    for k in KEYS:
        v = [r["selected_by_gold_dev"]["test_metrics"][k] for r in runs]
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary[k] = {"mean": mu, "std": sd, "values": v}
        print(f"{k:>20} {mu:>10.4f} {sd:>9.4f}")
    sa = [r["selected_by_gold_dev"]["silver_agreement"] for r in runs]
    summary["silver_agreement"] = {"mean": float(np.mean(sa)),
                                   "std": float(np.std(sa, ddof=1)) if len(sa) > 1 else 0.0,
                                   "values": sa}
    print(f"{'совпадение с silver':>20} {np.mean(sa):>10.4f} "
          f"{(np.std(sa, ddof=1) if len(sa) > 1 else 0.0):>9.4f}")

    f_g = [r["selected_by_gold_dev"]["test_metrics"]["f1"] for r in runs]
    f_s = [r["selected_by_silver_val"]["test_metrics"]["f1"] for r in runs]
    print(f"\nотбор по gold-dev   → gold-test F1 {np.mean(f_g):.4f}")
    print(f"отбор по silver-val → gold-test F1 {np.mean(f_s):.4f}   "
          f"(разница {np.mean(f_g)-np.mean(f_s):+.4f})")

    print("\nпо языкам (отбор по gold-dev, среднее ± std) — ⚠ in-language, НЕ трансфер:")
    for lg in sorted(runs[0]["selected_by_gold_dev"]["per_language"]):
        out = f"  {lg}:"
        for met in ("f1", "precision", "recall"):
            v = [r["selected_by_gold_dev"]["per_language"][lg][met] for r in runs]
            out += f"  {met} {np.mean(v):.4f}±{(np.std(v, ddof=1) if len(v) > 1 else 0.0):.4f}"
        print(out + f"   (n={runs[0]['selected_by_gold_dev']['per_language'][lg]['n']})")

    cm_mean = np.mean([np.array(r["selected_by_gold_dev"]["confusion_matrix"])
                       for r in runs], axis=0)
    print("\nусреднённая матрица ошибок (строки gold, столбцы предсказание):")
    print(f"  gold=0  {cm_mean[0,0]:8.1f}  {cm_mean[0,1]:8.1f}")
    print(f"  gold=1  {cm_mean[1,0]:8.1f}  {cm_mean[1,1]:8.1f}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({
        "model": "tfidf_char_wb_2_5 + LogisticRegression",
        "vectorizer": {"analyzer": "char_wb", "ngram_range": [2, 5], "min_df": 2,
                       "sublinear_tf": True, "fit_on": "train only"},
        "C_grid": C_GRID, "seeds": seeds,
        "split": {"file": args.gold_dev, "seed": sm["seed"],
                  "n_dev": len(dev_recs), "n_test": len(test_recs)},
        "runs": runs, "summary": summary,
        "f1_selected_by_gold_dev": f_g, "f1_selected_by_silver_val": f_s,
        "confusion_matrix_mean": cm_mean.tolist(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
