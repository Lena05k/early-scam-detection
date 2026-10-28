#!/usr/bin/env python3
"""Кросс-язычный перенос для слоя 1 (TF-IDF) и слоя 2 (эмбеддинги e5 + LightGBM).

Обе модели считаются локально: TF-IDF — секунды, слой 2 работает на готовом кэше
эмбеддингов `runs/layer2/cache/multilingual-e5-base_turns.npz` (агрегация mean —
та, что победила по gold-dev в разделе «Слой 2»). Kaggle не нужен.

Для КАЖДОЙ конфигурации из `crosslingual.enumerate_configs()` и каждого сида
перебирается сетка (C у TF-IDF, гиперпараметры у LightGBM) и на КАЖДОЙ точке сетки
снимаются метрики на трёх наборах отбора и на полном gold-test. Сам отбор делается
постфактум, поэтому честный zero-shot (`dev_source`) и контроли (`val_source`,
`dev_full`) сравниваются без единого повторного обучения.

⚠ Слой 1 язык-специфичен по построению (символьные n-граммы), и провал переноса на
zh при обучении на en/ru — ожидаемый результат, а не баг: он показывает цену отказа
от многоязычных представлений.

  python scripts/crosslingual_layers12.py
  python scripts/crosslingual_layers12.py --only tfidf --limit 4    # быстрый прогон
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import (LANGS, SEEDS, build_train, enumerate_configs,  # noqa: E402
                          load_all, metrics_by_lang, selection_sets)
from train_binary import build_text  # noqa: E402

C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
LGB_GRID = [{"n_estimators": n, "num_leaves": lv, "learning_rate": lr}
            for n in (200, 400) for lv in (15, 31) for lr in (0.05, 0.1, 0.2)]
LGB_FIXED = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
             "min_child_samples": 20, "verbose": -1, "n_jobs": -1}
EMB_CACHE = "runs/layer2/cache/multilingual-e5-base_turns.npz"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def load_embeddings(path=EMB_CACHE):
    """Кэш эмбеддингов реплик → {id: вектор диалога} (агрегация mean)."""
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"нет кэша эмбеддингов {p}; сначала "
                         f"`python scripts/train_layer2.py --encode-only`")
    z = np.load(p, allow_pickle=False)
    mat, ids, counts = z["matrix"], list(z["ids"]), z["counts"]
    off = np.concatenate([[0], np.cumsum(counts)])
    return {i: mat[off[k]:off[k + 1]].mean(axis=0) for k, i in enumerate(ids)}


def y_of(recs, gold=False):
    key = "gold_is_scam" if gold else "is_scam"
    return np.array([int(r[key]) for r in recs])


def eval_grid_point(y_pred_sets, sel, test_recs, y_test):
    """Метрики точки сетки: F1 на наборах отбора + полный разбор на тесте."""
    out = {}
    for name, recs in sel.items():
        if not recs:
            out[f"{name}_f1"] = None
            continue
        yt = y_of(recs, gold=name.startswith("dev"))
        out[f"{name}_f1"] = float(metrics_by_lang(recs, yt,
                                                  y_pred_sets[name])["overall"]["f1"])
    out["test"] = metrics_by_lang(test_recs, y_test, y_pred_sets["test"])
    return out


def run_tfidf(cfg, seed, data, sel, test_recs, y_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_recs, comp = build_train(cfg, data["train"])
    # IDF считается ТОЛЬКО по обучающей выборке конфигурации: при zero-shot
    # целевой язык не должен попасть даже в словарь признаков
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([build_text(r) for r in train_recs])
    ytr = y_of(train_recs)
    mats = {k: vec.transform([build_text(r) for r in v]) for k, v in sel.items() if v}
    Xte = vec.transform([build_text(r) for r in test_recs])

    grid = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(Xtr, ytr)
        preds = {k: clf.predict(m) for k, m in mats.items()}
        preds["test"] = clf.predict(Xte)
        grid.append({"params": {"C": C}} | eval_grid_point(preds, sel, test_recs, y_test))
    return grid, comp, int(Xtr.shape[1])


def run_layer2(cfg, seed, data, sel, test_recs, y_test, emb):
    from lightgbm import LGBMClassifier

    train_recs, comp = build_train(cfg, data["train"])
    Xtr = np.vstack([emb[r["id"]] for r in train_recs]).astype(np.float32)
    ytr = y_of(train_recs)
    mats = {k: np.vstack([emb[r["id"]] for r in v]).astype(np.float32)
            for k, v in sel.items() if v}
    Xte = np.vstack([emb[r["id"]] for r in test_recs]).astype(np.float32)

    grid = []
    for hp in LGB_GRID:
        clf = LGBMClassifier(random_state=seed, **hp, **LGB_FIXED).fit(Xtr, ytr)
        preds = {k: clf.predict(m) for k, m in mats.items()}
        preds["test"] = clf.predict(Xte)
        grid.append({"params": hp} | eval_grid_point(preds, sel, test_recs, y_test))
    return grid, comp, int(Xtr.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/crosslingual")
    ap.add_argument("--only", choices=["tfidf", "layer2"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="только первые N конфигураций")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "layers12.log")

    data = load_all()
    cfgs = enumerate_configs()
    if args.limit:
        cfgs = cfgs[:args.limit]
    test_recs = data["test"]
    y_test = y_of(test_recs, gold=True)
    emb = None if args.only == "tfidf" else load_embeddings()

    models = [m for m in ("tfidf", "layer2") if args.only in (None, m)]
    print(f"\nконфигураций {len(cfgs)} × сидов {len(SEEDS)} × моделей {len(models)}")
    print(f"сетки: TF-IDF {len(C_GRID)} точек, LightGBM {len(LGB_GRID)} точек\n")

    results = {m: {} for m in models}
    t0 = time.time()
    for ci, cfg in enumerate(cfgs, 1):
        sel = selection_sets(data["dev"], data["val"], cfg["sources"])
        n_sel = {k: len(v) for k, v in sel.items()}
        for model in models:
            per_seed = []
            for seed in SEEDS:
                if model == "tfidf":
                    grid, comp, nfeat = run_tfidf(cfg, seed, data, sel, test_recs, y_test)
                else:
                    grid, comp, nfeat = run_layer2(cfg, seed, data, sel, test_recs,
                                                   y_test, emb)
                per_seed.append({"seed": seed, "grid": grid, "n_features": nfeat})
            results[model][cfg["name"]] = {
                "config": cfg, "composition": comp, "selection_sizes": n_sel,
                "per_seed": per_seed,
            }
        el = time.time() - t0
        print(f"[{ci:>2}/{len(cfgs)}] {cfg['name']:<28} n_train={comp['n_total']:>5} "
              f"dev_source={n_sel['dev_source']:>3} val_source={n_sel['val_source']:>3} "
              f"| {el / 60:.1f} мин")

    payload = {
        "task": "кросс-язычный перенос, слои 1 и 2",
        "gold_file": "data/gold/gold_test_v2.csv",
        "split": {"seed": data["split_meta"]["seed"],
                  "n_dev": len(data["dev"]), "n_test": len(test_recs)},
        "seeds": SEEDS, "langs": LANGS,
        "grids": {"tfidf": [{"C": c} for c in C_GRID], "layer2": LGB_GRID},
        "layer2_encoder": "intfloat/multilingual-e5-base, агрегация mean, энкодер заморожен",
        "selection_note": ("dev_source — честный zero-shot (целевой язык не участвует); "
                           "val_source — контроль по silver; dev_full — верхняя граница, "
                           "НЕ zero-shot"),
        "models": results,
    }
    (out_dir / "layers12.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nготово за {(time.time() - t0) / 60:.1f} мин → {out_dir / 'layers12.json'}")


if __name__ == "__main__":
    main()
