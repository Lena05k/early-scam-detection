#!/usr/bin/env python3
"""Тесты значимости различий между моделями на gold-test v2 (план §7).

«Разница без значимости не заявляется». Считается ПАРНЫЙ бутстрап (одна выборка
индексов диалогов применяется к обеим моделям сразу — это сохраняет корреляцию
между ними и не завышает дисперсию разницы) и перестановочный тест.

⚠ ДВА ИСТОЧНИКА НЕОПРЕДЕЛЁННОСТИ, и их нельзя путать:
  1. конечность тестовой выборки (303 диалога) — её меряет бутстрап по диалогам;
  2. разброс между сидами обучения (у xlm-r std F1 = 0.0086) — бутстрап по диалогам
     о нём ничего не знает.
Разница xlm-r с ключевыми словами (0.0148) сопоставима со вторым источником,
поэтому считаются ОБА варианта: `test_only` (предсказания фиксированы голосованием
по сидам) и `test_and_seed` (в каждой реплике для каждой модели заново тянется сид).
Основным считается второй — он честнее отвечает на вопрос «повторится ли разница,
если переобучить модель».

Предсказания берутся из готовых снимков (xlm-r, mDeBERTa) либо пересчитываются
детерминированно (TF-IDF, слой 2, ключевые слова). Ничего не обучается заново
из того, что требует GPU.

  python scripts/significance.py
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import load_all  # noqa: E402
from train_binary import build_text  # noqa: E402

N_BOOT = 10000
N_PERM = 10000
BOOT_SEED = 20260810
SEEDS = [20260721, 20260722, 20260723]

PAIRS = [
    ("xlmr", "keywords"),        # 0.9344 против 0.9196 — самая узкая и самая важная
    ("xlmr", "mdeberta"),
    ("xlmr", "layer2"),
    ("xlmr", "tfidf"),
    ("keywords", "tfidf"),
]
RU = {"xlmr": "xlm-roberta-base", "keywords": "слой 0: ключевые слова",
      "mdeberta": "mDeBERTa-v3-base", "layer2": "слой 2: e5 + LightGBM",
      "tfidf": "слой 1: TF-IDF"}


def f1_np(y, p):
    """F1 по классу scam на numpy — быстрее sklearn в цикле на 10000 реплик."""
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


# ─────────────────── сбор предсказаний по каждой модели ───────────────────
def preds_from_snapshots(pattern, ids, pick_epoch):
    """Снимки по эпохам → {сид: массив предсказаний}, эпоха выбирается по gold-dev."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        seed = d["seed"]
        ep = pick_epoch(seed, d)
        i = d["epochs"].index(ep)
        order = {did: k for k, did in enumerate(d["dialog_ids"])}
        yp = np.array(d["per_epoch"][i]["y_pred"])
        out[seed] = np.array([yp[order[x]] for x in ids])
    return out


def xlmr_preds(ids):
    """xlm-r: выбранная эпоха 1 на всех трёх сидах (см. «Цена выбора эпохи»)."""
    return preds_from_snapshots("runs/epoch_cost/data/xlmr/gold_test_per_epoch_seed*.json",
                                ids, lambda seed, d: 1)


def mdeberta_preds(ids):
    """mDeBERTa: эпоха выбирается по gold-dev того же прогона (F1)."""
    def pick(seed, d):
        dv = json.loads(pathlib.Path(
            f"runs/epoch_cost/data/mdeberta/gold_dev_per_epoch_seed{seed}.json"
        ).read_text(encoding="utf-8"))
        f1s = [f1_np(np.array(pe["y_true"]), np.array(pe["y_pred"]))
               for pe in dv["per_epoch"]]
        return dv["epochs"][int(np.argmax(f1s))]
    return preds_from_snapshots(
        "runs/epoch_cost/data/mdeberta/gold_test_per_epoch_seed*.json", ids, pick)


def tfidf_preds(data, test_recs, dev_recs):
    """TF-IDF: C отбирается по gold-dev, обучение на train (silver). Детерминирован."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([build_text(r) for r in data["train"]])
    ytr = np.array([int(r["is_scam"]) for r in data["train"]])
    Xdev = vec.transform([build_text(r) for r in dev_recs])
    ydev = np.array([r["gold_is_scam"] for r in dev_recs])
    Xte = vec.transform([build_text(r) for r in test_recs])
    out = {}
    for seed in SEEDS:
        best, best_f1 = None, -1.0
        for C in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(Xtr, ytr)
            f = f1_np(ydev, clf.predict(Xdev))
            if f > best_f1:
                best, best_f1 = clf, f
        out[seed] = best.predict(Xte)
    return out


def layer2_preds(data, test_recs, dev_recs):
    """Слой 2: e5 (mean) + LightGBM, гиперпараметры по gold-dev. Кэш эмбеддингов готов."""
    from lightgbm import LGBMClassifier
    z = np.load("runs/layer2/cache/multilingual-e5-base_turns.npz", allow_pickle=False)
    mat, ids_, counts = z["matrix"], list(z["ids"]), z["counts"]
    off = np.concatenate([[0], np.cumsum(counts)])
    emb = {i: mat[off[k]:off[k + 1]].mean(axis=0) for k, i in enumerate(ids_)}
    X = lambda recs: np.vstack([emb[r["id"]] for r in recs]).astype(np.float32)  # noqa: E731
    Xtr, ytr = X(data["train"]), np.array([int(r["is_scam"]) for r in data["train"]])
    Xdev, ydev = X(dev_recs), np.array([r["gold_is_scam"] for r in dev_recs])
    Xte = X(test_recs)
    fixed = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
             "min_child_samples": 20, "verbose": -1, "n_jobs": -1}
    grid = [{"n_estimators": n, "num_leaves": lv, "learning_rate": lr}
            for n in (200, 400) for lv in (15, 31) for lr in (0.05, 0.1, 0.2)]
    out = {}
    for seed in SEEDS:
        best, best_f1 = None, -1.0
        for hp in grid:
            clf = LGBMClassifier(random_state=seed, **hp, **fixed).fit(Xtr, ytr)
            f = f1_np(ydev, clf.predict(Xdev))
            if f > best_f1:
                best, best_f1 = clf, f
        out[seed] = best.predict(Xte)
    return out


def keyword_preds(data, test_recs, dev_recs):
    """Слой 0: 20 слов на язык из train, порог N по gold-dev. Полностью детерминирован."""
    from train_layer0 import compile_keywords, kw_hits, mine_keywords, plain_text
    kw = mine_keywords(data["train"])
    comp = compile_keywords(kw)
    hits = lambda recs: np.array([kw_hits(plain_text(r), r["lang"], comp)  # noqa: E731
                                  for r in recs])
    hdev, hte = hits(dev_recs), hits(test_recs)
    ydev = np.array([r["gold_is_scam"] for r in dev_recs])
    best_n = max(range(1, 11), key=lambda n: (f1_np(ydev, (hdev >= n).astype(int)), -n))
    pred = (hte >= best_n).astype(int)
    return {s: pred for s in SEEDS}, best_n     # один и тот же на всех сидах


# ──────────────────────────── статистика ────────────────────────────
def paired_bootstrap(y, pa, pb, rng, n_boot, mix_seeds):
    """Парный бутстрап разницы F1(A) − F1(B).

    Одна выборка индексов применяется к ОБЕИМ моделям. mix_seeds=True — в каждой
    реплике для каждой модели независимо тянется сид обучения, то есть в разброс
    входит и неопределённость от переобучения, а не только от размера теста.
    """
    n = len(y)
    ka = [k for k in pa if isinstance(k, int)]
    kb = [k for k in pb if isinstance(k, int)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        a = pa[ka[rng.integers(len(ka))]] if mix_seeds else pa["_vote"]
        bb = pb[kb[rng.integers(len(kb))]] if mix_seeds else pb["_vote"]
        diffs[b] = f1_np(y[idx], a[idx]) - f1_np(y[idx], bb[idx])
    return diffs


def permutation_test(y, a, b, rng, n_perm):
    """Перестановочный тест: на каждом диалоге предсказания моделей меняются
    местами с вероятностью 0.5. Даёт p-значение для |ΔF1| под нулевой гипотезой
    «модели обменяемы»."""
    obs = abs(f1_np(y, a) - f1_np(y, b))
    cnt = 0
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        a2 = np.where(swap, b, a)
        b2 = np.where(swap, a, b)
        if abs(f1_np(y, a2) - f1_np(y, b2)) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/significance")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all()
    test_recs, dev_recs = data["test"], data["dev"]
    ids = [r["id"] for r in test_recs]
    y = np.array([r["gold_is_scam"] for r in test_recs])
    langs = np.array([r["lang"] for r in test_recs])

    print("\nсобираю пер-диалоговые предсказания на gold-test v2 (303)…")
    preds = {}
    preds["xlmr"] = xlmr_preds(ids)
    preds["mdeberta"] = mdeberta_preds(ids)
    preds["tfidf"] = tfidf_preds(data, test_recs, dev_recs)
    preds["layer2"] = layer2_preds(data, test_recs, dev_recs)
    kwp, best_n = keyword_preds(data, test_recs, dev_recs)
    preds["keywords"] = kwp
    print(f"  порог ключевых слов, выбранный по gold-dev: N={best_n}")

    # голосование по сидам — фиксированное предсказание модели
    summary = {}
    for m, d in preds.items():
        seed_keys = sorted(k for k in d if isinstance(k, int))
        v = (np.mean([d[s] for s in seed_keys], axis=0) >= 0.5).astype(int)
        per_seed = [f1_np(y, d[s]) for s in seed_keys]
        d["_vote"] = v          # добавляем ПОСЛЕ подсчёта, иначе ключи разнотипны
        summary[m] = {"f1_vote": f1_np(y, v),
                      "f1_per_seed": per_seed,
                      "f1_mean": float(np.mean(per_seed)),
                      "f1_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                      "per_language": {lg: f1_np(y[langs == lg], v[langs == lg])
                                       for lg in ("en", "ru", "zh")}}
        print(f"  {RU[m]:<26} F1 голосование {summary[m]['f1_vote']:.4f} | "
              f"по сидам {summary[m]['f1_mean']:.4f} ± {summary[m]['f1_std']:.4f}")

    rng = np.random.default_rng(BOOT_SEED)
    results = []
    print(f"\nпарный бутстрап ({args.n_boot} реплик) и перестановочный тест "
          f"({args.n_perm}):")
    for a, b in PAIRS:
        row = {"a": a, "b": b, "a_ru": RU[a], "b_ru": RU[b],
               "f1_a": summary[a]["f1_vote"], "f1_b": summary[b]["f1_vote"],
               "observed_diff": summary[a]["f1_vote"] - summary[b]["f1_vote"]}
        for tag, mix in (("test_only", False), ("test_and_seed", True)):
            d = paired_bootstrap(y, preds[a], preds[b], rng, args.n_boot, mix)
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[tag] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                        "p_a_better": float((d > 0).mean()),
                        "crosses_zero": bool(lo <= 0 <= hi)}
        obs, p = permutation_test(y, preds[a]["_vote"], preds[b]["_vote"], rng, args.n_perm)
        row["permutation"] = {"observed_abs_diff": float(obs), "p_value": float(p)}
        results.append(row)
        t, ts = row["test_only"], row["test_and_seed"]
        print(f"\n  {RU[a]} против {RU[b]}")
        print(f"    ΔF1 = {row['observed_diff']:+.4f}")
        print(f"    только тест:  ДИ [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  "
              f"P(A лучше) {t['p_a_better']:.3f}  "
              f"{'НЕ значимо' if t['crosses_zero'] else 'значимо'}")
        print(f"    тест + сиды:  ДИ [{ts['ci95'][0]:+.4f}, {ts['ci95'][1]:+.4f}]  "
              f"P(A лучше) {ts['p_a_better']:.3f}  "
              f"{'НЕ значимо' if ts['crosses_zero'] else 'значимо'}")
        print(f"    перестановочный тест: p = {p:.4f}")

    payload = {
        "gold": "data/gold/gold_test_v2.csv", "n_test": len(y),
        "n_boot": args.n_boot, "n_perm": args.n_perm, "boot_seed": BOOT_SEED,
        "seeds": SEEDS, "keyword_threshold": best_n,
        "note": ("test_only — бутстрап только по диалогам при фиксированных "
                 "предсказаниях (голосование по сидам); test_and_seed — в каждой "
                 "реплике дополнительно тянется сид обучения, то есть учтён и "
                 "разброс от переобучения. Ключевые слова и TF-IDF детерминированы, "
                 "у них сид ничего не меняет."),
        "models": summary, "pairs": results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "predictions.npz", y=y, ids=np.array(ids), langs=langs,
                        **{f"{m}_vote": preds[m]["_vote"] for m in preds})
    print(f"\nсохранено: {out_dir / 'results.json'}, {out_dir / 'predictions.npz'}")


if __name__ == "__main__":
    main()
