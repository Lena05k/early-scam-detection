#!/usr/bin/env python3
"""Слой 2: замороженные многоязычные эмбеддинги + градиентный бустинг.

Мост между TF-IDF (слой 1) и файнтюном трансформера: кросс-язычные представления
БЕЗ дообучения энкодера. Каждая реплика кодируется предобученным энкодером,
векторы реплик агрегируются в один вектор диалога, поверх обучается LightGBM.
Энкодер заморожен полностью — обучается только бустинг.

Энкодеры (оба, для сравнения):
  · sentence-transformers/LaBSE
  · intfloat/multilingual-e5-base   — с обязательным префиксом "query: "

Агрегации (три, лучшая выбирается по gold-dev):
  · mean      — среднее по репликам
  · mean_max  — конкатенация [среднее, максимум по каждой размерности]
  · pos_mean  — среднее с линейными весами по позиции (поздние реплики весомее)

Протокол как везде: обучение на train 1392 (silver), отбор гиперпараметров и
агрегации по gold-dev 100 (человеческие метки), финальный замер на gold-test v2
(303). Разбиение берётся готовым из gold_dev_split.json (ассерты). Контрольно
считается и то, что выбрал бы silver_val — у TF-IDF разница была +0.0008,
у xlm-r +0.069.

Сиды 20260721/22/23 осмысленны: LightGBM с feature_fraction/bagging_fraction < 1
семплирует признаки и строки, поэтому разброс по сидам реален (в отличие от
LogReg слоя 1, где std был ровно 0).

⚠ Реплики кодируются БЕЗ имён говорящих — имя в этом корпусе почти эквивалентно
метке (92.1% имён atk встречаются только в scam либо только в benign).

  python scripts/train_layer2.py                 # полный прогон
  python scripts/train_layer2.py --encode-only   # только посчитать кэш эмбеддингов
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import read_corpus, read_gold_csv  # noqa: E402

ENCODERS = {
    "LaBSE": {"hf": "sentence-transformers/LaBSE", "prefix": ""},
    "multilingual-e5-base": {"hf": "intfloat/multilingual-e5-base", "prefix": "query: "},
}
AGGREGATIONS = ["mean", "mean_max", "pos_mean"]

# сетка LightGBM: 2 × 2 × 3 = 12 комбинаций
GRID = [{"n_estimators": n, "num_leaves": lv, "learning_rate": lr}
        for n in (200, 400) for lv in (15, 31) for lr in (0.05, 0.1, 0.2)]
LGB_FIXED = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
             "min_child_samples": 20, "verbose": -1, "n_jobs": -1}
SEEDS = [20260721, 20260722, 20260723]

# опубликованные числа других слоёв (см. CLAUDE.md) — для лесенки и графика silver
LAYER1_TFIDF = {"f1": 0.8638, "gold": "v1", "silver_agreement": 0.9868}
MDEBERTA = {"f1": 0.8942, "gold": "v1", "silver_agreement": 0.9395}
XLMR = {"f1": 0.9344, "gold": "v2", "silver_agreement": 0.8372}


class Tee:
    """Печать одновременно в stdout и в лог-файл."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
        self.f.flush()          # без этого лог пуст до конца прогона

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# ───────────────────────────── эмбеддинги ─────────────────────────────
def encode_corpus(records, enc_name, cache_dir, batch_size=64):
    """Реплики всех диалогов → эмбеддинги. Кэш на диске, чтобы не пересчитывать.

    Возвращает {dialog_id: массив (T, D) float32}. Тексты реплик берутся без имён
    говорящих; для e5 каждой реплике предпосылается "query: ".
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{enc_name.replace('/', '_')}_turns.npz"
    ids = [r["id"] for r in records]
    counts = np.array([len(r["turns"]) for r in records])

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids and np.array_equal(z["counts"], counts):
            mat, off = z["matrix"], np.concatenate([[0], np.cumsum(z["counts"])])
            print(f"[{enc_name}] кэш: {cache} — {mat.shape[0]} реплик, dim {mat.shape[1]}")
            return {i: mat[off[k]:off[k + 1]] for k, i in enumerate(ids)}
        print(f"[{enc_name}] кэш устарел (состав корпуса изменился) — пересчитываю")

    from sentence_transformers import SentenceTransformer
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    cfg = ENCODERS[enc_name]
    print(f"[{enc_name}] загрузка {cfg['hf']} на {device} …")
    model = SentenceTransformer(cfg["hf"], device=device)
    model.eval()

    texts = [cfg["prefix"] + t["text"] for r in records for t in r["turns"]]
    print(f"[{enc_name}] кодирую {len(texts)} реплик (префикс {cfg['prefix']!r})")
    t0 = time.time()
    with torch.no_grad():
        mat = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    mat = np.asarray(mat, dtype=np.float32)
    print(f"[{enc_name}] готово за {time.time() - t0:.1f} с, матрица {mat.shape}")

    np.savez_compressed(cache, matrix=mat, ids=np.array(ids), counts=counts)
    print(f"[{enc_name}] кэш сохранён: {cache}")
    off = np.concatenate([[0], np.cumsum(counts)])
    return {i: mat[off[k]:off[k + 1]] for k, i in enumerate(ids)}


def aggregate(turn_vecs, how):
    """Векторы реплик (T, D) → один вектор диалога."""
    if how == "mean":
        return turn_vecs.mean(axis=0)
    if how == "mean_max":
        return np.concatenate([turn_vecs.mean(axis=0), turn_vecs.max(axis=0)])
    if how == "pos_mean":
        # линейные веса по позиции: последняя реплика весит ровно T раз больше первой
        w = np.arange(1, len(turn_vecs) + 1, dtype=np.float32)
        return (turn_vecs * w[:, None]).sum(axis=0) / w.sum()
    raise ValueError(f"неизвестная агрегация: {how}")


def build_matrix(recs, emb, how):
    return np.vstack([aggregate(emb[r["id"]], how) for r in recs]).astype(np.float32)


# ─────────────────────────────── метрики ───────────────────────────────
def metrics_from(y_true, y_pred, prob=None):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 cohen_kappa_score, precision_recall_fscore_support,
                                 roc_auc_score)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
           "recall": float(r), "f1": float(f),
           "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}
    if prob is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    return out


def f1_of(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def mstd(values):
    v = [float(x) for x in values]
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "values": v}


# ──────────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--gold", default="data/gold/gold_test_v2.csv")
    ap.add_argument("--gold-dev", default="data/gold/gold_dev_split.json")
    ap.add_argument("--out-dir", default="runs/layer2")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "train.log")

    from sklearn.metrics import confusion_matrix

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
    assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"], \
        "разбиение gold не совпало с gold_dev_split.json"
    gd = {r.get("scenario_group") for r in dev_recs}
    gt = {r.get("scenario_group") for r in test_recs}
    assert not (gd & gt), "группы scenario_group разорваны между dev и test"

    print(f"разбиение gold: seed={sm['seed']}  dev={len(dev_recs)}  test={len(test_recs)}"
          f"  (групп {len(gd)}/{len(gt)}, пересечение 0)")
    print(f"gold-файл {args.gold} | train {len(train_recs)} (silver) / "
          f"val {len(val_recs)} (silver)")

    y_tr = np.array([int(r["is_scam"]) for r in train_recs])
    y_val = np.array([int(r["is_scam"]) for r in val_recs])
    y_dev = np.array([r["gold_is_scam"] for r in dev_recs])
    y_te = np.array([r["gold_is_scam"] for r in test_recs])
    silver = {r["id"]: int(r["is_scam"]) for r in records}
    y_silver_te = np.array([silver[r["id"]] for r in test_recs])
    langs = sorted({r["lang"] for r in test_recs})

    # ── эмбеддинги (кэшируются) ──
    embeddings = {name: encode_corpus(records, name, out_dir / "cache")
                  for name in ENCODERS}
    if args.encode_only:
        print("--encode-only: кэш посчитан, выходим")
        return

    # LightGBM импортируется ТОЛЬКО ПОСЛЕ кодирования: и он, и torch тянут свой
    # libomp, и на macOS два OpenMP-рантайма в одном процессе дают SIGSEGV
    # (проверено: импорт lightgbm перед torch роняет процесс на загрузке LaBSE).
    # Когда кэш эмбеддингов готов, torch вообще не импортируется.
    from lightgbm import LGBMClassifier

    # ── перебор: энкодер × агрегация × сид × сетка ──
    configs, t_start = {}, time.time()
    for enc_name in ENCODERS:
        emb = embeddings[enc_name]
        for how in AGGREGATIONS:
            Xtr = build_matrix(train_recs, emb, how)
            Xval = build_matrix(val_recs, emb, how)
            Xdev = build_matrix(dev_recs, emb, how)
            Xte = build_matrix(test_recs, emb, how)
            key = f"{enc_name}|{how}"
            per_seed = []
            for seed in SEEDS:
                grid_rows = []
                for hp in GRID:
                    clf = LGBMClassifier(random_state=seed, **hp, **LGB_FIXED)
                    clf.fit(Xtr, y_tr)
                    grid_rows.append({
                        **hp,
                        "gold_dev_f1": f1_of(y_dev, clf.predict(Xdev)),
                        "silver_val_f1": f1_of(y_val, clf.predict(Xval)),
                    })

                def evaluate(hp):
                    clf = LGBMClassifier(random_state=seed, **hp, **LGB_FIXED)
                    clf.fit(Xtr, y_tr)
                    yp = clf.predict(Xte).astype(int)
                    pr = clf.predict_proba(Xte)[:, 1]
                    per_lang = {}
                    for lg in langs:
                        idx = [i for i, r in enumerate(test_recs) if r["lang"] == lg]
                        per_lang[lg] = metrics_from(y_te[idx], yp[idx]) | {"n": len(idx)}
                    return {"metrics": metrics_from(y_te, yp, pr),
                            "confusion_matrix": confusion_matrix(
                                y_te, yp, labels=[0, 1]).tolist(),
                            "per_language": per_lang,
                            "silver_agreement": float((yp == y_silver_te).mean())}

                # ключ сортировки фиксирован → выбор детерминирован при равных F1
                hp_keys = ("n_estimators", "num_leaves", "learning_rate")
                best_dev = max(grid_rows, key=lambda g: (g["gold_dev_f1"],
                                                         -g["n_estimators"], -g["num_leaves"]))
                best_sv = max(grid_rows, key=lambda g: (g["silver_val_f1"],
                                                        -g["n_estimators"], -g["num_leaves"]))
                per_seed.append({
                    "seed": seed, "grid": grid_rows,
                    "selected_by_gold_dev": {
                        "params": {k: best_dev[k] for k in hp_keys},
                        "gold_dev_f1": best_dev["gold_dev_f1"],
                        **evaluate({k: best_dev[k] for k in hp_keys})},
                    "selected_by_silver_val": {
                        "params": {k: best_sv[k] for k in hp_keys},
                        "silver_val_f1": best_sv["silver_val_f1"],
                        **evaluate({k: best_sv[k] for k in hp_keys})},
                })
            configs[key] = {"encoder": enc_name, "aggregation": how,
                            "dim": int(Xtr.shape[1]), "per_seed": per_seed}
            dev_f1 = mstd([s["selected_by_gold_dev"]["gold_dev_f1"] for s in per_seed])
            test_f1 = mstd([s["selected_by_gold_dev"]["metrics"]["f1"] for s in per_seed])
            configs[key]["gold_dev_f1"] = dev_f1
            configs[key]["gold_test_f1"] = test_f1
            print(f"  {key:<38} dim {Xtr.shape[1]:>4} | gold-dev F1 "
                  f"{dev_f1['mean']:.4f}±{dev_f1['std']:.4f} | gold-test F1 "
                  f"{test_f1['mean']:.4f}±{test_f1['std']:.4f}")
    print(f"\nперебор занял {time.time() - t_start:.0f} с "
          f"({len(ENCODERS)}×{len(AGGREGATIONS)}×{len(SEEDS)}×{len(GRID)} обучений LightGBM)")

    # ── ВЫБОР ЛУЧШЕЙ КОНФИГУРАЦИИ — ТОЛЬКО ПО GOLD-DEV ──
    best_key = max(configs, key=lambda k: configs[k]["gold_dev_f1"]["mean"])
    best = configs[best_key]
    print(f"\nлучшая конфигурация по gold-dev: {best_key} "
          f"(dev F1 {best['gold_dev_f1']['mean']:.4f})")

    print("\nсводка по конфигурациям (отбор по gold-dev):")
    hdr = (f"{'энкодер':<22} {'агрегация':<10} {'dev F1':>16} {'test F1':>16} "
           f"{'test κ':>16}")
    print(hdr)
    print("-" * len(hdr))
    for k, c in configs.items():
        kap = mstd([s["selected_by_gold_dev"]["metrics"]["cohen_kappa"]
                    for s in c["per_seed"]])
        star = " ←" if k == best_key else ""
        print(f"{c['encoder']:<22} {c['aggregation']:<10} "
              f"{c['gold_dev_f1']['mean']:>10.4f}±{c['gold_dev_f1']['std']:.4f} "
              f"{c['gold_test_f1']['mean']:>10.4f}±{c['gold_test_f1']['std']:.4f} "
              f"{kap['mean']:>10.4f}±{kap['std']:.4f}{star}")

    # ── итоговые метрики лучшей конфигурации ──
    KEYS = ["accuracy", "precision", "recall", "f1", "cohen_kappa", "roc_auc",
            "average_precision"]
    summary = {k: mstd([s["selected_by_gold_dev"]["metrics"][k] for s in best["per_seed"]])
               for k in KEYS}
    summary["silver_agreement"] = mstd([s["selected_by_gold_dev"]["silver_agreement"]
                                        for s in best["per_seed"]])
    print(f"\ngold-test v2 (n={len(test_recs)}), {best_key}, среднее ± std по 3 сидам:")
    for k in KEYS + ["silver_agreement"]:
        print(f"{k:>20} {summary[k]['mean']:>10.4f} {summary[k]['std']:>9.4f}")

    cm_mean = np.mean([np.array(s["selected_by_gold_dev"]["confusion_matrix"])
                       for s in best["per_seed"]], axis=0)
    print("\nусреднённая матрица ошибок (строки gold, столбцы предсказание):")
    print(f"  gold=0  {cm_mean[0, 0]:8.1f}  {cm_mean[0, 1]:8.1f}")
    print(f"  gold=1  {cm_mean[1, 0]:8.1f}  {cm_mean[1, 1]:8.1f}")

    per_lang = {lg: {m: mstd([s["selected_by_gold_dev"]["per_language"][lg][m]
                              for s in best["per_seed"]])
                     for m in ("f1", "precision", "recall")}
                for lg in langs}
    print("\nпо языкам (лучшая конфигурация):")
    for lg in langs:
        n = best["per_seed"][0]["selected_by_gold_dev"]["per_language"][lg]["n"]
        print(f"  {lg}: " + "  ".join(
            f"{m} {per_lang[lg][m]['mean']:.4f}±{per_lang[lg][m]['std']:.4f}"
            for m in ("f1", "precision", "recall")) + f"   (n={n})")

    # выбранные гиперпараметры
    print("\nвыбранные гиперпараметры по сидам (лучшая конфигурация):")
    for s in best["per_seed"]:
        print(f"  сид {s['seed']}: {s['selected_by_gold_dev']['params']} → "
              f"gold-test F1 {s['selected_by_gold_dev']['metrics']['f1']:.4f}")

    # ── контроль: отбор по silver_val ──
    print("\nконтроль — отбор гиперпараметров по silver_val вместо gold-dev:")
    sel_cmp = {}
    for k, c in configs.items():
        g = mstd([s["selected_by_gold_dev"]["metrics"]["f1"] for s in c["per_seed"]])
        v = mstd([s["selected_by_silver_val"]["metrics"]["f1"] for s in c["per_seed"]])
        sel_cmp[k] = {"gold_dev": g, "silver_val": v, "delta": g["mean"] - v["mean"]}
        print(f"  {k:<38} gold-dev {g['mean']:.4f} | silver_val {v['mean']:.4f} "
              f"| разница {g['mean'] - v['mean']:+.4f}")
    d_best = sel_cmp[best_key]["delta"]
    print(f"\nна лучшей конфигурации разница {d_best:+.4f} "
          f"(TF-IDF +0.0008, xlm-r +0.069)")

    # ── лесенка ──
    l0 = json.loads(pathlib.Path("runs/layer0/results.json").read_text(encoding="utf-8"))
    ladder = [{"model": p["name_ru"], "f1": p["metrics"]["f1"], "layer": "слой 0",
               "gold": "v2"} for p in (l0["predictors"][k] for k in
                                       ("const_scam", "const_benign", "length", "url",
                                        "keywords", "logreg3"))]
    ladder += [
        {"model": "TF-IDF + LogReg", "f1": LAYER1_TFIDF["f1"], "layer": "слой 1",
         "gold": LAYER1_TFIDF["gold"]},
        {"model": f"слой 2: {best['encoder']} / {best['aggregation']}",
         "f1": summary["f1"]["mean"], "std": summary["f1"]["std"],
         "layer": "слой 2", "gold": "v2"},
        {"model": "mDeBERTa-v3-base", "f1": MDEBERTA["f1"], "layer": "файнтюн",
         "gold": MDEBERTA["gold"]},
        {"model": "xlm-roberta-base", "f1": XLMR["f1"], "layer": "файнтюн",
         "gold": XLMR["gold"]},
    ]
    ladder.sort(key=lambda d: d["f1"])
    print("\nлесенка (F1 на gold-test, 303 диалога):")
    for d in ladder:
        print(f"  {d['model']:<40} {d['f1']:.4f}  {'█' * int(round(d['f1'] * 40)):<41}"
              f" {d['layer']} · gold {d['gold']}")

    silver_by_model = {
        "слой 0: ключевые слова": l0["diagnostics"]["silver_agreement_on_test"]["keywords"],
        "слой 0: логрегрессия-3": l0["diagnostics"]["silver_agreement_on_test"]["logreg3"],
        "слой 1: TF-IDF": LAYER1_TFIDF["silver_agreement"],
        f"слой 2: {best['encoder']}": summary["silver_agreement"]["mean"],
        "файнтюн: mDeBERTa": MDEBERTA["silver_agreement"],
        "файнтюн: xlm-r": XLMR["silver_agreement"],
    }
    print("\nсовпадение с silver-метками на тесте:")
    for k, v in silver_by_model.items():
        print(f"  {k:<32} {v:.4f}")
    print(f"  {'(сам silver против gold)':<32} "
          f"{l0['diagnostics']['silver_vs_gold_on_test']:.4f}")

    payload = {
        "task": "layer 2 — замороженные многоязычные эмбеддинги + LightGBM",
        "gold_file": args.gold,
        "split": {"file": args.gold_dev, "seed": sm["seed"],
                  "n_dev": len(dev_recs), "n_test": len(test_recs)},
        "encoders": {k: v["hf"] for k, v in ENCODERS.items()},
        "e5_prefix": ENCODERS["multilingual-e5-base"]["prefix"],
        "encoder_frozen": True,
        "aggregations": AGGREGATIONS,
        "lgb_grid": GRID, "lgb_fixed": LGB_FIXED, "seeds": SEEDS,
        "feature_text": "тексты реплик без имён говорящих, L2-нормированные эмбеддинги",
        "best_config": best_key,
        "summary": summary, "per_language": per_lang,
        "confusion_matrix_mean": cm_mean.tolist(),
        "selection_comparison": sel_cmp,
        "configs": configs,
        "ladder": ladder,
        "silver_agreement_by_model": silver_by_model,
        "silver_vs_gold": l0["diagnostics"]["silver_vs_gold_on_test"],
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out_dir / 'results.json'}, лог {out_dir / 'train.log'}")

    if not args.no_plots:
        from make_layer2_plots import make_plots
        make_plots(payload, out_dir / "plots")


if __name__ == "__main__":
    main()
