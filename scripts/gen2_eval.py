#!/usr/bin/env python3
"""Перенос на второй генератор: те же модели, чужой стиль (gen2).

Вопрос. Корпус v1 сделан одним вендором, и все модели без файнтюна воспроизводят его
silver почти точно: TF-IDF 0.9868, слой 2 0.9890, сам DeepSeek 0.9901. Отсюда подозрение,
что они ловят СТИЛЬ генератора, а не обман. gen2 — та же таксономия и те же промпты,
но другой вендор. Если качество на gen2 обвалится, подозрение подтвердится.

Что меряется (всё обучается на train v1, silver; gen2 в обучении не участвует):
  1. ключевые слова слоя 0 — 20 слов на язык, добытые из train v1, порог N=4 с gold-dev.
     ГЛАВНАЯ проверка: 0.9196 на v1 держались на стиле DeepSeek или нет.
  2. TF-IDF + LogReg (слой 1) — char_wb 2-5, min_df 2, C с gold-dev.
  3. эмбеддинги e5 + LightGBM (слой 2) — лучшая по gold-dev конфигурация (mean).
  4. xlm-roberta-base (файнтюн) — сохранённые веса models/binary_xlmr_seed*.
  5. согласие каждой модели с silver-метками gen2 против согласия с silver v1.

Опорная точка — v1 test-сплит (403), НЕ участвовавший в обучении. Метрики считаются
против двух целей: silver (сопоставимо с gen2, у которого только silver) и gold
(человеческие метки, есть только у v1).

Без данных gen2 не падает: сообщает, чего не хватает. `--check` проверяет готовность
(веса, артефакты слоя 0, пакеты) и ничего не считает.

⚠ macOS: lightgbm и torch тянут каждый свой libomp, два OpenMP в одном процессе дают
SIGSEGV. Поэтому lightgbm импортируется ПОСЛЕ кодирования эмбеддингов — как в
scripts/train_layer2.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import build_text, read_corpus, read_gold_csv  # noqa: E402
from train_layer0 import compile_keywords, kw_hits, plain_text  # noqa: E402
from train_layer2 import LGB_FIXED, aggregate, encode_corpus  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_V1 = ROOT / "data" / "corpus" / "corpus_v1.jsonl"
GEN2 = ROOT / "data" / "corpus" / "gen2_v1.jsonl"
GOLD = ROOT / "data" / "gold" / "gold_test_v2.csv"
DEV_SPLIT = ROOT / "data" / "gold" / "gold_dev_split.json"
LAYER0 = ROOT / "runs" / "layer0" / "results.json"
TFIDF_RUN = ROOT / "runs" / "tfidf_gold_dev" / "results.json"
MODEL_DIRS = sorted((ROOT / "models").glob("binary_xlmr_seed*"))
OUT_DIR = ROOT / "runs" / "gen2"

ENCODER = "multilingual-e5-base"      # лучшая по gold-dev конфигурация слоя 2
AGG = "mean"
C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS = [20260721, 20260722, 20260723]


def rel(p):
    """Путь относительно корня репозитория, если он внутри; иначе как есть."""
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def metrics_from(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"n": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}


def agreement(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b))) if len(a) else None


def mstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


# ─────────────────────────── предсказатели ───────────────────────────

def predict_keywords(recs, tables, threshold):
    """Слой 0: сработало >= threshold ключевых слов языка -> scam. Ничего не обучает."""
    compiled = compile_keywords({lang: {"keywords": t["keywords"]}
                                 for lang, t in tables.items()})
    hits = [kw_hits(plain_text(r), r["lang"], compiled) for r in recs]
    return (np.array(hits) >= threshold).astype(int), np.array(hits)


def fit_tfidf(train_recs, dev_recs, dev_y):
    """Слой 1: char_wb 2-5 на train v1 (silver); C отбирается по gold-dev."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([build_text(r) for r in train_recs])   # IDF только по train
    ytr = np.array([int(r["is_scam"]) for r in train_recs])
    Xdev = vec.transform([build_text(r) for r in dev_recs])
    best, best_f1 = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(max_iter=5000, C=C).fit(Xtr, ytr)
        f1 = f1_score(dev_y, clf.predict(Xdev), zero_division=0)
        if f1 > best_f1:
            best, best_f1 = (clf, C), f1
    clf, C = best
    return (lambda recs: clf.predict(vec.transform([build_text(r) for r in recs])),
            {"C": C, "dev_f1": float(best_f1), "n_features": int(len(vec.vocabulary_))})


def fit_layer2(train_recs, dev_recs, dev_y, emb, seed):
    """Слой 2: e5 (заморожен) + LightGBM на train v1; гиперпараметры по gold-dev."""
    import lightgbm as lgb                       # импорт ПОСЛЕ кодирования (см. модуль-док)
    from sklearn.metrics import f1_score

    def mat(recs):
        return np.vstack([aggregate(emb[r["id"]], AGG) for r in recs]).astype(np.float32)

    Xtr, ytr = mat(train_recs), np.array([int(r["is_scam"]) for r in train_recs])
    Xdev = mat(dev_recs)
    grid = [{"n_estimators": n, "num_leaves": lv, "learning_rate": lr}
            for n in (200, 400) for lv in (15, 31) for lr in (0.05, 0.1, 0.2)]
    best, best_f1 = None, -1.0
    for hp in grid:
        clf = lgb.LGBMClassifier(random_state=seed, **hp, **LGB_FIXED).fit(Xtr, ytr)
        f1 = f1_score(dev_y, clf.predict(Xdev), zero_division=0)
        if f1 > best_f1:
            best, best_f1 = (clf, hp), f1
    clf, hp = best
    return (lambda recs: clf.predict(mat(recs))), {"hp": hp, "dev_f1": float(best_f1)}


def predict_xlmr(recs, model_dir, batch_size=16, max_length=512):
    """Файнтюн: инференс сохранённых весов. Текст — как при обучении (с именами)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), dtype=torch.float32).to(device).eval()
    texts = [build_text(r) for r in recs]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    del model
    return np.concatenate(out)


# ─────────────────────────── готовность ───────────────────────────

def check_ready(gen2_path):
    """Что есть и чего нет для прогона. Возвращает (ok, отчёт)."""
    rep = {}
    rep["corpus_v1"] = CORPUS_V1.exists()
    rep["gold_test_v2"] = GOLD.exists()
    rep["gold_dev_split"] = DEV_SPLIT.exists()
    rep["layer0_keywords"] = LAYER0.exists()
    rep["xlmr_weights"] = [d.name for d in MODEL_DIRS]
    rep["gen2_data"] = pathlib.Path(gen2_path).exists()
    pkgs = {}
    for mod in ("sklearn", "lightgbm", "torch", "transformers", "sentence_transformers"):
        try:
            __import__(mod)
            pkgs[mod] = True
        except Exception as e:  # noqa: BLE001 — важен факт отсутствия, не тип
            pkgs[mod] = f"НЕТ: {type(e).__name__}"
    rep["packages"] = pkgs
    blockers = []
    for key in ("corpus_v1", "gold_test_v2", "gold_dev_split", "layer0_keywords"):
        if not rep[key]:
            blockers.append(f"нет файла: {key}")
    if not rep["xlmr_weights"]:
        blockers.append("нет весов models/binary_xlmr_seed*")
    for m, v in pkgs.items():
        if v is not True:
            blockers.append(f"пакет {m}: {v}")
    if not rep["gen2_data"]:
        blockers.append(f"НЕТ ДАННЫХ gen2: {gen2_path} "
                        f"(сгенерировать: scripts/generate_gen2.py, нужен ANTHROPIC_API_KEY)")
    rep["blockers"] = blockers
    return not blockers, rep


def print_check(rep):
    print("=== готовность к прогону gen2 ===")
    for k in ("corpus_v1", "gold_test_v2", "gold_dev_split", "layer0_keywords", "gen2_data"):
        print(f"  {'OK  ' if rep[k] else 'НЕТ '} {k}")
    print(f"  {'OK  ' if rep['xlmr_weights'] else 'НЕТ '} веса xlm-r: "
          f"{rep['xlmr_weights'] or '—'}")
    for m, v in rep["packages"].items():
        print(f"  {'OK  ' if v is True else 'НЕТ '} пакет {m}")
    if rep["blockers"]:
        print("\nблокирует:")
        for b in rep["blockers"]:
            print(f"  · {b}")
    else:
        print("\nвсё на месте — можно запускать без флагов.")


# ─────────────────────────── прогон ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Перенос моделей v1 на второй генератор")
    ap.add_argument("--corpus", default=str(CORPUS_V1))
    ap.add_argument("--gen2", default=str(GEN2))
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", default=str(OUT_DIR / "eval.json"))
    ap.add_argument("--check", action="store_true", help="только проверить готовность")
    ap.add_argument("--skip-xlmr", action="store_true", help="без файнтюна (быстро)")
    a = ap.parse_args()

    ok, rep = check_ready(a.gen2)
    if a.check:
        print_check(rep)
        return 0
    if not ok:
        print_check(rep)
        print("\nПрогон не запускался — сначала снять блокеры выше.")
        return 0                      # это не ошибка, а незавершённая подготовка

    t0 = time.time()
    records = read_corpus(a.corpus)
    gen2 = read_corpus(a.gen2)
    gold = read_gold_csv(a.gold)
    assign = json.loads(pathlib.Path(DEV_SPLIT).read_text(encoding="utf-8"))["assignment"]

    train_recs = [r for r in records if r.get("split") == "train"]
    v1_test = [r for r in records if r.get("split") == "test" and r["id"] in gold]
    dev_recs = [r for r in v1_test if assign.get(r["id"]) == "dev"]
    assert len(dev_recs) == 100 and len(v1_test) == len(gold), \
        f"разбиение gold не совпало: dev={len(dev_recs)}, test={len(v1_test)}"
    dev_y = np.array([gold[r["id"]] for r in dev_recs])

    # цели: у gen2 есть ТОЛЬКО silver, поэтому основное сравнение — против silver
    y_v1_silver = np.array([int(r["is_scam"]) for r in v1_test])
    y_v1_gold = np.array([gold[r["id"]] for r in v1_test])
    y_gen2_silver = np.array([int(r["is_scam"]) for r in gen2])

    print(f"train v1: {len(train_recs)} | v1 test: {len(v1_test)} | gen2: {len(gen2)}")
    print(f"доля scam: v1 silver {y_v1_silver.mean():.3f} | v1 gold {y_v1_gold.mean():.3f} "
          f"| gen2 silver {y_gen2_silver.mean():.3f}")
    print(f"генераторы gen2: {dict(Counter(r.get('generator') for r in gen2))}")

    results = {"task": "gen2_transfer", "n_train_v1": len(train_recs),
               "n_v1_test": len(v1_test), "n_gen2": len(gen2),
               "gen2_generators": dict(Counter(r.get("generator") for r in gen2)),
               "models": {}}

    def record(name, pred_v1, pred_gen2, extra=None):
        results["models"][name] = {
            "v1_test_vs_silver": metrics_from(y_v1_silver, pred_v1),
            "v1_test_vs_gold": metrics_from(y_v1_gold, pred_v1),
            "gen2_vs_silver": metrics_from(y_gen2_silver, pred_gen2),
            "silver_agreement_v1": agreement(pred_v1, y_v1_silver),
            "silver_agreement_gen2": agreement(pred_gen2, y_gen2_silver),
            "f1_drop_silver": (metrics_from(y_v1_silver, pred_v1)["f1"]
                               - metrics_from(y_gen2_silver, pred_gen2)["f1"]),
            **(extra or {}),
        }
        m = results["models"][name]
        print(f"  {name:<12} v1(silver) F1 {m['v1_test_vs_silver']['f1']:.4f} | "
              f"gen2 F1 {m['gen2_vs_silver']['f1']:.4f} | "
              f"падение {m['f1_drop_silver']:+.4f} | "
              f"согласие с silver {m['silver_agreement_v1']:.4f} -> "
              f"{m['silver_agreement_gen2']:.4f}")

    # ── 1. ключевые слова слоя 0 (главная проверка) ──
    print("\n[1] ключевые слова слоя 0 (списки и порог из v1, ничего не переобучается)")
    l0 = json.loads(LAYER0.read_text(encoding="utf-8"))
    tables = l0["keyword_mining"]["tables"]
    thr = int(l0["predictors"]["keywords"]["tuning"]["selected_n"])
    print(f"    порог N={thr}, слов на язык: "
          f"{ {k: len(v['keywords']) for k, v in tables.items()} }")
    kw_v1, hits_v1 = predict_keywords(v1_test, tables, thr)
    kw_g2, hits_g2 = predict_keywords(gen2, tables, thr)
    record("keywords", kw_v1, kw_g2, extra={
        "threshold": thr,
        "mean_hits_v1": {"scam": float(hits_v1[y_v1_silver == 1].mean()),
                         "benign": float(hits_v1[y_v1_silver == 0].mean())},
        "mean_hits_gen2": {"scam": float(hits_g2[y_gen2_silver == 1].mean()),
                           "benign": float(hits_g2[y_gen2_silver == 0].mean())},
        "hits_by_lang_gen2": {
            lang: float(np.mean([h for h, r in zip(hits_g2, gen2) if r["lang"] == lang]))
            for lang in sorted({r["lang"] for r in gen2})},
        "hits_by_lang_v1": {
            lang: float(np.mean([h for h, r in zip(hits_v1, v1_test) if r["lang"] == lang]))
            for lang in sorted({r["lang"] for r in v1_test})},
    })

    # ── 2. TF-IDF ──
    print("\n[2] TF-IDF + LogReg (обучение на train v1, C по gold-dev)")
    tfidf_pred, tfidf_info = fit_tfidf(train_recs, dev_recs, dev_y)
    print(f"    C={tfidf_info['C']}, признаков {tfidf_info['n_features']}, "
          f"dev F1 {tfidf_info['dev_f1']:.4f}")
    record("tfidf", tfidf_pred(v1_test), tfidf_pred(gen2), extra={"fit": tfidf_info})

    # ── 3. слой 2: эмбеддинги (torch) СНАЧАЛА, lightgbm ПОСЛЕ ──
    print(f"\n[3] слой 2: {ENCODER} + LightGBM ({AGG})")
    emb = encode_corpus(records, ENCODER, ROOT / "runs" / "layer2" / "cache")
    emb_g2 = encode_corpus(gen2, ENCODER, OUT_DIR / "cache")
    emb_all = {**emb, **emb_g2}
    l2_v1, l2_g2, l2_info = [], [], []
    for seed in SEEDS:
        pred, info = fit_layer2(train_recs, dev_recs, dev_y, emb_all, seed)
        l2_v1.append(pred(v1_test))
        l2_g2.append(pred(gen2))
        l2_info.append({"seed": seed, **info})
    # голосование по трём сидам — как в разделе «Тесты значимости»
    record("layer2", (np.mean(l2_v1, axis=0) >= 0.5).astype(int),
           (np.mean(l2_g2, axis=0) >= 0.5).astype(int),
           extra={"per_seed": l2_info,
                  "per_seed_gen2_f1": [metrics_from(y_gen2_silver, p)["f1"] for p in l2_g2]})

    # ── 4. xlm-roberta-base ──
    if not a.skip_xlmr:
        print(f"\n[4] xlm-roberta-base, {len(MODEL_DIRS)} сида (сохранённые веса)")
        pv1, pg2, per_seed = [], [], []
        for d in MODEL_DIRS:
            prob_v1 = predict_xlmr(v1_test, d)
            prob_g2 = predict_xlmr(gen2, d)
            pv1.append((prob_v1 >= 0.5).astype(int))
            pg2.append((prob_g2 >= 0.5).astype(int))
            per_seed.append({
                "dir": d.name,
                "v1_gold_f1": metrics_from(y_v1_gold, pv1[-1])["f1"],
                "gen2_silver_f1": metrics_from(y_gen2_silver, pg2[-1])["f1"],
            })
            print(f"    {d.name}: v1 gold F1 {per_seed[-1]['v1_gold_f1']:.4f} | "
                  f"gen2 F1 {per_seed[-1]['gen2_silver_f1']:.4f}")
        record("xlmr", (np.mean(pv1, axis=0) >= 0.5).astype(int),
               (np.mean(pg2, axis=0) >= 0.5).astype(int),
               extra={"per_seed": per_seed,
                      "v1_gold_f1_mstd": mstd([s["v1_gold_f1"] for s in per_seed]),
                      "gen2_f1_mstd": mstd([s["gen2_silver_f1"] for s in per_seed])})

    results["elapsed_sec"] = round(time.time() - t0, 1)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== итог: падение F1 при переходе на чужой генератор ===")
    print(f"{'модель':<12} {'v1 (silver)':>12} {'gen2':>8} {'падение':>9} "
          f"{'silver v1':>10} {'silver gen2':>12}")
    for name, m in results["models"].items():
        print(f"{name:<12} {m['v1_test_vs_silver']['f1']:>12.4f} "
              f"{m['gen2_vs_silver']['f1']:>8.4f} {m['f1_drop_silver']:>+9.4f} "
              f"{m['silver_agreement_v1']:>10.4f} {m['silver_agreement_gen2']:>12.4f}")
    print(f"\nрезультаты: {rel(out)}  ({results['elapsed_sec']} с)")
    print("графики: python scripts/make_gen2_plots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
