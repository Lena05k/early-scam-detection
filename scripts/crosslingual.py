#!/usr/bin/env python3
"""Кросс-язычный перенос: общие конфигурации обучения для всех трёх моделей.

Модуль задаёт ЕДИНЫЙ набор конфигураций, чтобы матрица «обучали на X → тест на Y»
считалась одинаково для слоя 1 (TF-IDF), слоя 2 (эмбеддинги + LightGBM) и файнтюна
(xlm-roberta-base, отдельный ноутбук на Kaggle).

Конфигурации:
  single-source            en / ru / zh                       (n≈464)
  leave-one-out            en+ru / en+zh / ru+zh              (n≈928)
  all-3                    en+ru+zh                           (n=1392)
  + вариант, ВЫРОВНЕННЫЙ ПО РАЗМЕРУ до 464 для всех, где train больше
  few-shot перенос         single-source + k примеров цели, k ∈ {10, 50, 100}

⚠ Контроль размера обязателен: один язык даёт ~464 примера, три — 1392, и без
подвыборки эффект переноса смешается с эффектом объёма. Подвыборка
детерминированная (SUBSAMPLE_SEED), стратифицированная по lang × is_scam.

⚠ Zero-shot: целевой язык не участвует НИ В ЧЁМ. Отбор гиперпараметра/эпохи идёт
по gold-dev, отфильтрованному по языкам-источникам (33–34 при одном источнике,
66–67 при двух). Это шумно: ДИ шире в 1.71× при одном источнике, а для en-источника
выбор совпадает с выбором полного dev лишь в 19.9% бутстрап-выборок
(`scripts/crosslingual_dev_ci.py`). Поэтому пишем метрики на ВСЕХ точках сетки для
нескольких наборов отбора сразу, а сам отбор делается ПОСТФАКТУМ — это позволяет
показать и честный zero-shot вариант, и контроль по silver-val источников, ничего
не переобучая.

Few-shot примеры цели вложены: k=10 ⊂ k=50 ⊂ k=100, стратифицированы scam/benign.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import read_corpus, read_gold_csv  # noqa: E402

LANGS = ["en", "ru", "zh"]
KS = [10, 50, 100]
SUBSAMPLE_SEED = 20260810      # подвыборка train до 464
FEWSHOT_SEED = 20260811        # выбор целевых примеров
SEEDS = [20260721, 20260722, 20260723]


# ───────────────────────────── конфигурации ─────────────────────────────
def enumerate_configs(size_matched_n=464):
    """Полный список конфигураций обучения. Порядок фиксирован."""
    cfgs = []
    for lg in LANGS:
        cfgs.append({"name": f"single_{lg}", "kind": "single", "sources": [lg],
                     "size_matched": False, "k": 0})
    for a, b in (("en", "ru"), ("en", "zh"), ("ru", "zh")):
        cfgs.append({"name": f"loo_{a}+{b}", "kind": "loo", "sources": [a, b],
                     "size_matched": False, "k": 0})
        cfgs.append({"name": f"loo_{a}+{b}_n{size_matched_n}", "kind": "loo",
                     "sources": [a, b], "size_matched": True, "k": 0})
    cfgs.append({"name": "all3", "kind": "all3", "sources": list(LANGS),
                 "size_matched": False, "k": 0})
    cfgs.append({"name": f"all3_n{size_matched_n}", "kind": "all3",
                 "sources": list(LANGS), "size_matched": True, "k": 0})
    for src in LANGS:
        for tgt in LANGS:
            if tgt == src:
                continue
            for k in KS:
                cfgs.append({"name": f"fewshot_{src}->{tgt}_k{k}", "kind": "fewshot",
                             "sources": [src], "target": tgt, "size_matched": False,
                             "k": k})
    return cfgs


def _stratified_subsample(recs, n, seed):
    """Подвыборка ровно n записей, пропорционально по (язык × is_scam)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for r in recs:
        groups.setdefault((r["lang"], int(r["is_scam"])), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["id"])            # детерминированная база
    total = len(recs)
    quotas, chosen = {}, []
    for key, g in sorted(groups.items()):
        quotas[key] = int(round(n * len(g) / total))
    # правим округление до ровно n
    diff = n - sum(quotas.values())
    for key in sorted(quotas, key=lambda k: -len(groups[k])):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        quotas[key] += step
        diff -= step
    for key, g in sorted(groups.items()):
        idx = rng.permutation(len(g))[:quotas[key]]
        chosen += [g[i] for i in sorted(idx)]
    chosen.sort(key=lambda r: r["id"])
    assert len(chosen) == n, f"подвыборка дала {len(chosen)}, ожидалось {n}"
    return chosen


def _fewshot_pool(train, target, seed):
    """Вложенные k: перемешиваем один раз внутри класса, дальше берём префиксы."""
    rng = np.random.default_rng(seed)
    pools = {}
    for cls in (0, 1):
        g = sorted([r for r in train if r["lang"] == target
                    and int(r["is_scam"]) == cls], key=lambda r: r["id"])
        pools[cls] = [g[i] for i in rng.permutation(len(g))]
    return pools


def build_train(cfg, train, size_matched_n=464):
    """Обучающая выборка конфигурации. Возвращает (записи, описание состава)."""
    pool = [r for r in train if r["lang"] in cfg["sources"]]
    if cfg["size_matched"]:
        pool = _stratified_subsample(pool, size_matched_n, SUBSAMPLE_SEED)
    extra = []
    if cfg["kind"] == "fewshot":
        pools = _fewshot_pool(train, cfg["target"], FEWSHOT_SEED)
        half = cfg["k"] // 2
        extra = pools[1][:half] + pools[0][:cfg["k"] - half]   # стратифицировано
        extra.sort(key=lambda r: r["id"])
    recs = sorted(pool + extra, key=lambda r: r["id"])
    comp = {"n_total": len(recs), "n_source": len(pool), "n_target_shots": len(extra),
            "by_lang": {lg: sum(1 for r in recs if r["lang"] == lg) for lg in LANGS},
            "scam_share": float(np.mean([int(r["is_scam"]) for r in recs]))}
    return recs, comp


# ─────────────────────────────── данные ───────────────────────────────
def load_all(corpus="data/corpus/corpus_v1.jsonl", gold="data/gold/gold_test_v2.csv",
             split_file="data/gold/gold_dev_split.json", verbose=True):
    """Корпус + gold v2 + разбиение dev/test с ассертами (сплиты не трогаем)."""
    records = read_corpus(corpus)
    gold_map = read_gold_csv(gold)
    train = [r for r in records if r.get("split") == "train"]
    val = [r for r in records if r.get("split") == "val"]

    all_test = []
    for r in records:
        if r.get("split") != "test":
            continue
        r2 = {k: v for k, v in r.items() if k != "is_scam"}
        if r2["id"] in gold_map:
            r2["gold_is_scam"] = gold_map[r2["id"]]
            all_test.append(r2)
    if len(all_test) != len(gold_map):
        raise SystemExit(f"❌ джойн неполный: {len(all_test)} из {len(gold_map)}")

    sm = json.loads(pathlib.Path(split_file).read_text(encoding="utf-8"))
    assign = sm["assignment"]
    dev = [r for r in all_test if assign.get(r["id"]) == "dev"]
    test = [r for r in all_test if assign.get(r["id"]) != "dev"]
    assert len(dev) == sm["n_dev"] and len(test) == sm["n_test"], \
        "разбиение gold не совпало с gold_dev_split.json"
    gd = {r.get("scenario_group") for r in dev}
    gt = {r.get("scenario_group") for r in test}
    assert not (gd & gt), "группы scenario_group разорваны между dev и test"
    if verbose:
        print(f"разбиение gold: seed={sm['seed']} dev={len(dev)} test={len(test)} "
              f"(групп {len(gd)}/{len(gt)}, пересечение 0)")
        print(f"train {len(train)} / val {len(val)} (silver); "
              f"по языкам train: " +
              ", ".join(f"{lg} {sum(1 for r in train if r['lang'] == lg)}" for lg in LANGS))
    return {"records": records, "train": train, "val": val, "dev": dev, "test": test,
            "split_meta": sm}


# ─────────────────────────────── метрики ───────────────────────────────
def metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    k = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(k if np.isfinite(k) else 0.0)}


def metrics_by_lang(recs, y_true, y_pred):
    """Общие метрики + отдельно по каждому языку (тест всегда полный, 303)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"overall": metrics(y_true, y_pred) | {"n": len(y_true)}}
    for lg in LANGS:
        idx = [i for i, r in enumerate(recs) if r["lang"] == lg]
        if idx:
            out[lg] = metrics(y_true[idx], y_pred[idx]) | {"n": len(idx)}
    return out


def selection_sets(dev, val, sources):
    """Наборы для отбора: честный zero-shot (только источники) и контроли.

    dev_source  — gold-dev по языкам-источникам: единственный вариант, где целевой
                  язык не участвует ни в чём (основной для zero-shot).
    val_source  — silver-val по языкам-источникам: контроль, у него больше примеров.
    dev_full    — полный gold-dev: НЕ zero-shot, служит только верхней границей.
    """
    return {
        "dev_source": [r for r in dev if r["lang"] in sources],
        "val_source": [r for r in val if r["lang"] in sources],
        "dev_full": list(dev),
    }


def mstd(vals):
    """Среднее, std, медиана и сами значения.

    Медиана нужна не для красоты: у кросс-язычных конфигураций отдельные сиды
    схлопываются в вырожденный предсказатель (recall≈0), и среднее по трём сидам
    смешивает коллапс с нормальными прогонами, давая величину, которой не
    соответствует ни один реальный прогон.
    """
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "values": v}


if __name__ == "__main__":
    data = load_all()
    cfgs = enumerate_configs()
    print(f"\nконфигураций: {len(cfgs)} "
          f"(ядро {sum(1 for c in cfgs if c['kind'] != 'fewshot')}, "
          f"few-shot {sum(1 for c in cfgs if c['kind'] == 'fewshot')})\n")
    print(f"{'конфигурация':<28}{'n':>6}{'источник':>10}{'цель':>7}   состав по языкам")
    for c in cfgs:
        recs, comp = build_train(c, data["train"])
        by = " ".join(f"{lg}:{comp['by_lang'][lg]}" for lg in LANGS if comp['by_lang'][lg])
        print(f"{c['name']:<28}{comp['n_total']:>6}{comp['n_source']:>10}"
              f"{comp['n_target_shots']:>7}   {by}  scam {comp['scam_share']:.3f}")
