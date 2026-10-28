#!/usr/bin/env python3
"""Слой 0: тривиальные бейзлайны is_scam — нижняя граница качества.

Шесть предсказателей, ни один не обучает нейросеть:

  1. const_scam    — всегда scam (показывает базовую долю класса)
  2. const_benign  — всегда benign
  3. length        — порог по числу реплик, направление и порог по gold-dev
  4. url           — регулярка на http/https/www и укороченные ссылки
  5. keywords      — списки слов, добытые из train (silver); порог N по gold-dev
  6. logreg3       — логрегрессия на трёх признаках (длина, есть URL, число слов):
                     обучение на train (silver), сила регуляризации по gold-dev

Протокол тот же, что у остальных моделей проекта: пороги подбираются на train
или gold-dev, финальный замер — gold-test v2 (303 диалога). Разбиение gold
берётся готовым из gold_dev_split.json и не пересоздаётся (проверяется ассертами).

СИДЫ НЕ НУЖНЫ: все шесть предсказателей детерминированы. У 1–5 нет ни одного
случайного элемента вообще; у 6 `LogisticRegression` с солвером lbfgs на
фиксированных признаках даёт побитово один и тот же результат при любом
`random_state` (сид в lbfgs не используется). Поэтому здесь нет ни среднего, ни
std по сидам — в отличие от трансформерных прогонов.

⚠ Признаки считаются по ТЕКСТУ реплик без имён говорящих: имя говорящего в этом
корпусе почти эквивалентно метке (92.1% имён atk встречаются только в scam либо
только в benign), и бейзлайн на именах измерял бы артефакт генератора.

  python scripts/train_layer0.py
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import read_corpus, read_gold_csv  # noqa: E402

# ── укороченные ссылки + обычные http/https/www ──
SHORTENERS = [
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "is.gd", "cutt.ly", "ow.ly",
    "rb.gy", "shorturl.at", "clck.ru", "vk.cc", "clc.to", "u.to",
    "dwz.cn", "t.cn", "url.cn", "suo.im", "sourl.cn", "mrw.so",
]
URL_RE = re.compile(
    r"(https?://|www\.|" + "|".join(re.escape(s) for s in SHORTENERS) + r")",
    re.IGNORECASE,
)

# кандидаты в ключевые слова
WORD_RE = re.compile(r"[a-zà-ÿа-яё]{3,}", re.IGNORECASE)   # ru/en: слова от 3 букв
CJK_RE = re.compile(r"[一-鿿]+")                    # zh: только иероглифы

N_KEYWORDS = 20          # сколько слов оставляем на язык
MIN_DF = 10              # кандидат должен встретиться минимум в 10 диалогах языка
KW_GRID = range(1, 11)   # сетка порога «сработало N и более слов»
C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# опубликованные числа обученных моделей — для лесенки (см. CLAUDE.md)
TRAINED = [
    ("TF-IDF + LogReg", 0.8638, "v1"),
    ("mDeBERTa-v3-base", 0.8942, "v1"),
    ("xlm-roberta-base", 0.9344, "v2"),
]

RU_NAME = {
    "const_scam": "константа «всегда scam»",
    "const_benign": "константа «всегда benign»",
    "length": "длина диалога (порог)",
    "url": "наличие URL",
    "keywords": "ключевые слова",
    "logreg3": "логрегрессия на 3 признаках",
}


# ─────────────────────────── данные и признаки ───────────────────────────
def plain_text(rec):
    """Текст диалога без имён говорящих и без служебной разметки."""
    return "\n".join(t["text"] for t in rec["turns"])


def has_url(text):
    return int(bool(URL_RE.search(text)))


def candidates(text, lang):
    """Множество кандидатов-подстрок в тексте: слова (ru/en) или n-граммы (zh)."""
    low = text.lower()
    if lang == "zh":
        out = set()
        for chunk in CJK_RE.findall(low):
            for n in (2, 3):
                for i in range(len(chunk) - n + 1):
                    out.add(chunk[i:i + n])
        return out
    return set(WORD_RE.findall(low))


def mine_keywords(train_recs, n_keywords=N_KEYWORDS, min_df=MIN_DF):
    """Слова с наибольшей разницей ЧАСТОТ ДОКУМЕНТОВ между scam и benign в train.

    Считается доля диалогов языка, где кандидат встречается: score = df_scam/N_scam
    − df_benign/N_benign. Метки train — silver (DeepSeek); это часть постановки,
    слой 0 обучающих меток почти не использует, но списки слов из них выводятся.
    """
    by_lang = {}
    for lang in sorted({r["lang"] for r in train_recs}):
        recs = [r for r in train_recs if r["lang"] == lang]
        n_scam = sum(1 for r in recs if int(r["is_scam"]) == 1)
        n_ben = len(recs) - n_scam
        df_s, df_b = {}, {}
        for r in recs:
            tgt = df_s if int(r["is_scam"]) == 1 else df_b
            for c in candidates(plain_text(r), lang):
                tgt[c] = tgt.get(c, 0) + 1
        scored = []
        for c in set(df_s) | set(df_b):
            s, b = df_s.get(c, 0), df_b.get(c, 0)
            if s + b < min_df:
                continue
            scored.append((s / n_scam - b / n_ben, c, s, b))
        # tie-break по самому слову → порядок детерминирован
        scored.sort(key=lambda x: (-x[0], x[1]))

        chosen = []
        for score, c, s, b in scored:
            if lang == "zh" and any(c in k["word"] or k["word"] in c for k in chosen):
                continue          # у иероглифных n-грамм режем вложенные повторы
            chosen.append({"word": c, "score": round(score, 4),
                           "df_scam": s, "df_benign": b,
                           "rate_scam": round(s / n_scam, 4),
                           "rate_benign": round(b / n_ben, 4)})
            if len(chosen) == n_keywords:
                break
        by_lang[lang] = {"n_train_scam": n_scam, "n_train_benign": n_ben,
                         "keywords": chosen}
    return by_lang


def compile_keywords(by_lang):
    """Скомпилированные матчеры: ru/en по границам слова, zh — подстрокой."""
    out = {}
    for lang, d in by_lang.items():
        words = [k["word"] for k in d["keywords"]]
        if lang == "zh":
            out[lang] = [(w, None) for w in words]
        else:
            out[lang] = [(w, re.compile(rf"(?<![a-zà-ÿа-яё]){re.escape(w)}"
                                        rf"(?![a-zà-ÿа-яё])", re.IGNORECASE))
                         for w in words]
    return out


def kw_hits(text, lang, compiled):
    """Сколько РАЗНЫХ ключевых слов языка встретилось в диалоге."""
    low = text.lower()
    n = 0
    for w, rx in compiled.get(lang, []):
        n += 1 if (w in low if rx is None else bool(rx.search(low))) else 0
    return n


# ─────────────────────────────── метрики ───────────────────────────────
def metrics_from(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    k = cohen_kappa_score(y_true, y_pred)
    if not np.isfinite(k):          # константный предсказатель → κ не определена
        k = 0.0
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f), "cohen_kappa": float(k)}


def f1_of(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


# ──────────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--gold", default="data/gold/gold_test_v2.csv")
    ap.add_argument("--gold-dev", default="data/gold/gold_dev_split.json")
    ap.add_argument("--out-dir", default="runs/layer0")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    records = read_corpus(args.corpus)
    gold = read_gold_csv(args.gold)
    train_recs = [r for r in records if r.get("split") == "train"]

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
    print(f"gold-файл: {args.gold}   train (silver) {len(train_recs)}")

    # ── ключевые слова из train ──
    kw_tables = mine_keywords(train_recs)
    compiled = compile_keywords(kw_tables)
    print(f"\nключевые слова из train (по {N_KEYWORDS} на язык, min_df={MIN_DF}; "
          f"score = доля scam-диалогов − доля benign-диалогов):")
    for lang, d in kw_tables.items():
        print(f"\n  [{lang}]  train: scam {d['n_train_scam']} / benign {d['n_train_benign']}")
        for i, k in enumerate(d["keywords"], 1):
            print(f"    {i:>2}. {k['word']:<22} score {k['score']:+.3f}   "
                  f"scam {k['rate_scam']:.3f} / benign {k['rate_benign']:.3f}")

    # ── признаки ──
    def feats(recs):
        txt = [plain_text(r) for r in recs]
        return {
            "n_turns": np.array([len(r["turns"]) for r in recs], dtype=float),
            "has_url": np.array([has_url(t) for t in txt], dtype=float),
            "n_kw": np.array([kw_hits(t, r["lang"], compiled)
                              for t, r in zip(txt, recs)], dtype=float),
        }

    F_tr, F_dev, F_te = feats(train_recs), feats(dev_recs), feats(test_recs)
    y_tr = np.array([int(r["is_scam"]) for r in train_recs])
    y_dev = np.array([r["gold_is_scam"] for r in dev_recs])
    y_te = np.array([r["gold_is_scam"] for r in test_recs])

    preds, tuning = {}, {}

    # 1–2. константы
    preds["const_scam"] = np.ones(len(test_recs), dtype=int)
    preds["const_benign"] = np.zeros(len(test_recs), dtype=int)

    # 3. длина диалога: порог и направление по gold-dev
    thr_grid = sorted({int(v) for v in F_tr["n_turns"]} | {int(v) for v in F_dev["n_turns"]})
    sweep_len = []
    for thr in thr_grid:
        for direction in ("ge", "le"):
            p = ((F_dev["n_turns"] >= thr) if direction == "ge"
                 else (F_dev["n_turns"] <= thr)).astype(int)
            sweep_len.append({"threshold": thr, "direction": direction,
                              "dev_f1": f1_of(y_dev, p)})
    best_len = max(sweep_len, key=lambda d: (d["dev_f1"], -d["threshold"], d["direction"]))
    preds["length"] = ((F_te["n_turns"] >= best_len["threshold"])
                       if best_len["direction"] == "ge"
                       else (F_te["n_turns"] <= best_len["threshold"])).astype(int)
    tuning["length"] = {"selected": best_len, "sweep": sweep_len,
                        "rule": ("число реплик ≥ " if best_len["direction"] == "ge"
                                 else "число реплик ≤ ") + str(best_len["threshold"])
                                + " → scam"}

    # 4. URL
    preds["url"] = F_te["has_url"].astype(int)
    tuning["url"] = {"rule": "есть http/https/www или укоротитель → scam",
                     "dev_f1": f1_of(y_dev, F_dev["has_url"].astype(int)),
                     "n_shorteners": len(SHORTENERS)}

    # 5. ключевые слова: порог N по gold-dev
    sweep_kw = [{"n": int(n), "dev_f1": f1_of(y_dev, (F_dev["n_kw"] >= n).astype(int))}
                for n in KW_GRID]
    best_kw = max(sweep_kw, key=lambda d: (d["dev_f1"], -d["n"]))
    preds["keywords"] = (F_te["n_kw"] >= best_kw["n"]).astype(int)
    tuning["keywords"] = {"selected_n": best_kw["n"], "dev_f1": best_kw["dev_f1"],
                          "sweep": sweep_kw,
                          "rule": f"сработало ≥ {best_kw['n']} ключевых слов → scam"}

    # 6. логрегрессия на трёх признаках
    Xtr = np.column_stack([F_tr[k] for k in ("n_turns", "has_url", "n_kw")])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xdev = np.column_stack([F_dev[k] for k in ("n_turns", "has_url", "n_kw")])
    Xte = np.column_stack([F_te[k] for k in ("n_turns", "has_url", "n_kw")])
    Xtr_s, Xdev_s, Xte_s = ((m - mu) / sd for m in (Xtr, Xdev, Xte))
    sweep_c = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000)
        clf.fit(Xtr_s, y_tr)
        sweep_c.append({"C": C, "dev_f1": f1_of(y_dev, clf.predict(Xdev_s))})
    best_c = max(sweep_c, key=lambda d: (d["dev_f1"], -d["C"]))
    clf = LogisticRegression(C=best_c["C"], max_iter=5000).fit(Xtr_s, y_tr)
    preds["logreg3"] = clf.predict(Xte_s).astype(int)
    tuning["logreg3"] = {
        "selected_C": best_c["C"], "dev_f1": best_c["dev_f1"], "sweep": sweep_c,
        "features": ["n_turns", "has_url", "n_kw"],
        "standardization": {"mean": mu.tolist(), "std": sd.tolist(), "fit_on": "train"},
        "coefficients": dict(zip(["n_turns", "has_url", "n_kw"],
                                 [float(c) for c in clf.coef_[0]])),
        "intercept": float(clf.intercept_[0]),
        "rule": "P(scam) > 0.5 по трём стандартизованным признакам",
    }

    # ── замер ──
    langs = sorted({r["lang"] for r in test_recs})
    results = {}
    for name, yp in preds.items():
        m = metrics_from(y_te, yp)
        per_lang = {}
        for lg in langs:
            idx = [i for i, r in enumerate(test_recs) if r["lang"] == lg]
            per_lang[lg] = metrics_from(y_te[idx], yp[idx]) | {"n": len(idx)}
        results[name] = {
            "name_ru": RU_NAME[name], "metrics": m,
            "confusion_matrix": confusion_matrix(y_te, yp, labels=[0, 1]).tolist(),
            "per_language": per_lang, "tuning": tuning.get(name, {"rule": "константа"}),
            "predicted_scam_rate": float(yp.mean()),
        }

    # ── диагностика: почему предсказатели ведут себя именно так ──
    silver = {r["id"]: int(r["is_scam"]) for r in records}
    y_silver = np.array([silver[r["id"]] for r in test_recs])
    n_url_corpus = sum(1 for r in records if has_url(plain_text(r)))
    diag = {
        "dialogs_with_url": {"corpus": n_url_corpus, "train": int(F_tr["has_url"].sum()),
                             "gold_dev": int(F_dev["has_url"].sum()),
                             "gold_test": int(F_te["has_url"].sum()),
                             "corpus_total": len(records)},
        "n_turns_range": {"train": [int(F_tr["n_turns"].min()), int(F_tr["n_turns"].max())],
                          "gold_test": [int(F_te["n_turns"].min()), int(F_te["n_turns"].max())]},
        "corr_n_kw_n_turns": {
            "train": float(np.corrcoef(F_tr["n_kw"], F_tr["n_turns"])[0, 1]),
            "gold_test": float(np.corrcoef(F_te["n_kw"], F_te["n_turns"])[0, 1])},
        "n_kw_mean_by_gold": {
            "scam": float(F_te["n_kw"][y_te == 1].mean()),
            "benign": float(F_te["n_kw"][y_te == 0].mean())},
        "silver_agreement_on_test": {n: float((yp == y_silver).mean())
                                     for n, yp in preds.items()},
        "silver_vs_gold_on_test": float((y_silver == y_te).mean()),
    }
    # контрфактика: тот же список слов, но порог N выбран по train (silver), а не по gold-dev.
    # Показывает, откуда у ключевых слов высокий recall — из человеческой точки отсечения.
    sweep_kw_silver = [{"n": int(n),
                        "train_silver_f1": f1_of(y_tr, (F_tr["n_kw"] >= n).astype(int))}
                       for n in KW_GRID]
    n_silver = max(sweep_kw_silver, key=lambda d: (d["train_silver_f1"], -d["n"]))["n"]
    m_silver = metrics_from(y_te, (F_te["n_kw"] >= n_silver).astype(int))
    diag["keywords_threshold_by_train_silver"] = {
        "selected_n": n_silver, "sweep": sweep_kw_silver, "test_metrics": m_silver}
    print("\nдиагностика:")
    print(f"  диалогов с настоящим URL во всём корпусе: {n_url_corpus} из {len(records)}")
    print(f"  число реплик, train {diag['n_turns_range']['train']} / "
          f"gold-test {diag['n_turns_range']['gold_test']}")
    print(f"  корреляция (число сработавших слов, число реплик): "
          f"train {diag['corr_n_kw_n_turns']['train']:.3f}, "
          f"gold-test {diag['corr_n_kw_n_turns']['gold_test']:.3f}")
    print(f"  среднее число сработавших слов: scam {diag['n_kw_mean_by_gold']['scam']:.2f} / "
          f"benign {diag['n_kw_mean_by_gold']['benign']:.2f}")
    print(f"  порог ключевых слов по train (silver): N={n_silver} → gold-test F1 "
          f"{m_silver['f1']:.4f} (recall {m_silver['recall']:.4f}); "
          f"по gold-dev: N={best_kw['n']} → F1 {f1_of(y_te, preds['keywords']):.4f} "
          f"(recall {metrics_from(y_te, preds['keywords'])['recall']:.4f})")
    print("  совпадение с silver на тесте: " + ", ".join(
        f"{k} {v:.3f}" for k, v in diag["silver_agreement_on_test"].items())
        + f"  (сам silver против gold {diag['silver_vs_gold_on_test']:.3f})")

    order = ["const_scam", "const_benign", "length", "url", "keywords", "logreg3"]
    print(f"\ngold-test v2 (n={len(test_recs)}), доля scam в gold {y_te.mean():.4f} "
          f"(dev {y_dev.mean():.4f}). Сиды не нужны: все предсказатели детерминированы.\n")
    hdr = f"{'предсказатель':<30} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'κ':>7}  правило"
    print(hdr)
    print("-" * len(hdr))
    for n in order:
        m = results[n]["metrics"]
        print(f"{results[n]['name_ru']:<30} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['cohen_kappa']:>7.4f}  "
              f"{results[n]['tuning']['rule']}")

    print("\nпо языкам (F1 / recall):")
    print(f"{'предсказатель':<30}" + "".join(f"{lg:>20}" for lg in langs))
    for n in order:
        row = f"{results[n]['name_ru']:<30}"
        for lg in langs:
            pl = results[n]["per_language"][lg]
            row += f"{pl['f1']:>13.4f}/{pl['recall']:.2f}"
        print(row)

    print("\nматрицы ошибок (строки gold, столбцы предсказание):")
    for n in order:
        cm = results[n]["confusion_matrix"]
        print(f"  {results[n]['name_ru']:<30} [[{cm[0][0]:>3}, {cm[0][1]:>3}], "
              f"[{cm[1][0]:>3}, {cm[1][1]:>3}]]")

    # ── сводная лесенка ──
    ladder = [{"model": results[n]["name_ru"], "f1": results[n]["metrics"]["f1"],
               "layer": "слой 0", "gold": "v2"} for n in order]
    ladder += [{"model": m, "f1": f, "layer": "обученные", "gold": g} for m, f, g in TRAINED]
    ladder.sort(key=lambda d: d["f1"])
    print("\nлесенка (F1 на gold-test, 303 диалога):")
    for d in ladder:
        bar = "█" * int(round(d["f1"] * 40))
        print(f"  {d['model']:<32} {d['f1']:.4f} {bar:<41} {d['layer']} · gold {d['gold']}")
    print("⚠ TF-IDF и mDeBERTa — опубликованные числа на gold v1; xlm-r и слой 0 — на v2.")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "layer 0 — тривиальные бейзлайны is_scam",
        "deterministic": True,
        "seeds": "не нужны: предсказатели 1–5 не содержат случайности, "
                 "LogisticRegression(lbfgs) на фиксированных признаках детерминирована",
        "gold_file": args.gold,
        "split": {"file": args.gold_dev, "seed": sm["seed"],
                  "n_dev": len(dev_recs), "n_test": len(test_recs)},
        "n_train": len(train_recs),
        "gold_scam_rate": {"dev": float(y_dev.mean()), "test": float(y_te.mean())},
        "feature_text": "тексты реплик без имён говорящих",
        "keyword_mining": {"n_per_lang": N_KEYWORDS, "min_df": MIN_DF,
                           "source": "train split, silver-метки",
                           "score": "df_scam/N_scam − df_benign/N_benign",
                           "tables": kw_tables},
        "url_regex": URL_RE.pattern,
        "diagnostics": diag,
        "predictors": results,
        "ladder": ladder,
        "trained_reference": [{"model": m, "f1": f, "gold": g} for m, f, g in TRAINED],
        "note_tfidf_v2": ("контрольный пересчёт TF-IDF на gold v2 тем же "
                          "scripts/train_tfidf.py --gold data/gold/gold_test_v2.csv даёт "
                          "F1 0.8713 (против опубликованных 0.8638 на v1); в лесенке "
                          "оставлено опубликованное число"),
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out_dir / 'results.json'}")

    if not args.no_plots:
        from make_layer0_plots import make_plots
        make_plots(payload, out_dir / "plots")


if __name__ == "__main__":
    main()
