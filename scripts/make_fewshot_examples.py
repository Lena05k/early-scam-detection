#!/usr/bin/env python3
"""Шесть фиксированных few-shot примеров для слоя 4 — ТОЛЬКО из train.

Состав по заданию: 2 примера на язык, 3 scam / 3 benign, то есть на каждый язык
ровно один scam и один benign. Отбор детерминированный (без случайности):

  scam   — предпочитается онсет ≥ 2 (чтобы промпт не внушал «онсет всегда 1»),
           непустые подтипы, длина ближе к 10 репликам;
  benign — предпочитается hard_negative=1 (деньги/ссылка/срочность БЕЗ обмана —
           главный источник ложных тревог, инвариант 6), длина ближе к 10.

Тай-брейк везде по `id`, поэтому результат воспроизводим побитово.

⚠ Метки train — SILVER (`gold_src=llm_draft`, DeepSeek). То есть few-shot примеры
несут разметочные соглашения DeepSeek, а не человека. Это прямое следствие
требования «примеры только из train» и учитывается при чтении результатов:
few-shot режимы подталкивают модель к silver-конвенциям.

  python scripts/make_fewshot_examples.py
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_binary import read_corpus  # noqa: E402

TARGET_TURNS = 10


def dialog_text(rec):
    """Диалог в том же виде, в каком он уйдёт в LLM: одна реплика на строку."""
    return "\n".join(f"[{t['i']}|{t['role']}] {t['text']}" for t in rec["turns"])


def pick(records, lang, want_scam, used_scenarios):
    """Детерминированный выбор одного примера языка.

    used_scenarios — сценарии, уже занятые другими примерами: при прочих равных
    берётся НОВЫЙ сценарий, иначе все три benign схлопываются в один и тот же
    (honest_marketing) и промпт не показывает разнообразия hard-negatives.
    """
    pool = [r for r in records if r["lang"] == lang and int(r["is_scam"]) == int(want_scam)]
    if want_scam:
        pref = [r for r in pool if (r.get("onset") or 0) >= 2 and r.get("subtypes")]
    else:
        pref = [r for r in pool if int(r.get("hard_negative", 0)) == 1]
    pool = pref or pool
    fresh = [r for r in pool if r.get("scam_type") not in used_scenarios]
    pool = fresh or pool
    return min(pool, key=lambda r: (abs(len(r["turns"]) - TARGET_TURNS), r["id"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--out", default="runs/layer4/few_shot_examples.json")
    args = ap.parse_args()

    records = read_corpus(args.corpus)
    train = [r for r in records if r.get("split") == "train"]
    print(f"train: {len(train)} диалогов (метки silver)")

    examples, used = [], set()
    for lang in ("en", "ru", "zh"):
        for want_scam in (1, 0):
            r = pick(train, lang, want_scam, used)
            used.add(r.get("scam_type"))
            examples.append({
                "id": r["id"], "lang": lang, "n_turns": len(r["turns"]),
                "scenario": r.get("scam_type"), "hard_negative": int(r.get("hard_negative", 0)),
                "dialog": dialog_text(r),
                "answer": {"is_scam": int(r["is_scam"]),
                           "subtypes": list(r.get("subtypes") or []),
                           "onset": r.get("onset") if int(r["is_scam"]) == 1 else None},
            })

    n_scam = sum(e["answer"]["is_scam"] for e in examples)
    assert len(examples) == 6 and n_scam == 3, "должно быть 6 примеров, 3 scam / 3 benign"
    assert len({e["id"] for e in examples}) == 6, "дубликаты среди примеров"
    ids = {r["id"] for r in train}
    assert all(e["id"] in ids for e in examples), "пример не из train!"

    for e in examples:
        print(f"  {e['lang']}  is_scam={e['answer']['is_scam']}  "
              f"onset={e['answer']['onset']}  hard_neg={e['hard_negative']}  "
              f"{e['n_turns']:>2} реплик  {e['scenario']:<18} {e['id']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source_split": "train", "label_source": "silver (llm_draft, DeepSeek)",
        "selection": "детерминированный: scam с онсетом ≥2 и подтипами, benign с "
                     "hard_negative=1, длина ближе к 10 репликам, тай-брейк по id",
        "warning": "метки примеров silver, а не человеческие — few-shot несёт "
                   "разметочные соглашения DeepSeek",
        "examples": examples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
