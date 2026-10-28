#!/usr/bin/env python3
"""Манифест обучающих выборок для кросс-язычного прогона.

Ноутбук на Kaggle не должен повторять логику отбора: он читает ГОТОВЫЕ списки
`dialog_id` из этого манифеста. Так конфигурации файнтюна побитово совпадают с теми,
на которых посчитаны слои 1 и 2, и подвыборка не разъезжается из-за разных версий
numpy или порядка обхода.

  python scripts/make_crosslingual_manifest.py
"""
import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import (KS, LANGS, SEEDS, build_train, enumerate_configs,  # noqa: E402
                          load_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/crosslingual/train_manifest.json")
    args = ap.parse_args()

    data = load_all()
    cfgs = enumerate_configs()
    entries = {}
    for cfg in cfgs:
        recs, comp = build_train(cfg, data["train"])
        ids = [r["id"] for r in recs]
        assert len(set(ids)) == len(ids), f"дубликаты id в {cfg['name']}"
        entries[cfg["name"]] = {
            "config": cfg, "composition": comp, "train_ids": ids,
            "md5": hashlib.md5("\n".join(ids).encode()).hexdigest(),
        }

    # вложенность few-shot: k=10 ⊂ k=50 ⊂ k=100
    for src in LANGS:
        for tgt in LANGS:
            if tgt == src:
                continue
            sets = {k: set(entries[f"fewshot_{src}->{tgt}_k{k}"]["train_ids"]) for k in KS}
            for a, b in zip(KS, KS[1:]):
                assert sets[a] <= sets[b], f"few-shot не вложен: {src}->{tgt} {a}⊄{b}"
    # zero-shot: в single-source не должно быть ни одного диалога целевого языка
    for lg in LANGS:
        by = entries[f"single_{lg}"]["composition"]["by_lang"]
        assert all(by[o] == 0 for o in LANGS if o != lg), f"утечка языка в single_{lg}"
        assert len(entries[f"single_{lg}"]["train_ids"]) == by[lg]

    payload = {
        "source": "scripts/crosslingual.py (enumerate_configs + build_train)",
        "corpus": "data/corpus/corpus_v1.jsonl", "split_field": "train",
        "gold": "data/gold/gold_test_v2.csv",
        "seeds": SEEDS, "langs": LANGS, "ks": KS,
        "n_configs": len(entries), "configs": entries,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(e["train_ids"]) for e in entries.values())
    print(f"конфигураций {len(entries)}, суммарно {total} обучающих строк "
          f"({total * len(SEEDS)} с учётом сидов)")
    print("проверки пройдены: вложенность few-shot, отсутствие целевого языка в single")
    print(f"сохранено: {p}  ({p.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
