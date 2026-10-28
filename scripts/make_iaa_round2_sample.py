#!/usr/bin/env python3
"""Отбор 100 диалогов gold-test v2 под независимую повторную разметку (IAA, раунд 2).

Зачем: план (tech_report §12) требовал минимум двух аннотаторов, сделан один. Раздел
CLAUDE.md «Открытые ⚠» держит это как незакрытый хвост. Этот скрипт готовит выборку,
на которой второй человек размечает ВСЛЕПУЮ, а scripts/iaa_round2.py считает согласие.

Стратификация (в порядке применения, каждый уровень — largest remainder):
  1. язык                — ~33 на каждый (пул сбалансирован: ru 135 / en 134 / zh 134);
  2. is_scam             — в пропорции пула внутри языка;
  3. бины онсета         — {1, 2, 3-4, 5+} внутри scam, с полом 1 на каждый бин,
                           присутствующий в пуле языка (требование «покрыть все бины»);
  4. hard_negative       — в пропорции пула внутри ячейки, с глобальной проверкой доли.

⚠ hard_negative берётся из КОРПУСА (corpus_v1_gold.jsonl), а не из колонки gold-CSV:
   разметка test-пака не проставила флаг ни разу (403/403 = 0), см. CLAUDE.md «Открытые ⚠».

Слепота. dialog_id содержит имя сценария (corpus_en_real_2fa_0586) — это фактически
метка. Поэтому в HTML-пак уходит только непрозрачный pid; соответствие pid -> dialog_id
живёт ЗДЕСЬ, в data/gold/iaa_round2_sample.json, и аннотатору не показывается.

Ничего не переразмечает и gold-файлы не трогает: только читает и пишет свой JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "gold_test_v2.csv"
CORPUS = ROOT / "data" / "corpus" / "corpus_v1_gold.jsonl"
DEV_SPLIT = ROOT / "data" / "gold" / "gold_dev_split.json"
OUT = ROOT / "data" / "gold" / "iaa_round2_sample.json"

SEED = 20260812
N_SAMPLE = 100
BINS = ("1", "2", "3-4", "5+")


def onset_bin(onset):
    """Бин позиции онсета; None у benign (онсета нет)."""
    if onset is None:
        return None
    if onset == 1:
        return "1"
    if onset == 2:
        return "2"
    if onset <= 4:
        return "3-4"
    return "5+"


def read_gold(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if not did:
                continue
            onset = (r.get("onset") or "").strip()
            rows.append({
                "dialog_id": did,
                "is_scam": int(r["is_scam"]),
                "onset": int(onset) if onset else None,
            })
    return rows


def read_corpus(path, ids):
    found = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") in ids:
                found[obj["id"]] = obj
    return found


def largest_remainder(counts, total):
    """Пропорциональное распределение ``total`` по ключам ``counts`` (метод наибольших
    остатков). Детерминированно: остатки сортируются по (-дробная часть, ключ)."""
    keys = sorted(counts, key=str)
    pool = sum(counts.values())
    if pool == 0 or total <= 0:
        return {k: 0 for k in keys}
    exact = {k: counts[k] * total / pool for k in keys}
    base = {k: int(exact[k]) for k in keys}
    rest = total - sum(base.values())
    order = sorted(keys, key=lambda k: (-(exact[k] - base[k]), str(k)))
    for k in order[:rest]:
        base[k] += 1
    return base


def apply_floor(alloc, available, floor=1):
    """Поднимает каждую непустую в пуле позицию до ``floor``, забирая единицы у самых
    крупных. Возвращает (alloc, сколько единиц переставлено)."""
    alloc = dict(alloc)
    moved = 0
    need = [k for k in sorted(alloc, key=str)
            if available.get(k, 0) > 0 and alloc[k] < floor]
    for k in need:
        while alloc[k] < floor:
            donors = sorted(
                (d for d in alloc if d != k and alloc[d] > floor and alloc[d] > 0),
                key=lambda d: (-alloc[d], str(d)))
            if not donors:
                donors = sorted((d for d in alloc if d != k and alloc[d] > 0),
                                key=lambda d: (-alloc[d], str(d)))
            if not donors:
                break
            alloc[donors[0]] -= 1
            alloc[k] += 1
            moved += 1
    return alloc, moved


def allocate(pool, n_sample):
    """Иерархическая раскладка квот по стратам. pool: список dict-записей.

    Возвращает (quota, trace) — quota[(lang, is_scam, bin, hn)] = сколько отобрать.
    """
    trace = {"floor_moves": []}
    by_lang = Counter(r["lang"] for r in pool)
    lang_q = largest_remainder(by_lang, n_sample)
    trace["lang"] = dict(lang_q)

    quota = {}
    trace["by_lang"] = {}
    for lang in sorted(by_lang):
        recs = [r for r in pool if r["lang"] == lang]
        scam_pool = Counter(r["is_scam"] for r in recs)
        scam_q = largest_remainder(scam_pool, lang_q[lang])
        node = {"is_scam": dict(scam_q), "bins": {}}

        # --- scam: раскладка по бинам онсета, пол 1 на каждый присутствующий бин ---
        scam_recs = [r for r in recs if r["is_scam"] == 1]
        bin_pool = Counter(r["bin"] for r in scam_recs)
        bin_q = largest_remainder(bin_pool, scam_q.get(1, 0))
        bin_q, moved = apply_floor(bin_q, bin_pool, floor=1)
        if moved:
            trace["floor_moves"].append({"lang": lang, "level": "onset_bin", "moved": moved})
        node["bins"] = dict(bin_q)

        for b, q in bin_q.items():
            cell = [r for r in scam_recs if r["bin"] == b]
            hn_pool = Counter(r["hard_negative"] for r in cell)
            hn_q = largest_remainder(hn_pool, q)
            for hn, k in hn_q.items():
                if k:
                    quota[(lang, 1, b, hn)] = k

        # --- benign: бина онсета нет, делим только по hard_negative ---
        ben_recs = [r for r in recs if r["is_scam"] == 0]
        hn_pool = Counter(r["hard_negative"] for r in ben_recs)
        hn_q = largest_remainder(hn_pool, scam_q.get(0, 0))
        node["benign_hard_negative"] = dict(hn_q)
        for hn, k in hn_q.items():
            if k:
                quota[(lang, 0, None, hn)] = k
        trace["by_lang"][lang] = node
    return quota, trace


def pick(pool, quota, seed):
    """Детерминированный отбор внутри каждой страты.

    Квоты страт задаются allocate() и здесь НЕ меняются. Свобода остаётся только в том,
    КОГО именно взять внутри страты, и она тратится на разнообразие сценариев: внутри
    ячейки идёт round-robin по сценариям, а не слепая выборка. Без этого при 100 из 403
    целые сценарии выпадали (в частности real_bank_notice — один из трёх главных
    источников ложных тревог, 10 подтверждённых спорных), что для IAA плохо: согласие
    на hard negatives — самое интересное место.
    """
    cells = defaultdict(list)
    for r in pool:
        cells[(r["lang"], r["is_scam"], r["bin"], r["hard_negative"])].append(r)
    chosen = []
    for key in sorted(quota, key=str):
        k = quota[key]
        members = cells[key]
        if k > len(members):
            raise SystemExit(f"страта {key}: запрошено {k}, в пуле {len(members)}")
        rnd = random.Random(f"{seed}|{key}")
        by_scen = defaultdict(list)
        for x in sorted(members, key=lambda r: r["dialog_id"]):
            by_scen[x["scenario"]].append(x)
        for lst in by_scen.values():
            rnd.shuffle(lst)
        scen_order = sorted(by_scen)
        rnd.shuffle(scen_order)
        picked, i = [], 0
        while len(picked) < k:                      # round-robin по сценариям ячейки
            bucket = by_scen[scen_order[i % len(scen_order)]]
            if bucket:
                picked.append(bucket.pop())
            i += 1
        chosen.extend(picked)
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Выборка 100 диалогов под IAA раунд 2")
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    gold = read_gold(a.gold)
    corpus = read_corpus(a.corpus, {r["dialog_id"] for r in gold})
    missing = [r["dialog_id"] for r in gold if r["dialog_id"] not in corpus]
    if missing:
        raise SystemExit(f"нет в корпусе {len(missing)} id, первые: {missing[:5]}")
    assign = json.loads(pathlib.Path(DEV_SPLIT).read_text(encoding="utf-8"))["assignment"]

    pool = []
    for r in gold:
        obj = corpus[r["dialog_id"]]
        pool.append({
            "dialog_id": r["dialog_id"],
            "lang": obj["lang"],
            "scenario": obj["scam_type"],
            "scenario_group": obj.get("scenario_group"),
            "is_scam": r["is_scam"],
            "onset": r["onset"],
            "bin": onset_bin(r["onset"]),
            # флаг корпуса, НЕ колонка gold-CSV (та везде 0)
            "hard_negative": int(obj.get("hard_negative", 0) or 0),
            "n_turns": len(obj.get("turns") or []),
            "gold_dev": assign.get(r["dialog_id"]) == "dev",
        })

    print(f"пул: {len(pool)} диалогов из {a.gold}")
    print("  языки:", dict(Counter(r['lang'] for r in pool)))
    print("  is_scam:", dict(Counter(r['is_scam'] for r in pool)))
    print("  бины t*:", dict(Counter(str(r['bin']) for r in pool)))
    print("  hard_negative (корпус):", dict(Counter(r['hard_negative'] for r in pool)))

    quota, trace = allocate(pool, a.n)
    chosen = pick(pool, quota, a.seed)
    if len(chosen) != a.n:
        raise SystemExit(f"отобрано {len(chosen)}, ожидалось {a.n}")

    order = random.Random(f"{a.seed}|order")
    shuffled = sorted(chosen, key=lambda r: r["dialog_id"])
    order.shuffle(shuffled)
    items = []
    for i, r in enumerate(shuffled, 1):
        items.append({"pid": f"r2_{i:03d}", **r})

    # ---- проверки, которые обязаны пройти ----
    langs = Counter(i["lang"] for i in items)
    bins = Counter(i["bin"] for i in items if i["bin"])
    hn = sum(i["hard_negative"] for i in items)
    hn_pool_share = sum(r["hard_negative"] for r in pool) / len(pool)
    checks = {
        "n": len(items) == a.n,
        "unique_ids": len({i["dialog_id"] for i in items}) == a.n,
        "lang_balanced": all(abs(v - a.n / 3) <= 1.5 for v in langs.values()),
        "all_onset_bins_covered": all(bins.get(b, 0) > 0 for b in BINS),
        "hard_negative_present": hn > 0,
        "hard_negative_share_ok": hn >= int(hn_pool_share * a.n) - 1,
    }
    scam_share_pool = sum(r["is_scam"] for r in pool) / len(pool)
    scam_share_sample = sum(i["is_scam"] for i in items) / len(items)

    payload = {
        "task": "iaa_round2_sample",
        "seed": a.seed,
        "n": len(items),
        "source_gold": str(pathlib.Path(a.gold).relative_to(ROOT)),
        "source_corpus": str(pathlib.Path(a.corpus).relative_to(ROOT)),
        "pool_size": len(pool),
        "stratification": ["lang", "is_scam", "onset_bin", "hard_negative"],
        "allocation_trace": trace,
        "quota": {"|".join(map(str, k)): v for k, v in sorted(quota.items(), key=str)},
        "composition": {
            "lang": dict(langs),
            "is_scam": dict(Counter(i["is_scam"] for i in items)),
            "onset_bins": dict(bins),
            "hard_negative": hn,
            "scenarios": dict(Counter(i["scenario"] for i in items)),
            "gold_dev_members": sum(i["gold_dev"] for i in items),
            "scam_share_sample": round(scam_share_sample, 4),
            "scam_share_pool": round(scam_share_pool, 4),
            "hard_negative_share_sample": round(hn / len(items), 4),
            "hard_negative_share_pool": round(hn_pool_share, 4),
        },
        "checks": checks,
        "items": items,
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nотобрано: {len(items)} -> {out.relative_to(ROOT)}")
    print("  языки:", dict(langs))
    print(f"  is_scam: {dict(Counter(i['is_scam'] for i in items))} "
          f"(доля scam {scam_share_sample:.3f}, в пуле {scam_share_pool:.3f})")
    print(f"  бины t*: {dict(bins)}")
    print(f"  hard_negative: {hn} (доля {hn / len(items):.3f}, в пуле {hn_pool_share:.3f})")
    print(f"  сценариев: {len(payload['composition']['scenarios'])} из 19")
    print(f"  из них лежат в gold-dev: {payload['composition']['gold_dev_members']}")
    print("\nпроверки:")
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    if not all(checks.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
