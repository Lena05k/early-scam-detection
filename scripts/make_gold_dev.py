#!/usr/bin/env python3
"""Разрез золотого теста на gold-dev (отбор чекпоинта) и gold-test (финальная оценка).

Зачем: отбирать чекпоинт по val-сплиту нельзя — его метки silver (копия LLM), а
recall там насыщен в 1.0 с первой эпохи, так что отбор по факту случаен. Дев из
ЧЕЛОВЕЧЕСКИХ меток даёт осмысленный сигнал отбора ценой сжатия теста.

Разрез групповой и стратифицированный, ровно как основные сплиты:
  - целые `scenario_group` не рвутся (твины не расходятся по разные стороны);
  - стратификация по lang × gold is_scam;
  - фиксированный seed, результат воспроизводим.

Ни gold-CSV, ни корпус не изменяются — пишется отдельный маппинг.

  python scripts/make_gold_dev.py --n-dev 100
"""
import argparse
import collections
import csv
import json
import pathlib
import random
import sys


def read_gold(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if did:
                out[did] = int(r["is_scam"])
    return out


def read_meta(corpus_path, ids):
    meta = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in ids:
                meta[r["id"]] = {"lang": r["lang"],
                                 "group": r.get("scenario_group") or r["id"],
                                 "split": r.get("split")}
    return meta


def carve(gold, meta, n_dev, seed):
    """Групповой стратифицированный отбор ~n_dev диалогов в dev. Чистая функция."""
    groups = collections.defaultdict(list)
    for did in gold:
        groups[meta[did]["group"]].append(did)

    total = len(gold)
    cells = collections.Counter((meta[d]["lang"], gold[d]) for d in gold)
    quota = {c: round(n * n_dev / total) for c, n in cells.items()}

    rnd = random.Random(seed)
    order = sorted(groups)          # детерминированный старт
    rnd.shuffle(order)

    dev, have = set(), collections.Counter()
    for g in order:
        members = groups[g]
        if len(dev) + len(members) > n_dev:
            continue
        gain = collections.Counter((meta[d]["lang"], gold[d]) for d in members)
        # берём группу, только если ни одну ячейку не переполняем
        if all(have[c] + k <= quota.get(c, 0) for c, k in gain.items()):
            dev.update(members)
            have.update(gain)

    # добор одиночными группами, если квоты недовыбраны из-за крупных групп
    for g in order:
        if len(dev) >= n_dev:
            break
        members = groups[g]
        if set(members) & dev or len(dev) + len(members) > n_dev:
            continue
        dev.update(members)
        have.update(collections.Counter((meta[d]["lang"], gold[d]) for d in members))

    return dev, quota, have


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", default="data/gold/gold_test_v1.csv")
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--out", default="data/gold/gold_dev_split.json")
    p.add_argument("--n-dev", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260808)
    args = p.parse_args()

    gold = read_gold(args.gold)
    meta = read_meta(args.corpus, set(gold))
    missing = sorted(set(gold) - set(meta))
    if missing:
        sys.exit(f"❌ {len(missing)} id из gold нет в корпусе: {missing[:5]}")
    bad = [d for d in gold if meta[d]["split"] != "test"]
    if bad:
        sys.exit(f"❌ {len(bad)} id gold лежат не в test-сплите: {bad[:5]}")

    dev, quota, have = carve(gold, meta, args.n_dev, args.seed)
    mapping = {d: ("dev" if d in dev else "test") for d in sorted(gold)}

    # проверки целостности
    gdev = {meta[d]["group"] for d in dev}
    gtest = {meta[d]["group"] for d in gold if d not in dev}
    overlap = gdev & gtest
    assert not overlap, f"группы разорваны между dev и test: {sorted(overlap)[:5]}"

    n_test = len(gold) - len(dev)
    print(f"gold всего {len(gold)} → dev {len(dev)} / test {n_test}")
    print(f"групп: dev {len(gdev)}, test {len(gtest)}, пересечение {len(overlap)}  ✅")
    print(f"\n{'ячейка':>14} {'квота':>6} {'взято':>6} {'в gold':>7}")
    cells = collections.Counter((meta[d]["lang"], gold[d]) for d in gold)
    for c in sorted(cells):
        print(f"{c[0]+' is_scam='+str(c[1]):>14} {quota.get(c,0):6d} {have[c]:6d} {cells[c]:7d}")
    dev_pos = sum(gold[d] for d in dev) / len(dev)
    test_pos = sum(gold[d] for d in gold if d not in dev) / n_test
    print(f"\nдоля scam: dev {dev_pos:.3f} / test {test_pos:.3f} "
          f"(в целом {sum(gold.values())/len(gold):.3f})")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"seed": args.seed, "n_dev": len(dev), "n_test": n_test,
                                "assignment": mapping}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nмаппинг: {outp}  (gold-CSV и корпус не изменялись)")


if __name__ == "__main__":
    main()
