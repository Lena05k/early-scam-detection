#!/usr/bin/env python3
"""Разметка scenario_group в корпусе — защита от утечки «близнецов» при сплите.

Считает попарную похожесть ТОЛЬКО внутри групп (язык × тип сценария) по
символьным n-граммам (char_wb, n=3..5), собирает связные компоненты (union-find)
и вписывает поле scenario_group в каждую запись JSONL. Твин-пары (twin_id)
объединяются принудительно. Диалоги и twin_id не трогаются — только разметка.

Пример:
  python scripts/annotate_scenario_groups.py
  python scripts/annotate_scenario_groups.py --in data/corpus/corpus_v1.jsonl \
      --report reports/scenario_groups_report.json

Перед записью делает резервную копию входного файла (<file>.bak_scenario_groups).
"""
import argparse
import json
import pathlib
import shutil
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from scam_onset.data.scenario_groups import (  # noqa: E402
    SIM_FIRST2_THRESHOLD,
    SIM_FULL_THRESHOLD,
    NGRAM_MIN,
    NGRAM_MAX,
    build_scenario_groups,
)

DEFAULT_IN = "data/corpus/corpus_v1.jsonl"
DEFAULT_REPORT = "reports/scenario_groups_report.json"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_turn(rec):
    turns = rec.get("turns") or []
    return turns[0].get("text", "") if turns else ""


def build_report(records, result, args):
    group_of = result["group_of"]
    groups = result["groups"]
    sizes = Counter(len(m) for m in groups.values())
    multi = {g: m for g, m in groups.items() if len(m) > 1}
    n_in_multi = sum(len(m) for m in multi.values())
    max_size = max((len(m) for m in groups.values()), default=0)
    largest_gid = max(groups, key=lambda g: len(groups[g])) if groups else None

    by_id = {r["id"]: r for r in records}
    # топ-10 самых похожих пар
    top = sorted(result["pairs"], key=lambda p: max(p[2], p[3]), reverse=True)[:10]
    top_pairs = [
        {
            "a": a, "b": b,
            "lang": by_id[a]["lang"], "scam_type": by_id[a]["scam_type"],
            "sim_first2": round(s2, 4), "sim_full": round(sf, 4),
            "a_turn1": first_turn(by_id[a])[:200],
            "b_turn1": first_turn(by_id[b])[:200],
        }
        for a, b, s2, sf in top
    ]

    # предупреждение о крупных группах (честный сплит станет несбалансированным)
    warn = []
    if max_size > 0.02 * len(records):
        warn.append(
            f"крупнейшая группа {largest_gid} содержит {max_size} диалогов "
            f"({max_size/len(records)*100:.1f}% корпуса) — сплит по группам "
            f"может стать несбалансированным"
        )

    return {
        "params": {
            "sim_first2_threshold": SIM_FIRST2_THRESHOLD,
            "sim_full_threshold": SIM_FULL_THRESHOLD,
            "ngram_range": [NGRAM_MIN, NGRAM_MAX],
            "analyzer": "char_wb",
        },
        "n_dialogs": len(records),
        "n_groups": len(groups),
        "n_singleton_groups": sizes.get(1, 0),
        "n_multi_groups": len(multi),
        "n_dialogs_in_multi_groups": n_in_multi,
        "max_group_size": max_size,
        "largest_group_id": largest_gid,
        "forced_twin_merges": result["forced_twins"],
        "n_qualified_pairs": len(result["pairs"]),
        "group_size_distribution": {str(k): sizes[k] for k in sorted(sizes)},
        "top_similar_pairs": top_pairs,
        "warnings": warn,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--no-backup", action="store_true", help="не делать .bak (для тестов)")
    ap.add_argument("--dry-run", action="store_true", help="не писать JSONL, только отчёт в stdout")
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    print(f"Прочитано диалогов: {len(records)}", file=sys.stderr)

    result = build_scenario_groups(records)
    for r in records:
        r["scenario_group"] = result["group_of"][r["id"]]

    report = build_report(records, result, args)

    if not args.dry_run:
        # 2.1 резервная копия перед любой записью
        if not args.no_backup:
            bak = args.inp + ".bak_scenario_groups"
            shutil.copy2(args.inp, bak)
            print(f"Резервная копия: {bak}", file=sys.stderr)
        write_jsonl(args.inp, records)
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Отчёт: {args.report}", file=sys.stderr)

    # краткая сводка в stdout
    print(json.dumps({
        "n_groups": report["n_groups"],
        "n_multi_groups": report["n_multi_groups"],
        "n_dialogs_in_multi_groups": report["n_dialogs_in_multi_groups"],
        "max_group_size": report["max_group_size"],
        "forced_twin_merges": report["forced_twin_merges"],
        "group_size_distribution": report["group_size_distribution"],
        "warnings": report["warnings"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
