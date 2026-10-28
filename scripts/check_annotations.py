#!/usr/bin/env python3
"""Проверка CSV аннотаций против манифеста пака.

Полнота (все pid пака размечены), консистентность v1.2 (scam → t* и >=1
подтип; benign → без подтипов/t*; hard_negative только у benign), диапазон
t* ∈ [1..T] и попадание в реплику атакующего.

Примеры:
  python scripts/check_annotations.py annotations_test_2026-07-21.csv \
      --manifest data/annotations/packs/pack_test403_s20260723.manifest.json
  ... --allow-partial            # частичная разметка (промежуточный экспорт)
  ... --resolved-out gold.csv    # копия CSV с реальными id корпуса
"""
import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scam_onset.data import annotation_pack as ap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path")
    p.add_argument("--manifest", required=True)
    p.add_argument("--allow-partial", action="store_true",
                   help="не требовать полноты (промежуточный экспорт)")
    p.add_argument("--resolved-out", default=None,
                   help="записать копию CSV с реальными dialog_id корпуса")
    args = p.parse_args()

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    try:
        rows = ap.read_annotations_csv(args.csv_path)
    except ValueError as e:
        sys.exit(f"❌ {e}")

    result = ap.validate_annotations(rows, manifest, allow_partial=args.allow_partial)

    ids = result["checks"]["ids"]
    print(f"пак {manifest['pack_id']}: размечено {ids['n_annotated']} / {ids['n_pack']}")
    for name, c in result["checks"].items():
        print(f"  {'✅' if c['pass'] else '❌'} {name}")
        for key in ("unknown", "duplicates", "missing"):
            if c.get(key):
                print(f"      {key}: {c[key][:10]}{' …' if len(c[key]) > 10 else ''}")
        for e in c.get("errors", [])[:20]:
            print(f"      {e}")
        if len(c.get("errors", [])) > 20:
            print(f"      … и ещё {len(c['errors']) - 20}")

    vm = result["checks"].get("view_mode", {})
    if vm.get("n_full"):
        print(f"  ⚠ {vm['n_full']} меток размечены в режиме «весь текст сразу» "
              f"(не префиксная верификация онсета, инвариант 5); "
              f"seq: {vm.get('n_seq', 0)}, без пометки: {vm.get('n_unknown', 0)}")

    if args.resolved_out and result["pass"]:
        resolved = ap.resolve_dialog_ids(rows, manifest)
        with open(args.resolved_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(resolved)
        print(f"CSV с реальными id: {args.resolved_out}")

    if not result["pass"]:
        sys.exit("❌ проверка НЕ пройдена")
    print("✅ проверка пройдена")


if __name__ == "__main__":
    main()
