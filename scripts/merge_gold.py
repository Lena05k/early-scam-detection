#!/usr/bin/env python3
"""Слияние человеческой золотой разметки с корпусом (стадия 2, пакетно).

Читает CSV разметки с РЕАЛЬНЫМИ id (см. check_annotations.py --resolved-out) и
корпус с ЧЕРНОВЫМИ метками; пишет новый jsonl, где у размеченных диалогов
is_scam / subtypes / onset взяты у человека, gold_src="human_reviewed",
review={by,date,mode}. Черновые метки НЕ теряются — уезжают в поле `draft`.

Провенанс `review.mode`: "full-text" (view_mode=full — НЕ префиксная
верификация онсета, инвариант 5) / "sequential" (view_mode=seq).

hard_negative: если аннотатор не проставил поле ни разу, корпусное значение
СОХРАНЯЕТСЯ (перезапись нулями удалила бы сигнал инварианта 6). Случаи
hard_negative=1 при человеческом is_scam=1 противоречивы по определению
(hard-neg = benign без обмана) и выписываются в conflicts.

Пример:
  python scripts/merge_gold.py data/annotations/annotations_test_2026-08-06_resolved.csv \
      --corpus data/corpus/corpus_v1.jsonl \
      --out data/corpus/corpus_v1_gold.jsonl \
      --by elena --date 2026-08-06
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

SUBTYPES = ["phishing_link", "money_request", "other"]
LABEL_FIELDS = ["is_scam", "subtypes", "onset", "hard_negative"]


def read_gold_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if not did:
                continue
            onset = (r.get("onset") or "").strip()
            hn = (r.get("hard_negative") or "").strip()
            out[did] = {
                "is_scam": int(r["is_scam"]),
                "subtypes": [s.strip() for s in (r.get("subtypes") or "").split("|") if s.strip()],
                "onset": int(onset) if onset else None,
                "hard_negative": int(hn) if hn else None,
                "doubt": int(r.get("doubt") or 0),
                "note": (r.get("note") or "").strip(),
                "view_mode": (r.get("view_mode") or "").strip(),
                "time_spent_sec": float(r.get("time_spent_sec") or 0),
            }
    return out


def merge_record(rec, g, by, date, use_human_hn):
    """Черновая запись + человеческое суждение → золотая запись. Чистая функция.

    use_human_hn=False → корпусное hard_negative сохраняется (колонка не
    использовалась аннотатором: UI пишет 0 по умолчанию, поэтому «0 везде»
    неотличим от «не трогал»).
    """
    out = dict(rec)
    out["draft"] = {k: rec.get(k) for k in LABEL_FIELDS}
    out["draft"]["gold_src"] = rec.get("gold_src")

    out["is_scam"] = g["is_scam"]
    out["subtypes"] = [s for s in SUBTYPES if s in g["subtypes"]] if g["is_scam"] else []
    out["onset"] = g["onset"] if g["is_scam"] else None

    if use_human_hn and g["hard_negative"] is not None:
        out["hard_negative"] = g["hard_negative"]
        out["hard_negative_src"] = "human"
    else:
        out["hard_negative_src"] = "llm_draft"   # колонка не использована — не трогаем

    out["gold_src"] = "human_reviewed"
    out["review"] = {
        "by": by, "date": date,
        "mode": {"full": "full-text", "seq": "sequential"}.get(g["view_mode"], "unknown"),
        "time_spent_sec": g["time_spent_sec"],
    }
    if g["doubt"]:
        out["review"]["doubt"] = 1
    if g["note"]:
        out["review"]["note"] = g["note"]
    return out


def check_invariants(rec):
    """Инварианты 1/3/6 + правило консистентности v1.2 → список нарушений."""
    bad = []
    if rec["is_scam"] not in (0, 1):
        bad.append("is_scam не 0/1")
    if rec["is_scam"] == 0:
        if rec.get("onset") is not None:
            bad.append("benign с онсетом (инвариант 3)")
        if rec.get("subtypes"):
            bad.append("benign с подтипами")
    else:
        if rec.get("onset") is None:
            bad.append("scam без онсета (v1.2)")
        elif not (1 <= rec["onset"] <= len(rec.get("turns") or [])):
            bad.append("онсет вне [1..T]")
        if not rec.get("subtypes"):
            bad.append("scam без подтипов (v1.2)")
    for s in rec.get("subtypes") or []:
        if s not in SUBTYPES:
            bad.append(f"неизвестный подтип {s}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gold_csv")
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--out", default="data/corpus/corpus_v1_gold.jsonl")
    p.add_argument("--by", default="human")
    p.add_argument("--date", required=True, help="дата разметки, YYYY-MM-DD")
    p.add_argument("--overwrite-hard-negative", action="store_true",
                   help="писать hard_negative=0 там, где аннотатор оставил поле пустым "
                        "(по умолчанию корпусное значение СОХРАНЯЕТСЯ)")
    args = p.parse_args()

    gold = read_gold_csv(args.gold_csv)
    # Колонка считается использованной, только если где-то стоит 1: HTML-пак пишет
    # 0 по умолчанию, поэтому сплошные нули = «аннотатор поле не трогал».
    hn_used = any(g["hard_negative"] for g in gold.values())
    use_human_hn = hn_used or args.overwrite_hard_negative

    records = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_id = {r["id"]: r for r in records}

    missing = sorted(set(gold) - set(by_id))
    if missing:
        sys.exit(f"❌ {len(missing)} id из CSV нет в корпусе: {missing[:5]}")

    # ── совпадение черновика с человеком (по полям) ──
    n = len(gold)
    match = collections.Counter()
    full_match = 0
    for i, g in gold.items():
        d = by_id[i]
        d_subs = set(d.get("subtypes") or [])
        g_subs = set(g["subtypes"]) if g["is_scam"] else set()
        g_onset = g["onset"] if g["is_scam"] else None
        d_onset = d.get("onset") or None
        ok_scam = g["is_scam"] == int(d.get("is_scam", 0))
        ok_subs = g_subs == d_subs
        ok_onset = g_onset == d_onset
        match["is_scam"] += ok_scam
        match["subtypes"] += ok_subs
        match["onset"] += ok_onset
        full_match += ok_scam and ok_subs and ok_onset

    out_records, conflicts, violations = [], [], []
    for r in records:
        if r["id"] in gold:
            m = merge_record(r, gold[r["id"]], args.by, args.date, use_human_hn)
            if m.get("hard_negative") == 1 and m["is_scam"] == 1:
                conflicts.append(m["id"])
            bad = check_invariants(m)
            if bad:
                violations.append((m["id"], bad))
            out_records.append(m)
        else:
            out_records.append(r)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── отчёт ──
    modes = collections.Counter(g["view_mode"] for g in gold.values())
    print(f"слито {n} золотых меток в {len(out_records)} записей → {args.out}")
    print(f"  gold_src=human_reviewed: {n}, остальные {len(out_records) - n} остались llm_draft")
    print(f"  review.mode: full-text {modes.get('full', 0)} / sequential {modes.get('seq', 0)}")

    print(f"\nсовпадение чернового LLM с человеком (n={n}):")
    for k in ("is_scam", "subtypes", "onset"):
        print(f"  {k:9s} {match[k]:4d}/{n}  = {match[k]/n:.1%}")
    print(f"  все три {full_match:4d}/{n}  = {full_match/n:.1%}")

    if not hn_used:
        kept = sum(1 for r in out_records if r.get("hard_negative_src") == "llm_draft"
                   and r.get("hard_negative") == 1)
        if args.overwrite_hard_negative:
            print("\n⚠ hard_negative=1 не проставлен ни разу, но --overwrite-hard-negative: "
                  "колонка человека применена (нули).")
        else:
            print(f"\n⚠ hard_negative=1 не проставлен ни разу (UI пишет 0 по умолчанию → "
                  f"колонка считается неиспользованной). Корпусные значения сохранены, "
                  f"{kept} флагов hard_negative=1 уцелело; hard_negative_src=llm_draft.")
    if conflicts:
        print(f"⚠ {len(conflicts)} записей: hard_negative=1 при человеческом is_scam=1 "
              f"(противоречит инварианту 6): {conflicts[:5]}"
              f"{' …' if len(conflicts) > 5 else ''}")
    if violations:
        print(f"❌ {len(violations)} нарушений инвариантов:")
        for i, bad in violations[:10]:
            print(f"    {i}: {'; '.join(bad)}")
    else:
        print("✅ инварианты 1/3/6 и консистентность v1.2 — нарушений нет")


if __name__ == "__main__":
    main()
