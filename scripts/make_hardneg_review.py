#!/usr/bin/env python3
"""Собирает reports/hardneg_review.html — самодостаточный интерфейс для ручного
разбора спорных диалогов (корпусный hard_negative=1 при человеческом is_scam=1).

Данные вшиваются в HTML при генерации: ничего не подгружается, сервер не нужен.
Ничего не переразмечает: решения принимает человек, результат — скачанный CSV.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISAGREEMENTS = ROOT / "reports" / "gold_vs_draft_test_disagreements.csv"
CORPUS = ROOT / "data" / "corpus" / "corpus_v1_gold.jsonl"
TEMPLATE = Path(__file__).resolve().parent / "hardneg_review_template.html"
OUT = ROOT / "reports" / "hardneg_review.html"

REQUIRED_FIELDS = ("id", "lang", "scam_type", "turns", "is_scam", "onset", "split")


def load_disputed() -> list[dict]:
    with DISAGREEMENTS.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        r for r in rows
        if r.get("llm_hard_negative") == "1" and r.get("human_is_scam") == "1"
    ]


def load_corpus(ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") in ids:
                found[obj["id"]] = obj
    return found


def check_structure(sample: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in sample]
    if missing:
        print("НЕ ХВАТАЕТ ПОЛЕЙ В JSONL:", ", ".join(missing))
        print("Структура первой найденной записи:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:4000])
        raise SystemExit(1)


def build_items(disputed: list[dict], corpus: dict[str, dict]) -> list[dict]:
    items = []
    for row in disputed:
        did = row["dialog_id"]
        obj = corpus[did]
        turns = [
            {
                "i": t.get("i"),
                "role": t.get("role", ""),
                "speaker": t.get("speaker", ""),
                "text": t.get("text", ""),
            }
            for t in obj["turns"]
        ]
        items.append({
            "dialog_id": did,
            "lang": obj["lang"],
            "scenario": obj["scam_type"],
            "n_turns": len(turns),
            "generator": obj.get("generator", ""),
            "turns": turns,
            "gold": {
                "is_scam": int(row["human_is_scam"]),
                "onset": row.get("human_onset") or None,
                "subtypes": row.get("human_subtypes") or "",
                "note": row.get("note") or "",
                "view_mode": row.get("view_mode") or "",
            },
            "silver": {
                "is_scam": int(row["llm_is_scam"]),
                "onset": row.get("llm_onset") or None,
                "subtypes": row.get("llm_subtypes") or "",
                "hard_negative": int(row["llm_hard_negative"]),
            },
        })
    return items


def main() -> None:
    disputed = load_disputed()
    ids = {r["dialog_id"] for r in disputed}
    corpus = load_corpus(ids)

    missing = ids - set(corpus)
    if missing:
        print(f"НЕ НАЙДЕНО В КОРПУСЕ {len(missing)} id:", sorted(missing)[:10])
        raise SystemExit(1)

    check_structure(next(iter(corpus.values())))

    items = build_items(disputed, corpus)
    items.sort(key=lambda x: (x["lang"], x["scenario"], x["dialog_id"]))

    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # безопасно внутри <script>

    html = TEMPLATE.read_text(encoding="utf-8")
    assert "__DATA__" in html, "в шаблоне нет плейсхолдера __DATA__"
    html = html.replace("__DATA__", payload)
    OUT.write_text(html, encoding="utf-8")

    empty = [i["dialog_id"] for i in items if not i["turns"]]
    blank_text = [
        i["dialog_id"] for i in items
        if not any((t["text"] or "").strip() for t in i["turns"])
    ]

    print(f"записано: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} КБ)")
    print(f"диалогов вшито: {len(items)}")
    print(f"языки: {dict(Counter(i['lang'] for i in items))}")
    print(f"сценарии: {dict(Counter(i['scenario'] for i in items))}")
    print(f"реплик всего: {sum(i['n_turns'] for i in items)}"
          f", мин/медиана/макс: "
          f"{min(i['n_turns'] for i in items)}/"
          f"{sorted(i['n_turns'] for i in items)[len(items)//2]}/"
          f"{max(i['n_turns'] for i in items)}")
    print(f"пустых диалогов: {len(empty)}; диалогов без текста: {len(blank_text)}")


if __name__ == "__main__":
    main()
