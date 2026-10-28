#!/usr/bin/env python3
"""Собирает reports/iaa_round2.html — самодостаточный пак для ВТОРОГО аннотатора.

Вход: data/gold/iaa_round2_sample.json (выборка 100, scripts/make_iaa_round2_sample.py)
      + data/corpus/corpus_v1_gold.jsonl (тексты диалогов).
Выход: reports/iaa_round2.html — открыть двойным кликом, сервер не нужен.

СЛЕПОТА (главное требование раунда). В HTML не попадает НИЧЕГО, что подсказывает ответ:
  - меток Елены нет (ни is_scam, ни подтипов, ни онсета);
  - silver-меток DeepSeek нет;
  - реального dialog_id нет — он содержит имя сценария (corpus_en_real_2fa_0586),
    то есть фактически ответ; в паке только непрозрачный pid (r2_001…);
  - имени сценария, hard_negative и флага корпуса нет;
  - порядок перемешан (сид 20260812), твины и языки не идут блоками.
Соответствие pid -> dialog_id остаётся ТОЛЬКО в data/gold/iaa_round2_sample.json.
Скрипт проверяет утечку программно перед записью — assert'ы ниже.

⚠ localStorage/sessionStorage в шаблоне не используются: состояние живёт в памяти
вкладки, сохранение — кнопкой «Скачать CSV», продолжение — «Загрузить CSV».
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "gold" / "iaa_round2_sample.json"
CORPUS = ROOT / "data" / "corpus" / "corpus_v1_gold.jsonl"
TEMPLATE = pathlib.Path(__file__).resolve().parent / "iaa_round2_template.html"
OUT = ROOT / "reports" / "iaa_round2.html"


def load_corpus(path, ids):
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


def build_dialogs(items, corpus):
    """Только pid + реплики. Никаких меток и никаких имён сценариев."""
    out = []
    for it in items:
        obj = corpus[it["dialog_id"]]
        turns = [
            {"i": t.get("i"), "role": t.get("role", ""),
             "speaker": str(t.get("speaker", "")), "text": str(t.get("text", ""))}
            for t in (obj.get("turns") or [])
        ]
        if not turns:
            raise SystemExit(f"пустой диалог {it['dialog_id']}")
        out.append({"pid": it["pid"], "turns": turns})
    return out


def assert_blind(payload, items, corpus):
    """Программная проверка слепоты: ничего из ответов не должно попасть в пак."""
    blob = json.dumps(payload, ensure_ascii=False)
    # Метаданные = всё, КРОМЕ текста реплик: сам текст диалога — это и есть предмет
    # разметки, слова «refund»/«investment» встречаются в нём легитимно (в том числе
    # в benign-диалогах) и утечкой не являются. Утечка — имя сценария в СЛУЖЕБНЫХ полях.
    meta = {"pack": payload.get("pack"), "n": payload.get("n"),
            "dialogs": [{"pid": d["pid"],
                         "turns": [{k: v for k, v in t.items() if k != "text"}
                                   for t in d["turns"]]}
                        for d in payload["dialogs"]]}
    meta_blob = json.dumps(meta, ensure_ascii=False)
    # 1. реальных id нет нигде, включая текст
    leaked = [it["dialog_id"] for it in items if it["dialog_id"] in blob]
    assert not leaked, f"в пак утёк реальный dialog_id: {leaked[:3]}"
    # 2. имён сценариев нет в служебных полях (они же — ответ: real_2fa, romance…)
    scen = sorted({it["scenario"] for it in items})
    hit = [s for s in scen if re.search(rf"\b{re.escape(s)}\b", meta_blob)]
    assert not hit, f"в пак утекло имя сценария: {hit}"
    # 3. полей меток нет ни на одном уровне
    for d in payload["dialogs"]:
        bad = set(d) - {"pid", "turns"}
        assert not bad, f"лишние поля у {d['pid']}: {sorted(bad)}"
        for t in d["turns"]:
            bad = set(t) - {"i", "role", "speaker", "text"}
            assert not bad, f"лишние поля в реплике {d['pid']}: {sorted(bad)}"
    # 4. порядок не совпадает с сортировкой по реальному id (иначе id угадывается)
    by_real = [it["pid"] for it in sorted(items, key=lambda x: x["dialog_id"])]
    assert [d["pid"] for d in payload["dialogs"]] != by_real, "порядок = сортировка по id"
    # 5. номера реплик сплошные, роли известны
    for d in payload["dialogs"]:
        assert [t["i"] for t in d["turns"]] == list(range(1, len(d["turns"]) + 1)), \
            f"нумерация реплик не сплошная: {d['pid']}"
        assert all(t["role"] in ("atk", "tgt") for t in d["turns"]), \
            f"неизвестная role: {d['pid']}"
        assert any(t["role"] == "atk" for t in d["turns"]), \
            f"нет ни одной реплики A (онсет поставить некуда): {d['pid']}"


def main():
    ap = argparse.ArgumentParser(description="Пак для второго аннотатора (IAA раунд 2)")
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    sample = json.loads(pathlib.Path(a.sample).read_text(encoding="utf-8"))
    items = sample["items"]
    corpus = load_corpus(a.corpus, {it["dialog_id"] for it in items})
    missing = [it["dialog_id"] for it in items if it["dialog_id"] not in corpus]
    if missing:
        raise SystemExit(f"нет в корпусе {len(missing)}: {missing[:5]}")

    dialogs = build_dialogs(items, corpus)
    payload = {"pack": "iaa_round2", "n": len(dialogs), "dialogs": dialogs}
    assert_blind(payload, items, corpus)

    html = pathlib.Path(a.template).read_text(encoding="utf-8")
    if "__DATA__" not in html:
        raise SystemExit("в шаблоне нет плейсхолдера __DATA__")
    # Ищем ОБРАЩЕНИЕ к хранилищу (localStorage.get…, sessionStorage[…]), а не упоминание
    # слова: в комментарии шаблона оно стоит законно — там сказано, что их не используют.
    used = re.findall(r"(?:window\s*\.\s*)?(localStorage|sessionStorage)\s*(?:\.|\[)", html)
    if used:
        raise SystemExit(f"шаблон обращается к {sorted(set(used))} — запрещено в этом паке")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")            # безопасно внутри <script>
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__DATA__", blob), encoding="utf-8")

    n_turns = [len(d["turns"]) for d in dialogs]
    print(f"записано: {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} КБ)")
    print(f"диалогов: {len(dialogs)} | реплик всего: {sum(n_turns)} "
          f"| мин/медиана/макс: {min(n_turns)}/{sorted(n_turns)[len(n_turns) // 2]}/{max(n_turns)}")
    print(f"языки (из выборки, в паке НЕ показаны): "
          f"{dict(Counter(i['lang'] for i in items))}")
    print("слепота: реальные id, имена сценариев и любые метки в пак не попали (проверено)")
    print(f"соответствие pid -> dialog_id: {pathlib.Path(a.sample).relative_to(ROOT)} "
          f"— аннотатору НЕ открывать")
    return 0


if __name__ == "__main__":
    sys.exit(main())
