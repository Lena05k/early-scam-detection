#!/usr/bin/env python3
"""Слой 4 без API-ключа: LLM-разметчик — модель, ведущая сессию (не DeepSeek).

Зачем. Слот «слой 4» пуст: единственный доступный ключ — DeepSeek, а он породил
корпус и silver-метки, поэтому измеряет согласие с самим собой (см. CLAUDE.md,
«Слой 4, LLM few-shot»). Здесь роль LLM исполняет модель сессии. Прецедент в
проекте есть: пилот сгенерирован in-session без API.

Протокол. Промпты, схема ответа, разбор и метрики — импортируются из
scripts/train_layer4.py БЕЗ изменений. Разбиение gold переиспользуется из
data/gold/gold_dev_split.json с теми же ассертами (100/303, группы не пересечены).
Замер — на gold-test v2.

⚠ Слепота. В батчи уходят ТОЛЬКО непрозрачные pid: реальный dialog_id содержит
имя сценария (corpus_en_real_2fa_0586), то есть фактически ответ. Соответствие
pid → dialog_id лежит в манифесте и при разметке не открывается.

Использование:
  python3 scripts/layer4_insession.py dump  --split dev  --mode few_rules
  python3 scripts/layer4_insession.py score --split dev  --mode few_rules \
      --answers runs/layer4_insession/answers/dev_few_rules.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_layer4 import (  # noqa: E402
    INSTRUCTION, RULES, SUBTYPES, evaluate, parse_answer,
)
from make_fewshot_examples import dialog_text  # noqa: E402
from train_binary import read_corpus, read_gold_csv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "layer4_insession"
PID_SEED = 20260907


def load_split(corpus, gold_csv, gold_dev):
    """Записи dev/test с теми же ассертами, что в train_layer4.main."""
    records = read_corpus(corpus)
    gold = read_gold_csv(gold_csv)
    gold_sub, gold_onset = {}, {}
    import csv
    with open(gold_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            did = (row.get("dialog_id") or "").strip()
            if not did:
                continue
            raw = (row.get("subtypes") or "").strip()
            gold_sub[did] = [s.strip() for s in raw.split("|") if s.strip()] if raw else []
            o = (row.get("onset") or "").strip()
            gold_onset[did] = int(o) if o else None

    all_test = [r for r in records if r.get("split") == "test" and r["id"] in gold]
    if len(all_test) != len(gold):
        sys.exit(f"❌ джойн неполный: {len(all_test)} из {len(gold)}")
    sm = json.loads(pathlib.Path(gold_dev).read_text(encoding="utf-8"))
    assign = sm["assignment"]
    dev = [r for r in all_test if assign.get(r["id"]) == "dev"]
    test = [r for r in all_test if assign.get(r["id"]) != "dev"]
    assert len(dev) == sm["n_dev"] and len(test) == sm["n_test"], \
        "разбиение gold не совпало с gold_dev_split.json"
    assert not ({r.get("scenario_group") for r in dev}
                & {r.get("scenario_group") for r in test}), \
        "группы scenario_group разорваны между dev и test"
    silver = {r["id"]: int(r["is_scam"]) for r in records}
    return {"dev": dev, "test": test}, gold, gold_sub, gold_onset, silver


def assign_pids(recs, split):
    """Непрозрачные pid в перемешанном порядке (детерминированно)."""
    order = sorted(recs, key=lambda r: r["id"])
    random.Random(PID_SEED).shuffle(order)
    return {f"{split[0]}{i:04d}": r for i, r in enumerate(order, 1)}


def cmd_dump(args):
    splits, *_ = load_split(args.corpus, args.gold, args.gold_dev)
    recs = splits[args.split]
    pids = assign_pids(recs, args.split)
    d = OUT / "prompts"
    d.mkdir(parents=True, exist_ok=True)

    manifest = {pid: {"dialog_id": r["id"], "lang": r["lang"],
                      "n_turns": len(r["turns"])} for pid, r in pids.items()}
    (OUT / f"manifest_{args.split}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    system = INSTRUCTION + ("\n\n" + RULES if args.mode == "few_rules" else "")
    head = ["=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===", system]
    if args.mode in ("few", "few_rules"):
        ex = json.loads((ROOT / args.examples).read_text(encoding="utf-8"))["examples"]
        assert len(ex) == 6, "ожидалось 6 few-shot примеров"
        head.append("\n=== ПРИМЕРЫ (6, метки silver) ===")
        for e in ex:
            head.append(e["dialog"] + "\n→ " + json.dumps(e["answer"], ensure_ascii=False))
    (d / f"system_{args.mode}.txt").write_text("\n\n".join(head), encoding="utf-8")

    items = list(pids.items())
    n = 0
    for b in range(0, len(items), args.batch):
        chunk = items[b:b + args.batch]
        lines = []
        for pid, r in chunk:
            lines.append(f"### {pid} (реплик: {len(r['turns'])})")
            lines.append(dialog_text(r))
        path = d / f"{args.split}_{args.mode}_batch{b // args.batch + 1:02d}.txt"
        path.write_text("\n\n".join(lines), encoding="utf-8")
        n += 1
    print(f"режим {args.mode} | {args.split}: {len(items)} диалогов, {n} батчей → {d}")
    print(f"системный промпт: {d / f'system_{args.mode}.txt'}")
    print(f"манифест (не открывать при разметке): {OUT / f'manifest_{args.split}.json'}")


def cmd_score(args):
    splits, gold, gold_sub, gold_onset, silver = load_split(
        args.corpus, args.gold, args.gold_dev)
    recs = splits[args.split]
    pids = assign_pids(recs, args.split)

    raw = {}
    with open(args.answers, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            pid = o.pop("pid")
            raw[pid] = o
    missing = [p for p in pids if p not in raw]
    extra = [p for p in raw if p not in pids]
    if missing:
        sys.exit(f"❌ нет ответов на {len(missing)} диалогов: {missing[:5]}")
    if extra:
        sys.exit(f"❌ лишние pid: {extra[:5]}")

    ordered, answers = [], []
    for pid, r in pids.items():
        parsed, err = parse_answer(json.dumps(raw[pid], ensure_ascii=False),
                                   len(r["turns"]))
        ordered.append(r)
        answers.append({"parsed": parsed, "parse_error": err, "retries": 0,
                        "usage": {"in": 0, "out": 0}})
    langs = sorted({r["lang"] for r in ordered})
    res = evaluate(answers, ordered, gold, gold_sub, gold_onset, silver, langs)
    res["split"], res["mode"] = args.split, args.mode
    res["n_runs"] = 1
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"results_{args.split}_{args.mode}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    m = res["is_scam"]
    print(f"\n=== слой 4 in-session | {args.split} | режим {args.mode} | n={res['n_total']} ===")
    print(f"разбор: {res['parse_rate']:.4f}")
    print(f"accuracy {m['accuracy']:.4f} | precision {m['precision']:.4f} | "
          f"recall {m['recall']:.4f} | F1 {m['f1']:.4f} | κ {m['cohen_kappa']:.4f}")
    print(f"совпадение с silver: {res['silver_agreement']:.4f}")
    print(f"подтипы macro-F1 {res['subtypes']['macro_f1']:.4f} | "
          f"subset acc {res['subtypes']['subset_accuracy']:.4f}")
    for s in SUBTYPES:
        pl = res["subtypes"]["per_label"][s]
        print(f"   {s:15s} F1 {pl['f1']:.4f} (gold {pl['n_gold']}, pred {pl['n_pred']})")
    o = res["onset"]
    if o.get("n"):
        print(f"онсет (n={o['n']}): OAD {o['oad']:.4f} | exact {o['exact']:.4f} | "
              f"±1 {o['within1']:.4f} | смещение {o['bias']:+.4f}")
    for lg, v in sorted(res["per_language"].items()):
        print(f"   {lg}: F1 {v['f1']:.4f} (P {v['precision']:.3f} / R {v['recall']:.3f}, n={v['n']})")
    print(f"\nрезультаты: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--gold", default="data/gold/gold_test_v2.csv")
    ap.add_argument("--gold-dev", default="data/gold/gold_dev_split.json")
    ap.add_argument("--examples", default="runs/layer4/few_shot_examples.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump");  d.set_defaults(fn=cmd_dump)
    d.add_argument("--split", choices=["dev", "test"], required=True)
    d.add_argument("--mode", choices=["zero", "few", "few_rules"], required=True)
    d.add_argument("--batch", type=int, default=50)
    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--split", choices=["dev", "test"], required=True)
    s.add_argument("--mode", choices=["zero", "few", "few_rules"], required=True)
    s.add_argument("--answers", required=True)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
