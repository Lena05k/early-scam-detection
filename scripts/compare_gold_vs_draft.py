#!/usr/bin/env python3
"""Сравнение человеческой золотой разметки с ЧЕРНОВЫМИ LLM-метками корпуса.

Стадия 2 (верификация) в пакетном виде: CSV из HTML-пака (с РЕАЛЬНЫМИ id, см.
check_annotations.py --resolved-out) против меток в corpus_v1.jsonl.

Считает согласие человек↔LLM: is_scam (Cohen κ + accuracy, confusion),
подтипы (per-label κ), онсет (exact / ±1 / Krippendorff α interval на общем
подмножестве scam-диалогов), hard_negative. Разрезы по языку и сценарию.
Выгружает список расхождений для ручного разбора.

Примеры:
  python scripts/compare_gold_vs_draft.py \
      data/annotations/annotations_test_2026-08-06_resolved.csv \
      --corpus data/corpus/corpus_v1.jsonl \
      --report reports/gold_vs_draft_test.json \
      --disagreements reports/gold_vs_draft_test_disagreements.csv

ВАЖНО: если у метки view_mode=full, её онсет — НЕ префиксная верификация
(инвариант 5, hindsight bias). Метрики онсета считаются отдельно для seq/full.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scam_onset.iaa.harness import cohen_kappa, krippendorff_alpha_interval  # noqa: E402

SUBTYPES = ["phishing_link", "money_request", "other"]


def read_gold_csv(path):
    """CSV разметки (реальные id) → {dialog_id: dict}."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if not did:
                continue
            onset = (r.get("onset") or "").strip()
            subs = [s.strip() for s in (r.get("subtypes") or "").split("|") if s.strip()]
            out[did] = {
                "is_scam": int(r["is_scam"]),
                "subtypes": set(subs),
                "onset": int(onset) if onset else None,
                "hard_negative": int(r.get("hard_negative") or 0),
                "doubt": int(r.get("doubt") or 0),
                "note": (r.get("note") or "").strip(),
                "view_mode": (r.get("view_mode") or "").strip(),
            }
    return out


def read_corpus(path):
    """corpus jsonl → {id: dict} (только поля меток + мета)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["id"]] = {
                "is_scam": int(r.get("is_scam", 0)),
                "subtypes": set(r.get("subtypes") or []),
                "onset": r.get("onset") if r.get("onset") else None,
                "hard_negative": int(r.get("hard_negative") or 0),
                "lang": r.get("lang"),
                "scam_type": r.get("scam_type"),
                "split": r.get("split"),
                "generator": r.get("generator"),
                "n_turns": len(r.get("turns") or []),
            }
    return out


def onset_stats(pairs):
    """pairs = [(gold_onset, draft_onset)] по scam-диалогам, где обе метки заданы."""
    if not pairs:
        return {"n": 0}
    exact = sum(1 for g, d in pairs if g == d) / len(pairs)
    within1 = sum(1 for g, d in pairs if abs(g - d) <= 1) / len(pairs)
    mae = sum(abs(g - d) for g, d in pairs) / len(pairs)
    bias = sum(g - d for g, d in pairs) / len(pairs)  # >0 → человек ставит ПОЗЖЕ LLM
    alpha = krippendorff_alpha_interval([(g, d) for g, d in pairs])
    return {"n": len(pairs), "exact": round(exact, 4), "within1": round(within1, 4),
            "mae": round(mae, 3), "bias_human_minus_llm": round(bias, 3),
            "krippendorff_alpha": round(alpha, 4)}


def block(gold, draft, ids):
    """Все метрики согласия на подмножестве ids."""
    g_scam = [gold[i]["is_scam"] for i in ids]
    d_scam = [draft[i]["is_scam"] for i in ids]
    conf = collections.Counter(zip(g_scam, d_scam))  # (human, llm)
    res = {
        "n": len(ids),
        "is_scam": {
            "accuracy": round(sum(1 for a, b in zip(g_scam, d_scam) if a == b) / len(ids), 4),
            "kappa": round(cohen_kappa(g_scam, d_scam), 4),
            "confusion_human_x_llm": {f"{a}{b}": conf[(a, b)] for a in (0, 1) for b in (0, 1)},
        },
    }

    hn_ids = ids
    g_hn = [gold[i]["hard_negative"] for i in hn_ids]
    d_hn = [draft[i]["hard_negative"] for i in hn_ids]
    res["hard_negative"] = {
        "accuracy": round(sum(1 for a, b in zip(g_hn, d_hn) if a == b) / len(hn_ids), 4),
        "kappa": round(cohen_kappa(g_hn, d_hn), 4),
    }

    # подтипы — только там, где ОБА признали скамом (иначе множества тривиально пусты)
    both_scam = [i for i in ids if gold[i]["is_scam"] == 1 and draft[i]["is_scam"] == 1]
    subs = {}
    for s in SUBTYPES:
        a = [1 if s in gold[i]["subtypes"] else 0 for i in both_scam]
        b = [1 if s in draft[i]["subtypes"] else 0 for i in both_scam]
        subs[s] = {
            "n": len(both_scam),
            "kappa": round(cohen_kappa(a, b), 4) if both_scam else None,
            "human_pos": sum(a), "llm_pos": sum(b),
            "exact_match": round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4) if a else None,
        }
    ks = [subs[s]["kappa"] for s in SUBTYPES if subs[s]["kappa"] is not None]
    subs["macro_kappa"] = round(sum(ks) / len(ks), 4) if ks else None
    subs["set_exact_match"] = (
        round(sum(1 for i in both_scam if gold[i]["subtypes"] == draft[i]["subtypes"]) / len(both_scam), 4)
        if both_scam else None)
    res["subtypes"] = subs

    # онсет — на диалогах, где ОБА scam и оба онсета заданы
    onset_ids = [i for i in both_scam
                 if gold[i]["onset"] is not None and draft[i]["onset"] is not None]
    res["onset_all"] = onset_stats([(gold[i]["onset"], draft[i]["onset"]) for i in onset_ids])
    for vm in ("seq", "full"):
        sub = [i for i in onset_ids if gold[i]["view_mode"] == vm]
        res[f"onset_{vm}"] = onset_stats([(gold[i]["onset"], draft[i]["onset"]) for i in sub])
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gold_csv", help="CSV разметки с РЕАЛЬНЫМИ dialog_id")
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--report", default=None, help="куда сохранить JSON-отчёт")
    p.add_argument("--disagreements", default=None, help="куда сохранить CSV расхождений")
    args = p.parse_args()

    gold = read_gold_csv(args.gold_csv)
    draft = read_corpus(args.corpus)

    missing = sorted(set(gold) - set(draft))
    if missing:
        sys.exit(f"❌ {len(missing)} id из CSV нет в корпусе: {missing[:5]}")
    ids = sorted(gold)

    report = {
        "gold_csv": args.gold_csv,
        "corpus": args.corpus,
        "n_dialogs": len(ids),
        "view_mode_counts": dict(collections.Counter(gold[i]["view_mode"] for i in ids)),
        "n_doubt": sum(gold[i]["doubt"] for i in ids),
        "overall": block(gold, draft, ids),
        "by_lang": {}, "by_scam_type": {},
    }
    for lang in sorted({draft[i]["lang"] for i in ids}):
        sub = [i for i in ids if draft[i]["lang"] == lang]
        report["by_lang"][lang] = block(gold, draft, sub)
    for st in sorted({draft[i]["scam_type"] for i in ids}):
        sub = [i for i in ids if draft[i]["scam_type"] == st]
        if len(sub) >= 5:
            report["by_scam_type"][st] = block(gold, draft, sub)

    # ── расхождения ──
    rows = []
    for i in ids:
        g, d = gold[i], draft[i]
        kinds = []
        if g["is_scam"] != d["is_scam"]:
            kinds.append("is_scam")
        if g["is_scam"] == 1 and d["is_scam"] == 1:
            if g["subtypes"] != d["subtypes"]:
                kinds.append("subtypes")
            if g["onset"] != d["onset"]:
                kinds.append("onset")
        if g["hard_negative"] != d["hard_negative"]:
            kinds.append("hard_negative")
        if kinds:
            rows.append({
                "dialog_id": i, "lang": d["lang"], "scam_type": d["scam_type"],
                "n_turns": d["n_turns"], "generator": d["generator"],
                "kinds": "|".join(kinds),
                "human_is_scam": g["is_scam"], "llm_is_scam": d["is_scam"],
                "human_subtypes": "|".join(sorted(g["subtypes"])),
                "llm_subtypes": "|".join(sorted(d["subtypes"])),
                "human_onset": g["onset"] if g["onset"] is not None else "",
                "llm_onset": d["onset"] if d["onset"] is not None else "",
                "onset_delta": (g["onset"] - d["onset"]) if (g["onset"] is not None and d["onset"] is not None) else "",
                "human_hard_negative": g["hard_negative"], "llm_hard_negative": d["hard_negative"],
                "doubt": g["doubt"], "view_mode": g["view_mode"], "note": g["note"],
            })
    report["n_disagreements"] = len(rows)
    report["disagreement_kinds"] = dict(collections.Counter(
        k for r in rows for k in r["kinds"].split("|")))

    # ── печать ──
    o = report["overall"]
    print(f"диалогов: {report['n_dialogs']}  "
          f"(seq {report['view_mode_counts'].get('seq', 0)} / "
          f"full {report['view_mode_counts'].get('full', 0)}), "
          f"«сомневаюсь»: {report['n_doubt']}")
    print(f"\nis_scam:  acc={o['is_scam']['accuracy']}  κ={o['is_scam']['kappa']}")
    c = o["is_scam"]["confusion_human_x_llm"]
    print(f"  человек1/LLM0 (LLM пропустил скам): {c['10']}   "
          f"человек0/LLM1 (LLM ложная тревога): {c['01']}")
    print(f"hard_negative: acc={o['hard_negative']['accuracy']}  κ={o['hard_negative']['kappa']}")
    s = o["subtypes"]
    print(f"\nподтипы (оба scam, n={s['phishing_link']['n']}): macro-κ={s['macro_kappa']}  "
          f"полное совпадение множества={s['set_exact_match']}")
    for st in SUBTYPES:
        print(f"  {st:15s} κ={s[st]['kappa']}  человек+{s[st]['human_pos']} / LLM+{s[st]['llm_pos']}")
    for key, title in (("onset_all", "онсет (все)"), ("onset_seq", "онсет (seq — префиксные)"),
                       ("onset_full", "онсет (full — hindsight ⚠)")):
        st = o[key]
        if st["n"]:
            print(f"\n{title}: n={st['n']}  exact={st['exact']}  ±1={st['within1']}  "
                  f"MAE={st['mae']}  α={st['krippendorff_alpha']}  "
                  f"сдвиг(чел−LLM)={st['bias_human_minus_llm']}")
        else:
            print(f"\n{title}: нет данных")
    print("\nпо языкам (is_scam κ / онсет exact / онсет α):")
    for lang, b in report["by_lang"].items():
        print(f"  {lang}: n={b['n']}  κ={b['is_scam']['kappa']}  "
              f"exact={b['onset_all'].get('exact')}  α={b['onset_all'].get('krippendorff_alpha')}")
    print(f"\nрасхождений: {report['n_disagreements']} из {report['n_dialogs']}  "
          f"{report['disagreement_kinds']}")

    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {args.report}")
    if args.disagreements and rows:
        pathlib.Path(args.disagreements).parent.mkdir(parents=True, exist_ok=True)
        with open(args.disagreements, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["kinds"], r["dialog_id"])))
        print(f"расхождения: {args.disagreements}")


if __name__ == "__main__":
    main()
