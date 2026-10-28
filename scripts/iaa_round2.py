#!/usr/bin/env python3
"""Согласие двух аннотаторов на выборке IAA раунда 2 (100 диалогов gold-test v2).

Вход:
  --annotations  CSV второго аннотатора из reports/iaa_round2.html
                 (колонка dialog_id держит НЕПРОЗРАЧНЫЙ pid; шапка как у пака раунда 1)
  --sample       data/gold/iaa_round2_sample.json — pid -> dialog_id + страты
  --gold         data/gold/gold_test_v2.csv — метки первого аннотатора (Елена)

Считает: Fleiss κ и Cohen κ по is_scam, per-label и macro-κ по подтипам,
Krippendorff α (interval) по позиции онсета, exact/±1, разбивки по языкам и по
бинам t*, полный список расхождений со сценарием.

Неполный файл — штатная ситуация: считается по размеченным, недостающие
показываются отдельно (сколько и в каких стратах). Ноль строк — не падение,
а сообщение.

⚠ Сравнение с калибровкой (κ 0.857, α 0.708) — ОРИЕНТИР, не тождество: там было
ТРИ аннотатора на 24 синтетических диалогах, здесь ДВА на 100 диалогах корпуса.
При двух раторах Fleiss κ вырождается в Scott's π, поэтому печатаются обе меры.

Gold-файлы только читаются.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.iaa.harness import (  # noqa: E402
    cohen_kappa, fleiss_kappa, krippendorff_alpha_interval,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "gold" / "iaa_round2_sample.json"
GOLD = ROOT / "data" / "gold" / "gold_test_v2.csv"
OUT_DIR = ROOT / "runs" / "iaa_round2"

SUBTYPES = ("phishing_link", "money_request", "other")
BINS = ("1", "2", "3-4", "5+")
# Калибровка на 24 синтетических диалогах, 3 аннотатора (CLAUDE.md «Калибровка (итог)»).
PILOT = {"is_scam_fleiss_kappa": 0.857, "onset_alpha": 0.708, "subtypes_macro_kappa": 0.79}


def nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def read_labels_csv(path, key="dialog_id"):
    """CSV разметки -> {key: {...}}. Строки с пустым is_scam считаются неразмеченными."""
    out, blank = {}, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise SystemExit(f"{path}: нет колонки {key} (шапка: {reader.fieldnames})")
        for r in reader:
            k = (r.get(key) or "").strip()
            if not k:
                continue
            if (r.get("is_scam") or "").strip() == "":
                blank += 1
                continue
            raw_sub = (r.get("subtypes") or "").strip()
            subs = tuple(s.strip() for s in raw_sub.split("|") if s.strip()) if raw_sub else ()
            unknown = set(subs) - set(SUBTYPES)
            if unknown:
                raise SystemExit(f"{path}: неизвестные подтипы {sorted(unknown)} у {k}")
            onset = (r.get("onset") or "").strip()
            out[k] = {
                "is_scam": int(r["is_scam"]),
                "subtypes": tuple(sorted(subs)),
                "onset": int(onset) if onset else None,
                "note": (r.get("note") or "").strip(),
                "view_mode": (r.get("view_mode") or "").strip(),
                "doubt": (r.get("doubt") or "").strip(),
            }
    return out, blank


def consistency_problems(rec):
    p = []
    if rec["is_scam"] == 1 and not rec["subtypes"]:
        p.append("scam без подтипа")
    if rec["is_scam"] == 0 and rec["subtypes"]:
        p.append("benign с подтипом")
    if rec["is_scam"] == 0 and rec["onset"] is not None:
        p.append("benign с онсетом")
    return p


def per_label_kappa(pairs):
    """pairs: [(subs_a, subs_b)] -> {подтип: κ}, macro-κ по непустым."""
    out = {}
    for lab in SUBTYPES:
        a = [1 if lab in x else 0 for x, _ in pairs]
        b = [1 if lab in y else 0 for _, y in pairs]
        out[lab] = nan_to_none(cohen_kappa(a, b)) if a else None
    vals = [v for v in out.values() if v is not None]
    return out, (sum(vals) / len(vals) if vals else None)


def onset_block(units):
    """units: [(onset_a, onset_b)] по диалогам, где ОБА поставили скам и онсет."""
    if not units:
        return {"n": 0, "alpha": None, "exact": None, "within1": None,
                "mae": None, "mean_diff": None}
    alpha = nan_to_none(krippendorff_alpha_interval([[a, b] for a, b in units]))
    diffs = [a - b for a, b in units]
    return {
        "n": len(units),
        "alpha": alpha,
        "exact": sum(1 for d in diffs if d == 0) / len(diffs),
        "within1": sum(1 for d in diffs if abs(d) <= 1) / len(diffs),
        "mae": sum(abs(d) for d in diffs) / len(diffs),
        # знак: >0 — первый аннотатор ставит онсет ПОЗЖЕ второго
        "mean_diff": sum(diffs) / len(diffs),
    }


def scam_block(pairs):
    """pairs: [(is_scam_a, is_scam_b)] -> согласие двумя мерами.

    ⚠ Если у одного из аннотаторов метка внутри среза ПОСТОЯННА (так почти всегда в
    разрезе по бинам t*: бин 1 — это по определению только scam), κ вырождается:
    P_e схлопывается к P_o и κ выдаёт 0.000 при фактическом согласии 0.94, либо 1.000
    на константном согласии. Это свойство κ, а не измерение. В таких срезах κ помечается
    как неинтерпретируемая, а читать нужно простое согласие.
    """
    if not pairs:
        return {"n": 0, "agreement": None, "cohen_kappa": None, "fleiss_kappa": None,
                "degenerate_marginal": None}
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    degenerate = len(set(a)) == 1 or len(set(b)) == 1
    rows = [Counter({0: 0, 1: 0}) for _ in pairs]
    for row, (x, y) in zip(rows, pairs):
        row[x] += 1
        row[y] += 1
    return {
        "n": len(pairs),
        "agreement": sum(1 for x, y in pairs if x == y) / len(pairs),
        "cohen_kappa": None if degenerate else nan_to_none(cohen_kappa(a, b)),
        # при двух раторах Fleiss κ == Scott's π; отличается от Cohen κ трактовкой P_e
        "fleiss_kappa": None if degenerate else nan_to_none(fleiss_kappa(rows, [0, 1])),
        "degenerate_marginal": degenerate,
    }


def analyse(items, a1, a2):
    """items: элементы выборки; a1/a2: {dialog_id: rec}. Возвращает полный отчёт."""
    common = [it for it in items if it["dialog_id"] in a1 and it["dialog_id"] in a2]
    missing = [it for it in items if it["dialog_id"] not in a2]

    pairs_scam, pairs_sub, onset_units = [], [], []
    by_lang = defaultdict(lambda: {"scam": [], "onset": []})
    by_bin = defaultdict(lambda: {"scam": [], "onset": []})
    disagreements = []

    for it in common:
        did = it["dialog_id"]
        r1, r2 = a1[did], a2[did]
        pairs_scam.append((r1["is_scam"], r2["is_scam"]))
        pairs_sub.append((r1["subtypes"], r2["subtypes"]))
        by_lang[it["lang"]]["scam"].append((r1["is_scam"], r2["is_scam"]))
        b = it["bin"] or "benign"
        by_bin[b]["scam"].append((r1["is_scam"], r2["is_scam"]))

        both_scam_onset = (r1["is_scam"] == 1 and r2["is_scam"] == 1
                           and r1["onset"] is not None and r2["onset"] is not None)
        if both_scam_onset:
            onset_units.append((r1["onset"], r2["onset"]))
            by_lang[it["lang"]]["onset"].append((r1["onset"], r2["onset"]))
            by_bin[b]["onset"].append((r1["onset"], r2["onset"]))

        tags = []
        if r1["is_scam"] != r2["is_scam"]:
            tags.append("is_scam")
        if r1["subtypes"] != r2["subtypes"]:
            tags.append("subtypes")
        if r1["onset"] != r2["onset"]:
            tags.append("onset")
        if tags:
            disagreements.append({
                "pid": it["pid"], "dialog_id": did, "lang": it["lang"],
                "scenario": it["scenario"], "hard_negative": it["hard_negative"],
                "onset_bin": it["bin"], "n_turns": it["n_turns"], "what": tags,
                "a1": {"is_scam": r1["is_scam"], "subtypes": list(r1["subtypes"]),
                       "onset": r1["onset"]},
                "a2": {"is_scam": r2["is_scam"], "subtypes": list(r2["subtypes"]),
                       "onset": r2["onset"]},
                "a2_note": r2["note"], "a2_doubt": r2["doubt"],
            })

    sub_kappa, sub_macro = per_label_kappa(pairs_sub)
    exact_set = (sum(1 for x, y in pairs_sub if x == y) / len(pairs_sub)) if pairs_sub else None

    problems = {did: consistency_problems(a2[did]) for did in a2}
    problems = {k: v for k, v in problems.items() if v}

    return {
        "coverage": {
            "sample_size": len(items),
            "annotated_by_second": len(a2),
            "usable_pairs": len(common),
            "missing": len(missing),
            "missing_pids": [it["pid"] for it in missing],
            "missing_strata": dict(Counter(
                f"{it['lang']}/{'scam' if it['is_scam'] else 'benign'}/{it['bin'] or '-'}"
                for it in missing)),
            "complete": len(missing) == 0,
        },
        "is_scam": scam_block(pairs_scam),
        "subtypes": {"per_label_kappa": sub_kappa, "macro_kappa": sub_macro,
                     "exact_set_match": exact_set, "n": len(pairs_sub)},
        "onset": onset_block(onset_units),
        "by_language": {
            lang: {"is_scam": scam_block(v["scam"]), "onset": onset_block(v["onset"])}
            for lang, v in sorted(by_lang.items())
        },
        "by_onset_bin": {
            b: {"is_scam": scam_block(v["scam"]), "onset": onset_block(v["onset"])}
            for b, v in sorted(by_bin.items(), key=lambda kv: str(kv[0]))
        },
        "second_annotator_view_mode": dict(Counter(r["view_mode"] or "?" for r in a2.values())),
        "second_annotator_consistency_problems": problems,
        "disagreements": sorted(disagreements,
                                key=lambda d: (d["lang"], d["scenario"], d["pid"])),
        "pilot_reference": PILOT,
    }


def print_report(res, a1_modes):
    cov = res["coverage"]
    line = "=" * 70
    print(line)
    print(f"IAA раунд 2 | размечено вторым аннотатором: {cov['annotated_by_second']} "
          f"из {cov['sample_size']} | пар для сравнения: {cov['usable_pairs']}")
    print(line)
    if not cov["complete"]:
        print(f"\n⚠ НЕПОЛНЫЙ ФАЙЛ: не размечено {cov['missing']}. "
              f"Все числа ниже посчитаны по {cov['usable_pairs']} диалогам.")
        if cov["missing_strata"]:
            print("  недостающие страты (язык/метка/бин t*):")
            for k, v in sorted(cov["missing_strata"].items()):
                print(f"    {k:<24} {v}")
    if cov["usable_pairs"] == 0:
        print("\nСравнивать нечего: ни одной размеченной строки. "
              "Проверь, что CSV выгружен из reports/iaa_round2.html.")
        return

    s = res["is_scam"]
    print(f"\n[1] is_scam (n={s['n']})")
    print(f"    простое согласие      = {fmt(s['agreement'])}")
    print(f"    Fleiss κ (= Scott's π) = {fmt(s['fleiss_kappa'])}"
          f"   ← сравнивать с калибровкой {PILOT['is_scam_fleiss_kappa']:.3f}")
    print(f"    Cohen κ                = {fmt(s['cohen_kappa'])}")

    sub = res["subtypes"]
    print(f"\n[2] подтипы (n={sub['n']})")
    for lab in SUBTYPES:
        print(f"    {lab:<15} κ = {fmt(sub['per_label_kappa'][lab])}")
    print(f"    macro-κ         = {fmt(sub['macro_kappa'])}"
          f"   ← калибровка {PILOT['subtypes_macro_kappa']:.2f}")
    print(f"    множество совпало полностью = {fmt(sub['exact_set_match'])}")

    o = res["onset"]
    print(f"\n[3] онсет (оба поставили скам и онсет: n={o['n']})")
    print(f"    Krippendorff α (interval) = {fmt(o['alpha'])}"
          f"   ← калибровка {PILOT['onset_alpha']:.3f}")
    print(f"    exact = {fmt(o['exact'])} | ±1 = {fmt(o['within1'])} | "
          f"MAE = {fmt(o['mae'], 2)} реплик")
    print(f"    среднее (аннотатор 1 − аннотатор 2) = {fmt(o['mean_diff'], 2)} "
          f"(>0 — первый ставит онсет позже)")

    print("\n[4] по языкам")
    print(f"    {'язык':<6} {'n':>4} {'согласие':>9} {'κ is_scam':>10} "
          f"{'n онсет':>8} {'α':>7} {'exact':>7}")
    for lang, v in res["by_language"].items():
        s_ = v["is_scam"]
        kap = fmt(s_["cohen_kappa"]) + ("" if not s_["degenerate_marginal"] else "*")
        print(f"    {lang:<6} {s_['n']:>4} {fmt(s_['agreement']):>9} {kap:>10} "
              f"{v['onset']['n']:>8} {fmt(v['onset']['alpha']):>7} "
              f"{fmt(v['onset']['exact']):>7}")

    print("\n[5] по бинам t* (бин — по метке первого аннотатора, как при отборе выборки)")
    print(f"    {'бин':<8} {'n':>4} {'согласие':>9} {'κ is_scam':>10} "
          f"{'n онсет':>8} {'exact':>7} {'MAE':>6}")
    degen = False
    for b, v in res["by_onset_bin"].items():
        s_ = v["is_scam"]
        degen = degen or bool(s_["degenerate_marginal"])
        kap = fmt(s_["cohen_kappa"]) + ("" if not s_["degenerate_marginal"] else "*")
        print(f"    {b:<8} {s_['n']:>4} {fmt(s_['agreement']):>9} {kap:>10} "
              f"{v['onset']['n']:>8} {fmt(v['onset']['exact']):>7} "
              f"{fmt(v['onset']['mae'], 2):>6}")
    if degen:
        print("    * κ неинтерпретируема: внутри среза метка одного из аннотаторов")
        print("      постоянна (бин t* по определению состоит из scam) — читать согласие.")

    modes = res["second_annotator_view_mode"]
    print(f"\n[6] режим просмотра: второй аннотатор {modes} | первый {a1_modes}")
    if modes.get("seq", 0) and modes.get("full", 0):
        print("    ⚠ режимы смешаны — онсеты неоднородны по условиям (инвариант 5)")
    if a1_modes.get("full", 0) and modes.get("seq", 0) > modes.get("full", 0):
        print("    ⚠ первый аннотатор работал в основном 'весь текст', второй — 'по репликам': "
              "расхождение по онсету смешивает эффект аннотатора и эффект режима")

    probs = res["second_annotator_consistency_problems"]
    if probs:
        print(f"\n[7] ⚠ нарушения консистентности v1.2 у второго аннотатора: {len(probs)}")
        for k, v in sorted(probs.items())[:10]:
            print(f"    {k}: {'; '.join(v)}")
    else:
        print("\n[7] нарушений консистентности v1.2 у второго аннотатора нет")

    dis = res["disagreements"]
    print(f"\n[8] расхождения: {len(dis)} из {cov['usable_pairs']}")
    if dis:
        print(f"    {'pid':<8} {'язык':<4} {'сценарий':<22} {'что':<22} "
              f"{'аннотатор 1':<22} аннотатор 2")
        for d in dis:
            f1 = f"scam={d['a1']['is_scam']},t*={d['a1']['onset']}"
            f2 = f"scam={d['a2']['is_scam']},t*={d['a2']['onset']}"
            hn = "*" if d["hard_negative"] else " "
            print(f"    {d['pid']:<8} {d['lang']:<4} {d['scenario'][:21]:<22}{hn}"
                  f"{','.join(d['what']):<21} {f1:<22} {f2}")
        print("    (* — hard_negative корпуса)")
        by_scen = Counter(d["scenario"] for d in dis)
        print("\n    расхождения по сценариям: "
              + ", ".join(f"{k}={v}" for k, v in by_scen.most_common()))
        hn_share = sum(d["hard_negative"] for d in dis)
        print(f"    из них hard_negative: {hn_share}")

    print("\n[9] сверка с калибровкой (3 аннотатора, 24 synth-диалога)")
    print(f"    {'метрика':<24} {'раунд 2':>9} {'калибровка':>11} {'дельта':>8}")
    for name, cur, ref in (
            ("is_scam Fleiss κ", res["is_scam"]["fleiss_kappa"], PILOT["is_scam_fleiss_kappa"]),
            ("подтипы macro-κ", res["subtypes"]["macro_kappa"], PILOT["subtypes_macro_kappa"]),
            ("онсет α", res["onset"]["alpha"], PILOT["onset_alpha"])):
        delta = fmt(cur - ref) if cur is not None else "n/a"
        print(f"    {name:<24} {fmt(cur):>9} {ref:>11.3f} {delta:>8}")
    print("    ⚠ ориентир, не тождество: там 3 аннотатора и 24 синтетических диалога,")
    print("      здесь 2 аннотатора и диалоги реального корпуса.")
    print(line)


def synth_second_annotator(items, a1, seed=20260812):
    """Синтетический второй аннотатор для смоук-прогона (--selfcheck).

    НЕ данные: детерминированное возмущение меток первого аннотатора, нужное только
    чтобы прогнать все ветки отчёта на непустом входе.
    """
    import random
    rnd = random.Random(seed)
    out = {}
    for i, it in enumerate(items):
        did = it["dialog_id"]
        if did not in a1 or i % 7 == 3:            # часть «не размечена» — ветка неполноты
            continue
        r = dict(a1[did])
        r["subtypes"] = tuple(r["subtypes"])
        if rnd.random() < 0.10:                     # расхождение по is_scam
            r["is_scam"] = 1 - r["is_scam"]
            if r["is_scam"] == 0:
                r["subtypes"], r["onset"] = (), None
            elif not r["subtypes"]:
                r["subtypes"] = ("other",)
                atk_first = 1
                r["onset"] = atk_first
        if r["is_scam"] == 1 and r["onset"] is not None and rnd.random() < 0.30:
            r["onset"] = max(1, min(it["n_turns"], r["onset"] + rnd.choice([-1, 1, 2])))
        if r["is_scam"] == 1 and rnd.random() < 0.20:
            r["subtypes"] = tuple(sorted(set(r["subtypes"]) ^ {"other"}))
            if not r["subtypes"]:
                r["subtypes"] = ("money_request",)
        r["note"] = ""
        r["view_mode"] = "full"
        r["doubt"] = "0"
        out[did] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="IAA раунд 2: согласие двух аннотаторов")
    ap.add_argument("--annotations", default=None,
                    help="CSV второго аннотатора (из reports/iaa_round2.html)")
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", default=str(OUT_DIR / "results.json"))
    ap.add_argument("--selfcheck", action="store_true",
                    help="смоук-прогон на СИНТЕТИЧЕСКОМ втором аннотаторе (не данные)")
    a = ap.parse_args()

    if not a.annotations and not a.selfcheck:
        ap.error("нужен --annotations (или --selfcheck для проверки самого скрипта)")

    sample = json.loads(pathlib.Path(a.sample).read_text(encoding="utf-8"))
    items = sample["items"]
    pid_of = {it["dialog_id"]: it["pid"] for it in items}
    did_of = {it["pid"]: it["dialog_id"] for it in items}

    gold_all, _ = read_labels_csv(a.gold)
    a1 = {did: gold_all[did] for did in pid_of if did in gold_all}
    if len(a1) != len(items):
        raise SystemExit(f"в {a.gold} нашлось {len(a1)} из {len(items)} диалогов выборки")
    a1_modes = dict(Counter(r["view_mode"] or "?" for r in a1.values()))

    if a.selfcheck:
        print("СМОУК-ПРОГОН: второй аннотатор СИНТЕТИЧЕСКИЙ (возмущение меток первого).")
        print("Это проверка кода, а НЕ измерение согласия. Реальные числа появятся,")
        print("когда придёт CSV живого человека.\n")
        a2 = synth_second_annotator(items, a1)
        blank = 0
    else:
        raw, blank = read_labels_csv(a.annotations)
        unknown = [k for k in raw if k not in did_of]
        if unknown:
            raise SystemExit(
                f"{a.annotations}: {len(unknown)} pid не из этой выборки, "
                f"первые: {unknown[:5]}. Файл от другого пака?")
        a2 = {did_of[pid]: rec for pid, rec in raw.items()}

    if blank:
        print(f"⚠ строк с пустым is_scam (не размечено): {blank}\n")

    res = analyse(items, a1, a2)
    res["inputs"] = {
        "sample": str(pathlib.Path(a.sample).relative_to(ROOT)),
        "gold_annotator1": str(pathlib.Path(a.gold).relative_to(ROOT)),
        "annotations_annotator2": (None if a.selfcheck
                                   else str(pathlib.Path(a.annotations).resolve())),
        "synthetic_second_annotator": bool(a.selfcheck),
        "annotator1_view_mode": a1_modes,
        "blank_rows": blank,
    }
    print_report(res, a1_modes)

    out = pathlib.Path(a.out)
    if a.selfcheck:
        out = out.with_name("selfcheck_" + out.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nрезультаты: {out.relative_to(ROOT)}")
    if a.selfcheck:
        print("⚠ это смоук-файл на синтетике — в статью и CLAUDE.md не переносить.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
