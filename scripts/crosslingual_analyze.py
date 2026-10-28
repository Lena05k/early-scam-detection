#!/usr/bin/env python3
"""Разбор кросс-язычного переноса: матрица, зазоры, few-shot, письменности.

Работает с любым набором готовых моделей: слои 1 и 2 читаются из
`runs/crosslingual/layers12.json`, файнтюн — из `results_raw.json`, принесённого
с Kaggle. Отсутствующая модель просто пропускается, поэтому разбор можно гонять
до и после прогона на GPU.

Отбор точки сетки (у слоёв 1–2) и эпохи (у xlm-r) делается ЗДЕСЬ, постфактум:
  dev_source — честный zero-shot: целевой язык не участвует ни в чём (основной);
  val_source — контроль по silver-val языков-источников (примеров больше).
Тест всегда полный gold-test v2 (303), метрики отдельно по каждому языку.

  python scripts/crosslingual_analyze.py
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import LANGS, metrics, mstd  # noqa: E402

# dev_source — честный zero-shot (основной). Остальные два — ДИАГНОСТИКА с утечкой
# целевого языка: они показывают, сколько в аномалии от переноса, а сколько от шума
# отбора, и в основной результат не идут.
SELECTIONS = ["dev_source", "val_source", "dev_full", "oracle_test"]
LEAKY = {"dev_full", "oracle_test"}
SCRIPT_OF = {"en": "латиница", "ru": "кириллица", "zh": "иероглифы"}
KS = [0, 10, 50, 100]


# ──────────────────── приведение моделей к общему виду ────────────────────
def from_layers12(path):
    """layers12.json → {модель: {конфигурация: [по сидам: {grid: [...]}]}}."""
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for model, cfgs in d["models"].items():
        out[model] = {}
        for name, c in cfgs.items():
            runs = []
            for s in c["per_seed"]:
                pts = [{"key": g["params"],
                        "dev_source": g.get("dev_source_f1"),
                        "val_source": g.get("val_source_f1"),
                        "dev_full": g.get("dev_full_f1"),
                        "oracle_test": g["test"]["overall"]["f1"],
                        "test": g["test"]} for g in s["grid"]]
                runs.append({"seed": s["seed"], "points": pts})
            out[model][name] = {"config": c["config"], "composition": c["composition"],
                                "selection_sizes": c["selection_sizes"], "runs": runs}
    return out


def from_xlmr(path, test_langs=None, manifest="runs/crosslingual/train_manifest.json"):
    """results_raw.json с Kaggle → тот же вид. Точка сетки здесь — эпоха.

    Состав обучающей выборки по языкам берётся из манифеста: без него матрица не
    знает, какие языки были источниками, и помечает zero-shot КАЖДУЮ клетку,
    включая in-language.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    langs = np.array(test_langs or d["test_langs"])
    mf = pathlib.Path(manifest)
    comps = {}
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))["configs"]
        comps = {k: v["composition"] for k, v in m.items()}
    else:
        print(f"⚠ манифеста {mf} нет — состав по языкам восстановлен из имён конфигураций")
    out = {"xlmr": {}}
    for r in d["runs"]:
        pts = []
        for i in range(r["epochs_run"]):
            dev, val, te = (r["snapshots"][k][i] for k in ("dev_sel", "val_sel", "gold_test"))
            yt, yp = np.array(te["y_true"]), np.array(te["y_pred"])
            test = {"overall": metrics(yt, yp) | {"n": len(yt)}}
            for lg in LANGS:
                idx = np.flatnonzero(langs == lg)
                test[lg] = metrics(yt[idx], yp[idx]) | {"n": len(idx)}
            pts.append({
                "key": {"epoch": i + 1},
                "dev_source": metrics(np.array(dev["y_true"]),
                                      np.array(dev["y_pred"]))["f1"],
                "val_source": metrics(np.array(val["y_true"]),
                                      np.array(val["y_pred"]))["f1"],
                # полного gold-dev у xlm-r нет: снимки снимались только по языкам-
                # источникам, а чекпоинты не сохранялись. Вместо него — оракул по
                # тесту: верхняя граница «что дал бы идеальный отбор эпохи».
                "dev_full": None,
                "oracle_test": test["overall"]["f1"],
                "test": test,
            })
        comp = comps.get(r["config"])
        if comp is None:                     # запасной путь: языки из имени конфигурации
            name = r["config"]
            srcs = [lg for lg in LANGS if lg in name.split("_")[-1].split("+")] or \
                   [lg for lg in LANGS if lg in name]
            comp = {"n_total": r["n_train"], "by_lang": {lg: (1 if lg in srcs else 0)
                                                         for lg in LANGS}}
        e = out["xlmr"].setdefault(r["config"], {"config": None, "composition": comp,
                                                 "selection_sizes": r["sizes"], "runs": []})
        e["runs"].append({"seed": r["seed"], "points": pts})
    return out


# ─────────────────────────── отбор и агрегация ───────────────────────────
def select(entry, how):
    """Точка сетки по критерию `how`, отдельно на каждом сиде → метрики теста."""
    picked = []
    for run in entry["runs"]:
        pts = [p for p in run["points"] if p[how] is not None]
        if not pts:
            return None
        best = max(enumerate(pts), key=lambda kv: (kv[1][how], -kv[0]))[1]
        picked.append({"seed": run["seed"], "key": best["key"],
                       "criterion_value": best[how], "test": best["test"]})
    agg = {"chosen": [p["key"] for p in picked]}
    for scope in ["overall"] + LANGS:
        agg[scope] = {m: mstd([p["test"][scope][m] for p in picked])
                      for m in ("f1", "recall", "precision", "accuracy", "cohen_kappa")}
        # сколько сидов схлопнулось в вырожденный предсказатель: recall≈0 («всё benign»)
        # или recall≈1 при низкой precision («всё scam»). Среднее по таким сидам
        # смешивает несопоставимые режимы, поэтому их надо считать отдельно.
        rec = agg[scope]["recall"]["values"]
        pre = agg[scope]["precision"]["values"]
        agg[scope]["n_collapsed_benign"] = sum(1 for r in rec if r <= 0.10)
        agg[scope]["n_collapsed_scam"] = sum(1 for r, p in zip(rec, pre)
                                             if r >= 0.95 and p <= 0.70)
    return agg


def analyse_model(name, cfgs):
    """Все конфигурации модели → отобранные метрики по обоим критериям."""
    res = {}
    for cname, entry in cfgs.items():
        res[cname] = {"composition": entry.get("composition", {}),
                      "selection_sizes": entry.get("selection_sizes", {}),
                      "by_selection": {}}
        for how in SELECTIONS:
            sel = select(entry, how)
            if sel:
                res[cname]["by_selection"][how] = sel
    return res


# ─────────────────────────── производные величины ───────────────────────────
def transfer_matrix(res, how):
    """«Обучали на X → тест на Y»: F1 и recall в каждой клетке."""
    rows = [f"single_{lg}" for lg in LANGS] + \
           [f"loo_{a}+{b}" for a, b in itertools.combinations(LANGS, 2)] + \
           [f"loo_{a}+{b}_n464" for a, b in itertools.combinations(LANGS, 2)] + \
           ["all3", "all3_n464"]
    out = {}
    for r in rows:
        if r not in res or how not in res[r]["by_selection"]:
            continue
        sel = res[r]["by_selection"][how]
        out[r] = {lg: {"f1": sel[lg]["f1"], "recall": sel[lg]["recall"],
                       "n_collapsed_benign": sel[lg]["n_collapsed_benign"],
                       "n_collapsed_scam": sel[lg]["n_collapsed_scam"]} for lg in LANGS}
        out[r]["_n_train"] = res[r]["composition"].get("n_total")
        out[r]["_sources"] = [lg for lg in LANGS
                              if res[r]["composition"].get("by_lang", {}).get(lg, 0) > 0]
    return out


def transfer_gaps(res, how, stat="median"):
    """Зазор = F1(zero-shot на L) − F1(in-language на L), полный и выровненный.

    По умолчанию считается по МЕДИАНЕ сидов, а не по среднему: у части
    кросс-язычных конфигураций один сид из трёх схлопывается в вырожденный
    предсказатель (recall≈0), и среднее описывает режим, которого не было ни в
    одном реальном прогоне. Среднее сохраняется рядом как `gap_mean`.
    """
    gaps = {"single_source": {}, "loo_full": {}, "loo_matched": {}}
    for tgt in LANGS:
        base_key = f"single_{tgt}"
        if base_key not in res or how not in res[base_key]["by_selection"]:
            continue
        base = res[base_key]["by_selection"][how][tgt]["f1"][stat]
        base_mean = res[base_key]["by_selection"][how][tgt]["f1"]["mean"]
        # zero-shot из одного источника
        for src in LANGS:
            if src == tgt:
                continue
            k = f"single_{src}"
            if k in res and how in res[k]["by_selection"]:
                cell = res[k]["by_selection"][how][tgt]
                z, zm = cell["f1"][stat], cell["f1"]["mean"]
                gaps["single_source"].setdefault(tgt, {})[src] = {
                    "zero_shot_f1": z, "in_language_f1": base, "gap": z - base,
                    "zero_shot_f1_mean": zm, "gap_mean": zm - base_mean,
                    "n_collapsed": cell["n_collapsed_benign"] + cell["n_collapsed_scam"],
                    "script_pair": " ↔ ".join(sorted({SCRIPT_OF[src], SCRIPT_OF[tgt]}))}
        # zero-shot из двух источников (leave-one-out), полный и выровненный
        others = [lg for lg in LANGS if lg != tgt]
        pair = "+".join(sorted(others, key=LANGS.index))
        for tag, key in (("loo_full", f"loo_{pair}"), ("loo_matched", f"loo_{pair}_n464")):
            if key in res and how in res[key]["by_selection"]:
                cell = res[key]["by_selection"][how][tgt]
                z, zm = cell["f1"][stat], cell["f1"]["mean"]
                gaps[tag][tgt] = {"zero_shot_f1": z, "in_language_f1": base,
                                  "gap": z - base, "gap_mean": zm - base_mean,
                                  "n_collapsed": cell["n_collapsed_benign"]
                                  + cell["n_collapsed_scam"],
                                  "sources": others,
                                  "n_train": res[key]["composition"].get("n_total")}
    return gaps


def fewshot_curves(res, how):
    """F1 на целевом языке как функция k, плюс уровень in-language."""
    out = {}
    for src in LANGS:
        for tgt in LANGS:
            if src == tgt:
                continue
            pts = {}
            for k in KS:
                key = f"single_{src}" if k == 0 else f"fewshot_{src}->{tgt}_k{k}"
                if key in res and how in res[key]["by_selection"]:
                    pts[k] = res[key]["by_selection"][how][tgt]["f1"]["median"]
            base_key, all_key = f"single_{tgt}", "all3"
            ref = (res[base_key]["by_selection"][how][tgt]["f1"]["median"]
                   if base_key in res and how in res[base_key]["by_selection"] else None)
            ref_all = (res[all_key]["by_selection"][how][tgt]["f1"]["median"]
                       if all_key in res and how in res[all_key]["by_selection"] else None)
            reached = None
            if ref is not None:
                for k in sorted(pts):
                    if pts[k] >= ref:
                        reached = k
                        break
            out[f"{src}->{tgt}"] = {"points": pts, "in_language": ref, "all3": ref_all,
                                    "k_to_reach_in_language": reached,
                                    "script_pair": " ↔ ".join(
                                        sorted({SCRIPT_OF[src], SCRIPT_OF[tgt]}))}
    return out


def mark_degenerate(res, how, const_f1):
    """Клетки, неотличимые от константы «всегда scam».

    При recall≈1 и низкой precision модель просто объявляет скамом почти всё. F1 при
    этом остаётся приличным (доля скама в тесте 0.62), и клетка читается как удачный
    перенос, хотя переносом не является. Опорная величина — F1 тривиальной константы
    из слоя 0: en 0.7758, ru 0.7607, zh 0.7607.
    """
    flags = {}
    for cname, entry in res.items():
        if how not in entry["by_selection"]:
            continue
        sel = entry["by_selection"][how]
        for lg in LANGS:
            rec = sel[lg]["recall"]["mean"]
            pre = sel[lg]["precision"]["mean"]
            f1 = sel[lg]["f1"]["mean"]
            if rec >= 0.95 and pre <= 0.70:
                flags[f"{cname}|{lg}"] = {
                    "f1": f1, "recall": rec, "precision": pre,
                    "constant_f1": const_f1.get(lg),
                    "above_constant": f1 - const_f1.get(lg, 0.0),
                    "note": "recall≈1 при низкой precision — почти константа «всегда scam»"}
    return flags


def asymmetry(gaps):
    """Насколько зазор зависит от НАПРАВЛЕНИЯ переноса, а не от пары языков.

    Дистанция письменностей симметрична, поэтому если |gap(A→B) − gap(B→A)| велико,
    гипотезу графической дистанции этим одним фактом уже нельзя считать объяснением.
    """
    out = {}
    ss = gaps["single_source"]
    for a, b in itertools.combinations(LANGS, 2):
        try:
            ab = ss[b][a]["gap"]      # a → b  (зазор на целевом b)
            ba = ss[a][b]["gap"]      # b → a
        except KeyError:
            continue
        out[f"{a}<->{b}"] = {f"{a}->{b}": ab, f"{b}->{a}": ba,
                             "asymmetry": ab - ba, "abs_asymmetry": abs(ab - ba),
                             "script_pair": " ↔ ".join(sorted({SCRIPT_OF[a], SCRIPT_OF[b]}))}
    if out:
        out["_mean_abs_asymmetry"] = float(np.mean([v["abs_asymmetry"]
                                                    for v in out.values()
                                                    if isinstance(v, dict)]))
    return out


def collapse_summary(res, how):
    """Где и при каком объёме train сиды схлопываются в вырожденный предсказатель.

    Нужно, чтобы отличить свойство переноса от известной нестабильности файнтюна на
    малых выборках: у бустинга поверх ЗАМОРОЖЕННЫХ признаков траектории обучения,
    вдоль которой можно сорваться, попросту нет.
    """
    cells, by_n, by_tgt = [], {}, {}
    for cname, e in res.items():
        if how not in e["by_selection"]:
            continue
        sel = e["by_selection"][how]
        n = e["composition"].get("n_total")
        srcs = [lg for lg in LANGS if e["composition"].get("by_lang", {}).get(lg, 0) > 0]
        for lg in LANGS:
            nb = sel[lg]["n_collapsed_benign"]
            ns = sel[lg]["n_collapsed_scam"]
            if nb + ns:
                cells.append({"config": cname, "target": lg, "n_train": n,
                              "zero_shot": lg not in srcs,
                              "n_collapsed_benign": nb, "n_collapsed_scam": ns,
                              "f1_values": sel[lg]["f1"]["values"],
                              "recall_values": sel[lg]["recall"]["values"]})
                by_n[n] = by_n.get(n, 0) + 1
                by_tgt[lg] = by_tgt.get(lg, 0) + 1
    total = sum(1 for e in res.values() if how in e["by_selection"]) * len(LANGS)
    return {"n_cells_total": total, "n_cells_collapsed": len(cells),
            "share": len(cells) / total if total else 0.0,
            "by_n_train": dict(sorted(by_n.items())), "by_target": by_tgt,
            "n_zero_shot": sum(1 for c in cells if c["zero_shot"]),
            "cells": cells}


def source_vs_target(gaps):
    """Разложение зазора: эффект ЯЗЫКА-ИСТОЧНИКА против эффекта ЯЗЫКА-ЦЕЛИ.

    Пара письменностей — величина симметричная, поэтому она не может объяснить
    разницу между «zh как источник» и «zh как цель». Если разброс по источникам
    заметно больше разброса по целям, дело в направлении, а не в графике.
    """
    per_src, per_tgt = {}, {}
    for tgt, d in gaps["single_source"].items():
        for src, v in d.items():
            per_src.setdefault(src, []).append(v["gap"])
            per_tgt.setdefault(tgt, []).append(v["gap"])
    as_src = {k: float(np.mean(v)) for k, v in sorted(per_src.items())}
    as_tgt = {k: float(np.mean(v)) for k, v in sorted(per_tgt.items())}
    return {"as_source": as_src, "as_target": as_tgt,
            "spread_source": (max(as_src.values()) - min(as_src.values())) if as_src else 0.0,
            "spread_target": (max(as_tgt.values()) - min(as_tgt.values())) if as_tgt else 0.0}


def by_script(gaps):
    """Группировка зазоров по паре письменностей — проверка графической дистанции."""
    buckets = {}
    for tgt, per_src in gaps["single_source"].items():
        for src, d in per_src.items():
            buckets.setdefault(d["script_pair"], []).append(
                {"pair": f"{src}->{tgt}", "gap": d["gap"]})
    return {k: {"n": len(v), "mean_gap": float(np.mean([x["gap"] for x in v])),
                "std_gap": float(np.std([x["gap"] for x in v], ddof=1)) if len(v) > 1 else 0.0,
                "pairs": v}
            for k, v in sorted(buckets.items())}


# ──────────────────────────────── вывод ────────────────────────────────
def print_matrix(name, mat, how):
    print(f"\n  матрица переноса [{name}, отбор по {how}] — среднее~медиана:")
    print(f"    {'обучение':<22}{'n':>6}" + "".join(f"{'тест ' + lg:>16}" for lg in LANGS))
    for r, cells in mat.items():
        srcs = set(cells["_sources"])
        line = f"    {r:<22}{cells['_n_train'] or '':>6}"
        for lg in LANGS:
            v = cells[lg]
            mark = "" if lg in srcs else "*"           # * = zero-shot клетка
            col = v.get("n_collapsed_benign", 0) + v.get("n_collapsed_scam", 0)
            flag = f"!{col}" if col else "  "          # !N = N сидов схлопнулись
            line += (f"{v['f1']['mean']:>7.4f}{mark:<1}~{v['f1']['median']:.3f}{flag} ")
        print(line)
    print("    * — zero-shot; ~ — медиана по сидам; !N — N сидов из 3 схлопнулись "
          "в вырожденный предсказатель")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers12", default="runs/crosslingual/layers12.json")
    ap.add_argument("--xlmr", default="runs/crosslingual/results_raw.json")
    ap.add_argument("--out", default="runs/crosslingual/results.json")
    args = ap.parse_args()

    # опора для флага вырожденности: F1 константы «всегда scam» по языкам (слой 0)
    const_f1 = {}
    l0 = pathlib.Path("runs/layer0/results.json")
    if l0.exists():
        pl = json.loads(l0.read_text(encoding="utf-8"))["predictors"]["const_scam"]
        const_f1 = {lg: pl["per_language"][lg]["f1"] for lg in LANGS}
        print("F1 константы «всегда scam» по языкам: " +
              ", ".join(f"{lg} {v:.4f}" for lg, v in const_f1.items()))

    models = {}
    if pathlib.Path(args.layers12).exists():
        models |= from_layers12(args.layers12)
    models |= from_xlmr(args.xlmr)
    if not models:
        raise SystemExit("нет ни одного источника результатов")
    print(f"моделей: {list(models)}")

    payload = {"selections": SELECTIONS, "langs": LANGS, "ks": KS, "models": {}}
    for name, cfgs in models.items():
        res = analyse_model(name, cfgs)
        entry = {"configs": res, "by_selection": {}}
        for how in SELECTIONS:
            mat = transfer_matrix(res, how)
            gaps = transfer_gaps(res, how)
            entry["by_selection"][how] = {
                "matrix": mat, "gaps": gaps, "fewshot": fewshot_curves(res, how),
                "by_script": by_script(gaps), "asymmetry": asymmetry(gaps),
                "source_vs_target": source_vs_target(gaps),
                "collapse": collapse_summary(res, how),
                "degenerate_cells": mark_degenerate(res, how, const_f1),
            }
        payload["models"][name] = entry

        print(f"\n{'=' * 78}\nМОДЕЛЬ: {name}\n{'=' * 78}")
        for how in SELECTIONS:
            print_matrix(name, entry["by_selection"][how]["matrix"], how)
        g = entry["by_selection"]["dev_source"]["gaps"]
        print("\n  зазор zero-shot (отбор по dev_source), медиана F1 относительно in-language:")
        for tgt in LANGS:
            if tgt in g["single_source"]:
                parts = "  ".join(f"{s}→{tgt} {d['gap']:+.4f}"
                                  for s, d in sorted(g["single_source"][tgt].items()))
                print(f"    один источник  → {tgt}: {parts}")
        for tag, label in (("loo_full", "два источника, полный "),
                           ("loo_matched", "два источника, n=464")):
            row = "  ".join(f"→{t} {d['gap']:+.4f}" for t, d in sorted(g[tag].items()))
            if row:
                print(f"    {label}: {row}")
        print("\n  по парам письменностей:")
        for k, v in entry["by_selection"]["dev_source"]["by_script"].items():
            print(f"    {k:<26} n={v['n']}  средний зазор {v['mean_gap']:+.4f} "
                  f"± {v['std_gap']:.4f}")
        asym = entry["by_selection"]["dev_source"]["asymmetry"]
        print("\n  асимметрия направления (дистанция письменностей симметрична, "
              "зазор — нет):")
        for k, v in asym.items():
            if k.startswith("_"):
                continue
            a, b = k.split("<->")
            print(f"    {k:<12} {a}→{b} {v[f'{a}->{b}']:+.4f}  {b}→{a} "
                  f"{v[f'{b}->{a}']:+.4f}  |разница| {v['abs_asymmetry']:.4f}")
        if "_mean_abs_asymmetry" in asym:
            spread = [v["mean_gap"] for v in
                      entry["by_selection"]["dev_source"]["by_script"].values()]
            print(f"    средняя |асимметрия| {asym['_mean_abs_asymmetry']:.4f} против "
                  f"размаха между парами письменностей "
                  f"{max(spread) - min(spread):.4f}")
        svt = entry["by_selection"]["dev_source"]["source_vs_target"]
        print("\n  эффект источника против эффекта цели (средний зазор):")
        print("    как ИСТОЧНИК: " + "  ".join(f"{k} {v:+.4f}"
                                               for k, v in svt["as_source"].items())
              + f"   размах {svt['spread_source']:.4f}")
        print("    как ЦЕЛЬ:     " + "  ".join(f"{k} {v:+.4f}"
                                               for k, v in svt["as_target"].items())
              + f"   размах {svt['spread_target']:.4f}")
        cs = entry["by_selection"]["dev_source"]["collapse"]
        print(f"\n  схлопывание сидов: {cs['n_cells_collapsed']} клеток из "
              f"{cs['n_cells_total']} ({cs['share']:.0%}), из них zero-shot "
              f"{cs['n_zero_shot']}")
        if cs["by_n_train"]:
            print("    по объёму train: " + "  ".join(f"n={k}: {v}"
                                                      for k, v in cs["by_n_train"].items()))
            print("    по целевому языку: " + "  ".join(f"{k}: {v}"
                                                        for k, v in sorted(cs["by_target"].items())))
        deg = entry["by_selection"]["dev_source"]["degenerate_cells"]
        if deg:
            print(f"\n  ⚠ вырожденные клетки (recall≈1, низкая precision — почти "
                  f"«всегда scam»): {len(deg)}")
            for k, v in sorted(deg.items(), key=lambda kv: -kv[1]["f1"])[:6]:
                print(f"    {k:<28} F1 {v['f1']:.4f} (константа "
                      f"{v['constant_f1']:.4f}, +{v['above_constant']:.4f})  "
                      f"recall {v['recall']:.2f} precision {v['precision']:.2f}")
        print("\n  few-shot: k для догона in-language")
        for pair, d in entry["by_selection"]["dev_source"]["fewshot"].items():
            pts = "  ".join(f"k={k}:{v:.3f}" for k, v in sorted(d["points"].items()))
            r = d["k_to_reach_in_language"]
            print(f"    {pair:<10} {pts}   in-lang {d['in_language']:.3f} → "
                  f"{'догнал при k=' + str(r) if r is not None else 'НЕ догнал при k≤100'}")

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {p}")


if __name__ == "__main__":
    main()
