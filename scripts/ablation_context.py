#!/usr/bin/env python3
"""Ablation диалогового контекста: конфигурации обучения и правила агрегации.

Первый заявленный вклад работы (tech_report §1) — детекция на уровне ДИАЛОГА, а не
отдельного сообщения. До сих пор он был только заявлен: не показано, что контекст
вообще что-то даёт. Здесь единица обучения и предсказания меняется, а всё остальное
(модель, гиперпараметры, сплиты, протокол отбора, тест) держится постоянным.

Конфигурации:
  dialogue          эталон, уже посчитан (0.9344 ± 0.0097), заново не обучается —
                    пер-диалоговые вероятности берутся из
                    runs/earliness_v2/prefix_probs_by_seed.json (последний префикс
                    длиной T = полный диалог; воспроизводит 0.9437/0.9243/0.9351)
  turn_naive        обучение на отдельных репликах, метка реплики = метка диалога
  turn_onset        метка реплики = 1, если её индекс ≥ t* диалога (benign целиком 0)

⚠ КОНТРОЛЬ РАЗМЕРА. 1392 диалога против 14003 реплик — это разница в 10 раз, и без
контроля эффект единицы обучения смешается с эффектом объёма. Поэтому у каждой
turn-конфигурации два варианта: полный набор реплик и подвыборка ровно до 1392
реплик (детерминированная, стратифицированная по lang × метка).

⚠ АГРЕГАЦИЯ. Правило «хотя бы одна реплика scam» смещено к высокому recall: чем
длиннее диалог, тем выше шанс, что хоть где-то сработает. Поэтому считаются три
правила, и лучшее выбирается ТОЛЬКО по gold-dev:
  any        max_i p_i ≥ 0.5                      (эквивалент «хотя бы одна»)
  frac_k     доля реплик с p_i ≥ 0.5 не меньше k  (k по gold-dev)
  maxprob    max_i p_i ≥ θ                        (θ по gold-dev)
Смещение по длине проверяется отдельно (доля ложных тревог против числа реплик).

  python scripts/ablation_context.py            # собрать манифест обучающих выборок
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from crosslingual import load_all  # noqa: E402

SEEDS = [20260721, 20260722, 20260723]
SUBSAMPLE_N = 1392          # ровно столько диалогов в train — с этим и сравниваем
SUBSAMPLE_SEED = 20260811
LANGS = ["en", "ru", "zh"]

# сетки параметров агрегации; выбор — только по gold-dev
FRAC_GRID = [round(float(x), 3) for x in np.arange(0.02, 1.001, 0.02)]
MAXPROB_GRID = [round(float(x), 2) for x in np.arange(0.01, 1.0, 0.01)]


# ───────────────────────────── разметка реплик ─────────────────────────────
def turn_label(rec, turn, scheme):
    """Метка отдельной реплики по одной из двух схем.

    naive — метка диалога целиком (реплика «Здравствуйте» из скам-диалога = scam);
    onset — реплика скамом считается только начиная с онсета t* (benign целиком 0).
    """
    is_scam = int(rec["is_scam"])
    if scheme == "naive":
        return is_scam
    if scheme == "onset":
        onset = rec.get("onset")
        if not is_scam or not onset:
            return 0
        return int(turn["i"] >= int(onset))
    raise ValueError(f"неизвестная схема разметки: {scheme}")


def build_turns(recs, scheme):
    """Диалоги → плоский список реплик с метками. Порядок детерминирован."""
    out = []
    for r in sorted(recs, key=lambda x: x["id"]):
        for t in r["turns"]:
            out.append({"dialog_id": r["id"], "i": int(t["i"]), "lang": r["lang"],
                        "label": turn_label(r, t, scheme)})
    return out


def subsample_turns(turns, n, seed):
    """Подвыборка ровно n реплик, пропорционально по (язык × метка)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for t in turns:
        groups.setdefault((t["lang"], t["label"]), []).append(t)
    for g in groups.values():
        g.sort(key=lambda t: (t["dialog_id"], t["i"]))
    total = len(turns)
    quotas = {k: int(round(n * len(g) / total)) for k, g in sorted(groups.items())}
    diff = n - sum(quotas.values())
    for key in sorted(quotas, key=lambda k: -len(groups[k])):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        quotas[key] += step
        diff -= step
    chosen = []
    for key, g in sorted(groups.items()):
        idx = rng.permutation(len(g))[:quotas[key]]
        chosen += [g[i] for i in sorted(idx)]
    chosen.sort(key=lambda t: (t["dialog_id"], t["i"]))
    assert len(chosen) == n, f"подвыборка дала {len(chosen)}, ожидалось {n}"
    return chosen


def enumerate_configs():
    """Четыре turn-конфигурации: две схемы разметки × два размера."""
    cfgs = []
    for scheme in ("naive", "onset"):
        cfgs.append({"name": f"turn_{scheme}_full", "scheme": scheme,
                     "size_matched": False})
        cfgs.append({"name": f"turn_{scheme}_n{SUBSAMPLE_N}", "scheme": scheme,
                     "size_matched": True})
    return cfgs


def composition(turns):
    return {
        "n": len(turns),
        "n_dialogs": len({t["dialog_id"] for t in turns}),
        "scam_share": float(np.mean([t["label"] for t in turns])),
        "by_lang": {lg: sum(1 for t in turns if t["lang"] == lg) for lg in LANGS},
        "by_lang_scam_share": {
            lg: float(np.mean([t["label"] for t in turns if t["lang"] == lg]))
            for lg in LANGS},
    }


# ───────────────────────────── правила агрегации ─────────────────────────────
def aggregate(probs, rule, param):
    """Вероятности реплик одного диалога → предсказание диалога (0/1).

    any      max p ≥ 0.5              — «хотя бы одна реплика scam»
    frac     доля (p ≥ 0.5) ≥ param   — порог по доле реплик
    maxprob  max p ≥ param            — порог по максимуму вероятности
    """
    a = np.asarray(probs, dtype=float)
    if rule == "any":
        return int((a >= 0.5).any())
    if rule == "frac":
        return int(float((a >= 0.5).mean()) >= param)
    if rule == "maxprob":
        return int(float(a.max()) >= param)
    raise ValueError(f"неизвестное правило агрегации: {rule}")


def rule_grid():
    """Все (правило, параметр) в фиксированном порядке — порядок задаёт тай-брейк."""
    out = [("any", None)]
    out += [("frac", k) for k in FRAC_GRID]
    out += [("maxprob", t) for t in MAXPROB_GRID]
    return out


def f1_np(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def dialog_preds(turn_probs, dialog_ids, rule, param):
    """{dialog_id: [p_1..p_T]} + порядок диалогов → массив предсказаний."""
    return np.array([aggregate(turn_probs[d], rule, param) for d in dialog_ids])


# ───────────────────────────── манифест ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/ablation_context/train_manifest.json")
    args = ap.parse_args()

    data = load_all()
    train = data["train"]
    print(f"train {len(train)} диалогов, {sum(len(r['turns']) for r in train)} реплик")

    manifest = {"subsample_n": SUBSAMPLE_N, "subsample_seed": SUBSAMPLE_SEED,
                "seeds": SEEDS, "configs": []}
    for cfg in enumerate_configs():
        turns = build_turns(train, cfg["scheme"])
        if cfg["size_matched"]:
            turns = subsample_turns(turns, SUBSAMPLE_N, SUBSAMPLE_SEED)
        comp = composition(turns)
        manifest["configs"].append({
            **cfg,
            "composition": comp,
            "turns": [[t["dialog_id"], t["i"], t["label"]] for t in turns],
        })
        print(f"  {cfg['name']:<22} реплик {comp['n']:>6}  диалогов {comp['n_dialogs']:>5}"
              f"  доля scam {comp['scam_share']:.3f}  по языкам {comp['by_lang']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"\nсохранено: {out} ({out.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
