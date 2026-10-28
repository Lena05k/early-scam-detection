"""Групповой стратифицированный сплит train/val/test корпуса.

Единица распределения — scenario_group (не диалог): ни одна группа не
пересекает границу сплитов, что автоматически сохраняет целостность
twin-пар (они внутри одной группы по построению scenario_groups).

Стратификация по трём осям: язык × is_scam × бакет онсета
{benign(t*=∅), t*=1, t*∈[2..3], t*≥4}. Группа может быть смешанной
(scam+benign твины) — её профиль это вектор счётчиков по стратам.

Метод: рандомизированное жадное распределение групп (GRASP-подобное) с
минимизацией суммарной попарной JSD распределений страт между сплитами;
>= n_trials перезапусков от фиксированного master seed, выбирается лучший
валидный. Чистый stdlib, без ML-зависимостей (конвенция проекта).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from scam_onset.data.scenario_groups import (
    SIM_FIRST2_THRESHOLD,
    SIM_FULL_THRESHOLD,
    _first2_text,
    _full_text,
    cosine,
    tfidf_vectors,
)

SPLITS = ("train", "val", "test")

# --- параметры по умолчанию (переопределяются из configs/default.yaml) ---
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
DEFAULT_MASTER_SEED = 20260721
DEFAULT_N_TRIALS = 1000
PROPORTION_TOL_PP = 2.0        # допуск пропорций сплитов, п.п.
STRATA_TOL_PP = 5.0            # допуск доли страты в сплите vs глобальной, п.п.
HARD_NEG_TOL_PP = 7.0          # допуск доли hard-neg среди benign, п.п.
LARGE_GROUP_MIN_SIZE = 14      # группы >= этого размера: запрет val, в test <= 1
SECOND_BEST_PROB = 0.15        # GRASP: вероятность взять второй по счёту сплит


# ── страты ──

def onset_bucket(rec) -> str:
    """Бакет онсета: benign(t*=∅) / t1 / t2_3 / t4plus."""
    if rec["is_scam"] == 0:
        return "benign"
    t = rec["onset"]
    if t == 1:
        return "t1"
    if t in (2, 3):
        return "t2_3"
    return "t4plus"


def stratum(rec) -> tuple:
    return (rec["lang"], rec["is_scam"], onset_bucket(rec))


def stratum_label(key: tuple) -> str:
    lang, is_scam, bucket = key
    return f"{lang}/{'scam' if is_scam else 'benign'}/{bucket}"


# ── профили групп ──

def group_profiles(records) -> dict:
    """Профиль каждой scenario_group: размер, вектор страт, hard-neg и пр."""
    prof: dict[str, dict] = {}
    for r in records:
        g = r["scenario_group"]
        p = prof.setdefault(g, {
            "size": 0,
            "strata": Counter(),
            "n_benign": 0,
            "n_hard_neg": 0,
            "scam_types": set(),
        })
        p["size"] += 1
        p["strata"][stratum(r)] += 1
        if r["is_scam"] == 0:
            p["n_benign"] += 1
            p["n_hard_neg"] += int(r["hard_negative"])
        p["scam_types"].add(r["scam_type"])
    return prof


# ── дивергенция ──

def jsd(p: dict, q: dict, keys=None) -> float:
    """Дивергенция Йенсена–Шеннона (base 2, в [0,1]) двух распределений-словарей."""
    if keys is None:
        keys = set(p) | set(q)
    total = 0.0
    for k in keys:
        pk, qk = p.get(k, 0.0), q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if pk > 0:
            total += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            total += 0.5 * qk * math.log2(qk / mk)
    return total


def jsd_contributions(p: dict, q: dict, keys) -> dict:
    """Повкладовое разложение JSD по стратам (сумма = jsd(p, q))."""
    out = {}
    for k in keys:
        pk, qk = p.get(k, 0.0), q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        c = 0.0
        if pk > 0:
            c += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            c += 0.5 * qk * math.log2(qk / mk)
        out[k] = c
    return out


def _distribution(counts: Counter, n: int) -> dict:
    return {k: v / n for k, v in counts.items()} if n else {}


def total_pairwise_jsd(strata_by_split: dict, sizes: dict) -> float:
    dists = {s: _distribution(strata_by_split[s], sizes[s]) for s in SPLITS}
    return (jsd(dists["train"], dists["val"])
            + jsd(dists["train"], dists["test"])
            + jsd(dists["val"], dists["test"]))


# ── жадное распределение ──

def large_groups(profiles: dict, min_size: int = LARGE_GROUP_MIN_SIZE) -> list[str]:
    return sorted((g for g, p in profiles.items() if p["size"] >= min_size),
                  key=lambda g: -profiles[g]["size"])


def greedy_assign(profiles: dict, fractions: dict, rng: random.Random,
                  large: set[str], proportion_tol_pp: float = PROPORTION_TOL_PP,
                  second_best_prob: float = SECOND_BEST_PROB) -> dict:
    """Один рандомизированный жадный проход: group_id -> split.

    Группы обходятся по убыванию размера (ничьи перемешаны rng); каждая идёт
    в сплит с максимальной оставшейся «потребностью» её страт, нормированной
    на долю сплита. Крупные группы (large): val запрещён, в test не более одной.
    """
    n_total = sum(p["size"] for p in profiles.values())
    global_strata = Counter()
    for p in profiles.values():
        global_strata.update(p["strata"])

    cap = {s: (fractions[s] + proportion_tol_pp / 100.0) * n_total for s in SPLITS}
    sizes = {s: 0 for s in SPLITS}
    strata = {s: Counter() for s in SPLITS}
    large_in_test = 0

    order = list(profiles)
    rng.shuffle(order)
    order.sort(key=lambda g: -profiles[g]["size"])   # stable: ничьи в порядке shuffle

    assignment = {}
    for g in order:
        p = profiles[g]
        allowed = []
        for s in SPLITS:
            if g in large:
                if s == "val":
                    continue
                if s == "test" and large_in_test >= 1:
                    continue
            if sizes[s] + p["size"] > cap[s]:
                continue
            allowed.append(s)
        if not allowed:
            allowed = ["train"]   # запасной вариант; валидность проверится потом

        scored = []
        for s in allowed:
            f = fractions[s]
            need = sum(v * (f * global_strata[k] - strata[s][k]) / f
                       for k, v in p["strata"].items())
            scored.append((need + rng.random() * 1e-9, s))
        scored.sort(reverse=True)
        if len(scored) > 1 and rng.random() < second_best_prob:
            s_pick = scored[1][1]
        else:
            s_pick = scored[0][1]

        assignment[g] = s_pick
        sizes[s_pick] += p["size"]
        strata[s_pick].update(p["strata"])
        if g in large and s_pick == "test":
            large_in_test += 1
    return assignment


# ── быстрые проверки кандидата (без утечки текста — она по построению) ──

def _quick_valid(profiles: dict, assignment: dict, fractions: dict,
                 proportion_tol_pp: float = PROPORTION_TOL_PP,
                 strata_tol_pp: float = STRATA_TOL_PP,
                 hard_neg_tol_pp: float = HARD_NEG_TOL_PP) -> bool:
    n_total = sum(p["size"] for p in profiles.values())
    sizes = {s: 0 for s in SPLITS}
    strata = {s: Counter() for s in SPLITS}
    benign = {s: 0 for s in SPLITS}
    hard_neg = {s: 0 for s in SPLITS}
    train_scenarios = set()
    global_strata = Counter()
    global_benign = global_hn = 0

    for g, p in profiles.items():
        s = assignment[g]
        sizes[s] += p["size"]
        strata[s].update(p["strata"])
        benign[s] += p["n_benign"]
        hard_neg[s] += p["n_hard_neg"]
        if s == "train":
            train_scenarios |= p["scam_types"]
        global_strata.update(p["strata"])
        global_benign += p["n_benign"]
        global_hn += p["n_hard_neg"]

    for s in SPLITS:
        if abs(sizes[s] / n_total - fractions[s]) * 100 > proportion_tol_pp:
            return False
    for k, gk in global_strata.items():
        g_share = gk / n_total
        for s in SPLITS:
            if abs(strata[s][k] / sizes[s] - g_share) * 100 > strata_tol_pp:
                return False
    global_hn_share = global_hn / global_benign
    for s in SPLITS:
        if benign[s] == 0:
            return False
        if abs(hard_neg[s] / benign[s] - global_hn_share) * 100 > hard_neg_tol_pp:
            return False
    all_scenarios = set().union(*(p["scam_types"] for p in profiles.values()))
    return train_scenarios == all_scenarios


def search_best_split(records, fractions: dict | None = None,
                      master_seed: int = DEFAULT_MASTER_SEED,
                      n_trials: int = DEFAULT_N_TRIALS,
                      large_group_min_size: int = LARGE_GROUP_MIN_SIZE) -> dict:
    """>= n_trials рандомизированных жадных проходов; лучший валидный по JSD.

    Возвращает dict: assignment (group -> split), split_of (id диалога -> split),
    best_trial, total_jsd, n_valid_trials, large (крупные группы и их сплиты).
    """
    fractions = fractions or DEFAULT_FRACTIONS
    profiles = group_profiles(records)
    large = set(large_groups(profiles, large_group_min_size))

    best = None   # (jsd, trial, assignment)
    n_valid = 0
    for trial in range(n_trials):
        rng = random.Random(master_seed * 1_000_003 + trial)
        assignment = greedy_assign(profiles, fractions, rng, large)
        if not _quick_valid(profiles, assignment, fractions):
            continue
        n_valid += 1
        sizes = {s: 0 for s in SPLITS}
        strata = {s: Counter() for s in SPLITS}
        for g, p in profiles.items():
            s = assignment[g]
            sizes[s] += p["size"]
            strata[s].update(p["strata"])
        score = total_pairwise_jsd(strata, sizes)
        if best is None or score < best[0]:
            best = (score, trial, assignment)

    if best is None:
        raise RuntimeError(
            f"ни один из {n_trials} проходов не дал валидный сплит — "
            "ослабь допуски или увеличь n_trials")

    score, trial, assignment = best
    split_of = {r["id"]: assignment[r["scenario_group"]] for r in records}
    return {
        "assignment": assignment,
        "split_of": split_of,
        "best_trial": trial,
        "total_jsd": score,
        "n_valid_trials": n_valid,
        "large": {g: assignment[g] for g in sorted(large, key=lambda g: -profiles[g]["size"])},
    }


# ── валидация готового сплита (шаг 3.5; переиспользуется тестами) ──

def validate_split(records, fractions: dict | None = None,
                   proportion_tol_pp: float = PROPORTION_TOL_PP,
                   strata_tol_pp: float = STRATA_TOL_PP,
                   hard_neg_tol_pp: float = HARD_NEG_TOL_PP) -> dict:
    """Все проверки 3.5, кроме утечки текста (она отдельно — дорогая).

    Каждая проверка: {"pass": bool, ...детали}. Записи должны нести поле split.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    n_total = len(records)
    checks = {}

    # группа не в двух сплитах
    splits_of_group = defaultdict(set)
    for r in records:
        splits_of_group[r["scenario_group"]].add(r["split"])
    broken_groups = sorted(g for g, ss in splits_of_group.items() if len(ss) > 1)
    checks["no_group_across_splits"] = {"pass": not broken_groups,
                                        "broken_groups": broken_groups}

    # twin-пары не разорваны
    by_id = {r["id"]: r for r in records}
    broken_twins = []
    for r in records:
        tw = r.get("twin_id")
        if tw and tw in by_id and by_id[tw]["split"] != r["split"]:
            pair = tuple(sorted((r["id"], tw)))
            if pair not in broken_twins:
                broken_twins.append(pair)
    checks["no_broken_twins"] = {"pass": not broken_twins,
                                 "broken_twins": sorted(broken_twins)}

    # пропорции
    sizes = Counter(r["split"] for r in records)
    props = {s: sizes[s] / n_total for s in SPLITS}
    checks["proportions"] = {
        "pass": all(abs(props[s] - fractions[s]) * 100 <= proportion_tol_pp
                    for s in SPLITS),
        "sizes": {s: sizes[s] for s in SPLITS},
        "shares": {s: round(props[s], 4) for s in SPLITS},
        "target": fractions, "tolerance_pp": proportion_tol_pp,
    }

    # страты: доля в сплите vs глобальная, <= strata_tol_pp п.п.
    strata_split = {s: Counter() for s in SPLITS}
    global_strata = Counter()
    for r in records:
        k = stratum(r)
        strata_split[r["split"]][k] += 1
        global_strata[k] += 1
    worst = 0.0
    violations = []
    strata_table = {}
    for k in sorted(global_strata):
        g_share = global_strata[k] / n_total
        row = {"global_share": round(g_share, 4)}
        for s in SPLITS:
            share = strata_split[s][k] / sizes[s]
            dev = abs(share - g_share) * 100
            worst = max(worst, dev)
            row[s] = round(share, 4)
            if dev > strata_tol_pp:
                violations.append({"stratum": stratum_label(k), "split": s,
                                   "deviation_pp": round(dev, 2)})
        strata_table[stratum_label(k)] = row
    checks["strata_balance"] = {"pass": not violations,
                                "worst_deviation_pp": round(worst, 2),
                                "tolerance_pp": strata_tol_pp,
                                "violations": violations,
                                "table": strata_table}

    # hard-neg среди benign
    global_benign = sum(1 for r in records if r["is_scam"] == 0)
    global_hn = sum(r["hard_negative"] for r in records if r["is_scam"] == 0)
    g_share = global_hn / global_benign
    hn_rows, hn_ok = {}, True
    for s in SPLITS:
        b = sum(1 for r in records if r["split"] == s and r["is_scam"] == 0)
        h = sum(r["hard_negative"] for r in records
                if r["split"] == s and r["is_scam"] == 0)
        share = h / b if b else 0.0
        ok = b > 0 and abs(share - g_share) * 100 <= hard_neg_tol_pp
        hn_ok &= ok
        hn_rows[s] = {"benign": b, "hard_neg": h, "share": round(share, 4)}
    checks["hard_neg_share"] = {"pass": hn_ok, "global_share": round(g_share, 4),
                                "tolerance_pp": hard_neg_tol_pp, "by_split": hn_rows}

    # покрытие сценариев
    scenarios = {s: set() for s in SPLITS}
    all_scen = set()
    for r in records:
        scenarios[r["split"]].add(r["scam_type"])
        all_scen.add(r["scam_type"])
    missing_val = sorted(all_scen - scenarios["val"])
    missing_test = sorted(all_scen - scenarios["test"])
    checks["scenario_coverage"] = {
        "pass": scenarios["train"] == all_scen,
        "n_scenarios": len(all_scen),
        "missing_in_train": sorted(all_scen - scenarios["train"]),
        "missing_in_val": missing_val,
        "missing_in_test": missing_test,
    }
    return checks


def cross_split_leakage(records, split_a: str = "train", split_b: str = "test",
                        sim_first2_threshold: float = SIM_FIRST2_THRESHOLD,
                        sim_full_threshold: float = SIM_FULL_THRESHOLD) -> dict:
    """Контроль утечки текста между сплитами — та же метрика, что в scenario_groups.

    TF-IDF считается внутри группы (lang × scam_type) по ВСЕМ её диалогам
    (идентично разметке групп), сравниваются только пары split_a × split_b.
    """
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["lang"], r["scam_type"])].append(r)

    leaks = []
    n_pairs = 0
    for _, recs in buckets.items():
        idx_a = [i for i, r in enumerate(recs) if r["split"] == split_a]
        idx_b = [i for i, r in enumerate(recs) if r["split"] == split_b]
        if not idx_a or not idx_b:
            continue
        v2 = tfidf_vectors([_first2_text(r) for r in recs])
        vf = tfidf_vectors([_full_text(r) for r in recs])
        for i in idx_a:
            for j in idx_b:
                n_pairs += 1
                s2 = cosine(v2[i], v2[j])
                sf = cosine(vf[i], vf[j])
                if s2 >= sim_first2_threshold or sf >= sim_full_threshold:
                    leaks.append({"a": recs[i]["id"], "b": recs[j]["id"],
                                  "sim_first2": round(s2, 3),
                                  "sim_full": round(sf, 3)})
    return {"pass": not leaks, "pairs_checked": n_pairs, "leaks": leaks}
