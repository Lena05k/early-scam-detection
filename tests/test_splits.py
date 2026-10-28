"""Тесты группового стратифицированного сплита (шаг 3).

Две части: юнит-тесты алгоритма на синтетике и валидации 3.5 на реальных
артефактах (data/corpus/corpus_v1.jsonl + data/corpus/splits_v1.json) —
пропускаются, если корпус ещё не размечен полем split.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from scam_onset.data import splits as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "corpus_v1.jsonl"
SPLITS_MAP = ROOT / "data" / "corpus" / "splits_v1.json"


# ── юнит: бакеты и страты ──

def test_onset_bucket_benign_is_none_bucket():
    assert sp.onset_bucket({"is_scam": 0, "onset": None}) == "benign"


@pytest.mark.parametrize("onset,bucket", [
    (1, "t1"), (2, "t2_3"), (3, "t2_3"), (4, "t4plus"), (11, "t4plus"),
])
def test_onset_bucket_scam(onset, bucket):
    assert sp.onset_bucket({"is_scam": 1, "onset": onset}) == bucket


# ── юнит: JSD ──

def test_jsd_identical_is_zero():
    p = {"a": 0.5, "b": 0.5}
    assert sp.jsd(p, dict(p)) == pytest.approx(0.0)


def test_jsd_disjoint_is_one():
    # base-2 JSD ограничена 1 и достигает её на непересекающихся распределениях
    assert sp.jsd({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_jsd_contributions_sum_to_jsd():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    q = {"a": 0.5, "b": 0.4, "c": 0.1}
    keys = ["a", "b", "c"]
    contrib = sp.jsd_contributions(p, q, keys)
    assert sum(contrib.values()) == pytest.approx(sp.jsd(p, q))


# ── юнит: поиск на синтетике ──

def _synth_records(n_groups=120, group_size=2):
    """Синтетический корпус: твин-пары scam+benign, равномерно по языкам."""
    recs = []
    langs = ["ru", "en", "zh"]
    scens = ["phishing", "romance", "ordinary_request"]
    for g in range(n_groups):
        lang = langs[g % 3]
        scen = scens[g % 3]
        gid = f"sg_{g:04d}"
        for m in range(group_size):
            is_scam = m % 2
            recs.append({
                "id": f"d{g:04d}_{m}",
                "lang": lang,
                "scam_type": scen,
                "is_scam": is_scam,
                "onset": (g % 4 + 1) if is_scam else None,
                "hard_negative": int(not is_scam and g % 3 == 0),
                "twin_id": f"d{g:04d}_{1 - m}" if group_size == 2 else "",
                "scenario_group": gid,
                "turns": [{"i": i + 1, "text": "x"} for i in range(8)],
            })
    return recs


def test_search_keeps_groups_whole_and_proportions():
    recs = _synth_records()
    res = sp.search_best_split(recs, master_seed=7, n_trials=50)
    split_of = res["split_of"]
    # группы целиком в одном сплите
    by_group = {}
    for r in recs:
        by_group.setdefault(r["scenario_group"], set()).add(split_of[r["id"]])
    assert all(len(s) == 1 for s in by_group.values())
    # пропорции в допуске
    n = len(recs)
    for s, f in sp.DEFAULT_FRACTIONS.items():
        share = sum(1 for i in split_of.values() if i == s) / n
        assert abs(share - f) * 100 <= sp.PROPORTION_TOL_PP


def test_search_respects_large_group_constraint():
    recs = _synth_records(n_groups=150)
    # две «крупные» группы по 20 диалогов
    for g in range(2):
        gid = f"sg_big_{g}"
        for m in range(20):
            recs.append({
                "id": f"big{g}_{m}", "lang": "en", "scam_type": "phishing",
                "is_scam": m % 2, "onset": 2 if m % 2 else None,
                "hard_negative": 0, "twin_id": "", "scenario_group": gid,
                "turns": [{"i": i + 1, "text": "x"} for i in range(8)],
            })
    res = sp.search_best_split(recs, master_seed=7, n_trials=80,
                               large_group_min_size=14)
    placements = list(res["large"].values())
    assert len(placements) == 2
    assert "val" not in placements                      # ни одной в val
    assert placements.count("test") <= 1                # не более одной в test


def test_search_is_deterministic():
    recs = _synth_records(n_groups=120)
    r1 = sp.search_best_split(recs, master_seed=13, n_trials=20)
    r2 = sp.search_best_split(recs, master_seed=13, n_trials=20)
    assert r1["split_of"] == r2["split_of"]
    assert r1["best_trial"] == r2["best_trial"]


# ── валидации 3.5 на реальном корпусе ──

def _load_corpus():
    if not CORPUS.exists():
        pytest.skip("corpus_v1.jsonl отсутствует")
    with open(CORPUS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if "split" not in records[0]:
        pytest.skip("корпус ещё не размечен полем split (scripts/make_splits.py)")
    return records


@pytest.fixture(scope="module")
def corpus():
    return _load_corpus()


@pytest.fixture(scope="module")
def checks(corpus):
    return sp.validate_split(corpus)


def test_no_group_across_splits(checks):
    assert checks["no_group_across_splits"]["pass"], \
        checks["no_group_across_splits"]["broken_groups"][:5]


def test_no_broken_twins(checks):
    assert checks["no_broken_twins"]["pass"], \
        checks["no_broken_twins"]["broken_twins"][:5]


def test_proportions_70_10_20(checks):
    assert checks["proportions"]["pass"], checks["proportions"]


def test_strata_balance_within_5pp(checks):
    assert checks["strata_balance"]["pass"], \
        checks["strata_balance"]["violations"]


def test_hard_neg_share_within_7pp(checks):
    assert checks["hard_neg_share"]["pass"], checks["hard_neg_share"]


def test_all_scenarios_in_train(checks):
    c = checks["scenario_coverage"]
    assert c["pass"] and not c["missing_in_train"]
    assert c["n_scenarios"] == 19


def test_text_leakage_train_test(corpus):
    # контрольная проверка: групповой сплит закрыл утечку по тем же порогам,
    # что строили scenario_group (sim_first2 >= 0.70 / sim_full >= 0.60)
    res = sp.cross_split_leakage(corpus, "train", "test")
    assert res["pass"], res["leaks"][:5]


def test_splits_map_matches_corpus(corpus):
    if not SPLITS_MAP.exists():
        pytest.skip("splits_v1.json отсутствует")
    mapping = json.loads(SPLITS_MAP.read_text(encoding="utf-8"))
    assert len(mapping) == len(corpus)
    for r in corpus:
        assert mapping[r["id"]] == r["split"]
