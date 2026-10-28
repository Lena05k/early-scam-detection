"""Тесты для разметки scenario_group (защита от утечки близнецов при сплите)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from scam_onset.data.scenario_groups import (
    char_wb_ngrams,
    tfidf_vectors,
    cosine,
    build_scenario_groups,
    _DSU,
    _group_id,
)


# ── char_wb n-граммы ──

def test_char_wb_short_word_counted_once():
    # слово, чей паддинг ровно длины n, даёт эту n-грамму один раз и обрывает
    # длинные n (семантика sklearn: `if offset == 0: break`).
    # "a" -> " a " (len 3): n=3 даёт " a " один раз, n=4,5 не добавляются.
    grams = char_wb_ngrams("a", nmin=3, nmax=5)
    assert grams == [" a "]


def test_char_wb_word_boundaries_and_lowercase():
    grams = char_wb_ngrams("Hi", nmin=3, nmax=3)
    # " hi " -> " hi", "hi "
    assert grams == [" hi", "hi "]


def test_char_wb_works_on_chinese():
    # для китайского нет пробелов — весь прогон это одно «слово», n-граммы есть
    grams = char_wb_ngrams("银行卡", nmin=3, nmax=3)
    assert grams  # непусто
    assert all(len(g) == 3 for g in grams)


# ── TF-IDF + косинус ──

def test_cosine_identical_is_one():
    v = tfidf_vectors(["hello world", "hello world"])
    assert abs(cosine(v[0], v[1]) - 1.0) < 1e-9


def test_cosine_disjoint_is_zero():
    v = tfidf_vectors(["aaaa", "zzzz"])
    assert cosine(v[0], v[1]) == 0.0


def test_cosine_symmetric_and_bounded():
    v = tfidf_vectors(["the quick brown fox", "the quick brown cat", "totally other text"])
    s = cosine(v[0], v[1])
    assert abs(s - cosine(v[1], v[0])) < 1e-12
    assert 0.0 <= s <= 1.0
    # похожая пара ближе, чем непохожая
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])


# ── union-find ──

def test_dsu_transitive_merge():
    d = _DSU(["a", "b", "c", "x"])
    d.union("a", "b")
    d.union("b", "c")
    assert d.find("a") == d.find("c")
    assert d.find("a") != d.find("x")
    assert d.union("a", "c") is False  # уже вместе


# ── детерминизм id группы ──

def test_group_id_deterministic_order_independent():
    assert _group_id(["b", "a"]) == _group_id(["a", "b"])
    assert _group_id(["a", "b"]) != _group_id(["a", "c"])
    assert _group_id(["a"]).startswith("sg_")


# ── интеграция build_scenario_groups ──

def _rec(rid, lang, scam_type, turns_texts, twin_id=None):
    return {
        "id": rid, "lang": lang, "scam_type": scam_type,
        "turns": [{"i": i + 1, "role": "atk", "text": t} for i, t in enumerate(turns_texts)],
        **({"twin_id": twin_id} if twin_id else {}),
    }


def test_similar_pair_grouped_singletons_unique():
    base = ["Hello this is the bank calling about your account security today",
            "We detected suspicious activity please confirm your card number now"]
    recs = [
        _rec("d1", "en", "phishing", base),
        _rec("d2", "en", "phishing", base),  # почти копия d1
        _rec("d3", "en", "phishing",
             ["Completely different opening about a weather forecast for tomorrow",
              "Nothing to do with the other dialogues here at all whatsoever"]),
    ]
    res = build_scenario_groups(recs)
    g = res["group_of"]
    assert g["d1"] == g["d2"]      # близнецы вместе
    assert g["d1"] != g["d3"]      # непохожий — отдельно
    # d3 — синглтон со своей уникальной группой
    assert len(res["groups"][g["d3"]]) == 1


def test_different_langs_never_grouped():
    txt = ["Confirm your card number immediately please", "Send the code now to verify"]
    recs = [
        _rec("en1", "en", "phishing", txt),
        _rec("ru1", "ru", "phishing", txt),  # тот же текст, другой язык
    ]
    res = build_scenario_groups(recs)
    assert res["group_of"]["en1"] != res["group_of"]["ru1"]


def test_twin_pair_force_merged_across_scam_types():
    # твины всегда в разных scam_type -> похожесть их не свяжет, нужен forced merge
    a = _rec("a1", "en", "phishing",
             ["Your parcel is held at customs pay a fee", "Click the link to release it"],
             twin_id="b1")
    b = _rec("b1", "en", "real_delivery",
             ["Totally unrelated benign delivery note about a scheduled drop off", "Have a nice day"],
             twin_id="a1")
    res = build_scenario_groups([a, b])
    assert res["group_of"]["a1"] == res["group_of"]["b1"]
    assert res["forced_twins"] == 1


def test_every_record_gets_a_group():
    recs = [_rec(f"d{i}", "en", "prize", [f"unique opening number {i} xyzzy", f"body {i} qwerty"])
            for i in range(5)]
    res = build_scenario_groups(recs)
    assert set(res["group_of"]) == {r["id"] for r in recs}
    assert all(v for v in res["group_of"].values())


def test_deterministic_repeat_runs():
    recs = [_rec(f"d{i}", "en", "prize", [f"opening {i}", f"body {i}"]) for i in range(6)]
    r1 = build_scenario_groups(recs)
    r2 = build_scenario_groups(recs)
    assert r1["group_of"] == r2["group_of"]
