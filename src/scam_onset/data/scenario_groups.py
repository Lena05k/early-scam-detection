"""Разметка scenario_group: связные группы шаблонно похожих диалогов.

Назначение — не дать «близнецам» (шаблонно одинаковым диалогам) расползтись
по разным train/val/test-сплитам, что дало бы утечку. Похожесть считается
ТОЛЬКО внутри групп (язык × тип сценария) на символьных n-граммах (char_wb),
одинаково для ru/en/zh — словесная токенизация для zh не работает.

Чистый stdlib, без ML-зависимостей (конвенция проекта для не-моделей).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from itertools import combinations

# --- пороги и параметры (крутить здесь) ---
SIM_FIRST2_THRESHOLD = 0.70   # похожесть первых 2 реплик (шаблон завязки)
SIM_FULL_THRESHOLD = 0.60     # похожесть полного текста диалога
NGRAM_MIN = 3
NGRAM_MAX = 5

_WHITESPACE = re.compile(r"\s\s+")


def char_wb_ngrams(text: str, nmin: int = NGRAM_MIN, nmax: int = NGRAM_MAX) -> list[str]:
    """Символьные n-граммы в границах слов (семантика sklearn analyzer='char_wb').

    Каждое слово дополняется пробелом с двух сторон; n-граммы скользят внутри
    слова. Слово короче n учитывается один раз. Текст приводится к нижнему
    регистру (для zh это no-op, метрика единообразна для всех языков).
    """
    text = _WHITESPACE.sub(" ", text.lower())
    ngrams: list[str] = []
    for w in text.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(nmin, nmax + 1):
            offset = 0
            ngrams.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams.append(w[offset:offset + n])
            if offset == 0:  # слово короче n — считаем один раз
                break
    return ngrams


def tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    """TF-IDF по char_wb n-граммам с L2-нормировкой.

    Повторяет дефолт sklearn TfidfVectorizer: smooth_idf, norm='l2',
    idf(t) = ln((1+N)/(1+df)) + 1, вес = tf * idf. IDF считается по переданному
    набору docs (внутри группы), поэтому список должен быть одной группой.
    """
    counts = [Counter(char_wb_ngrams(d)) for d in docs]
    n = len(docs)
    df: Counter[str] = Counter()
    for c in counts:
        df.update(c.keys())
    idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}

    vecs: list[dict[str, float]] = []
    for c in counts:
        v = {t: tf * idf[t] for t, tf in c.items()}
        norm = math.sqrt(sum(w * w for w in v.values()))
        if norm > 0:
            v = {t: w / norm for t, w in v.items()}
        vecs.append(v)
    return vecs


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Косинус двух L2-нормированных разреженных векторов = их скалярное произведение."""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


class _DSU:
    """Union-find с path compression + union by size."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _first2_text(rec) -> str:
    turns = rec.get("turns", [])[:2]
    return "\n".join(t.get("text", "") for t in turns)


def _full_text(rec) -> str:
    return "\n".join(t.get("text", "") for t in rec.get("turns", []))


def _group_id(member_ids) -> str:
    """Детерминированный id группы из отсортированных id участников (от содержимого)."""
    h = hashlib.sha1("\n".join(sorted(member_ids)).encode("utf-8")).hexdigest()
    return "sg_" + h[:12]


def build_scenario_groups(
    records,
    sim_first2_threshold: float = SIM_FIRST2_THRESHOLD,
    sim_full_threshold: float = SIM_FULL_THRESHOLD,
):
    """Строит scenario_group для всех записей.

    Ребро добавляется между двумя диалогами одной группы (lang × scam_type),
    если sim(первые 2 реплики) >= порог1 ИЛИ sim(полный текст) >= порог2.
    Твин-пары (twin_id) объединяются принудительно поверх этого.

    Возвращает dict:
      group_of      : id диалога -> id scenario_group
      pairs         : список квалифицированных пар (a, b, sim_first2, sim_full)
      forced_twins  : сколько твин-пар были разведены похожестью и слиты принудительно
      groups        : id scenario_group -> список id участников
    """
    by_id = {r["id"]: r for r in records}
    ids = list(by_id)
    dsu = _DSU(ids)

    # --- рёбра похожести внутри групп (lang × scam_type) ---
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["lang"], r["scam_type"])].append(r["id"])

    pairs = []
    for key, member_ids in buckets.items():
        if len(member_ids) < 2:
            continue
        recs = [by_id[i] for i in member_ids]
        v_first2 = tfidf_vectors([_first2_text(r) for r in recs])
        v_full = tfidf_vectors([_full_text(r) for r in recs])
        for a, b in combinations(range(len(recs)), 2):
            s2 = cosine(v_first2[a], v_first2[b])
            sf = cosine(v_full[a], v_full[b])
            if s2 >= sim_first2_threshold or sf >= sim_full_threshold:
                dsu.union(member_ids[a], member_ids[b])
                pairs.append((member_ids[a], member_ids[b], s2, sf))

    # --- принудительное слияние твин-пар ---
    forced_twins = 0
    seen_twin_pairs = set()
    for r in records:
        tw = r.get("twin_id")
        if not tw or tw not in by_id:
            continue
        pair_key = tuple(sorted((r["id"], tw)))
        if pair_key in seen_twin_pairs:
            continue
        seen_twin_pairs.add(pair_key)
        if dsu.union(r["id"], tw):
            forced_twins += 1

    # --- материализация групп ---
    comp = defaultdict(list)
    for i in ids:
        comp[dsu.find(i)].append(i)
    group_of = {}
    groups = {}
    for members in comp.values():
        gid = _group_id(members)
        groups[gid] = sorted(members)
        for m in members:
            group_of[m] = gid

    return {
        "group_of": group_of,
        "pairs": pairs,
        "forced_twins": forced_twins,
        "groups": groups,
    }
