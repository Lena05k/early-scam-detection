"""Стадия 0: программные скелеты S = ⟨lang, scenario, roles, onset_cand, twin⟩ (методичка §9.1).

Без LLM: раскладка сценариев/языков, баланс scam/benign ~50/50, пары скам↔benign-двойник.
Детерминированно по seed (из configs/default.yaml: seeds).
"""
import itertools
import random
from collections import deque
from dataclasses import dataclass

from .scenarios import BENIGN_SCENARIOS, SCAM_SCENARIOS, SCENARIOS


@dataclass
class Skeleton:
    sid: str                    # id диалога, напр. pilot_ru_romance_007
    lang: str
    scenario_id: str
    is_scam: int
    subtypes: tuple             # ожидаемые подтипы сценария (черновик ставит LLM)
    hard_negative: int
    n_turns: int                # целевая длина (реплик)
    onset_cand: int | None      # запланированный ход онсета t*_cand; benign -> None
    twin_sid: str | None = None # sid парного диалога (скам ↔ benign-двойник)


def build_skeletons(n, langs, turn_min=6, turn_max=14, seed=13):
    """Строит n скелетов: чередует scam/benign (баланс ~50/50, |разница| <= 1).

    Scam-сценарии идут по кругу; если у сценария есть benign-двойник — пара
    связывается через twin_sid (одинаковый язык), иначе benign берётся из
    общего цикла benign-сценариев без связки.
    """
    rng = random.Random(seed)
    out, k, pair_i, scam_i, benign_i = [], 0, 0, 0, 0
    while len(out) < n:
        lang = langs[pair_i % len(langs)]
        pair_i += 1
        sc = SCAM_SCENARIOS[scam_i % len(SCAM_SCENARIOS)]
        scam_i += 1
        n_turns = rng.randint(turn_min, turn_max)
        lo, hi = sc.onset_window
        # оставить >=2 реплики после онсета под эскалацию (методичка §9.1: эскалация после t*)
        onset_cand = max(1, min(rng.randint(lo, hi), n_turns - 2))
        k += 1
        scam_skel = Skeleton(
            sid=f"pilot_{lang}_{sc.id}_{k:03d}", lang=lang, scenario_id=sc.id,
            is_scam=1, subtypes=sc.subtypes, hard_negative=0,
            n_turns=n_turns, onset_cand=onset_cand,
        )
        out.append(scam_skel)
        if len(out) >= n:
            break
        if sc.twin:
            b = SCENARIOS[sc.twin]
        else:
            b = BENIGN_SCENARIOS[benign_i % len(BENIGN_SCENARIOS)]
            benign_i += 1
        k += 1
        benign_skel = Skeleton(
            sid=f"pilot_{lang}_{b.id}_{k:03d}", lang=lang, scenario_id=b.id,
            is_scam=0, subtypes=(), hard_negative=b.hard_negative,
            n_turns=rng.randint(turn_min, turn_max), onset_cand=None,
            twin_sid=scam_skel.sid if sc.twin else None,
        )
        if sc.twin:
            scam_skel.twin_sid = benign_skel.sid
        out.append(benign_skel)
    return out


def _link_twins(scam_list, benign_list):
    """Связать скам-скелеты с benign-двойниками нужного сценария (в пределах языка).

    Каждый twinnable-скам берёт свободный benign своего twin-сценария; если такого
    в пуле нет — остаётся без пары (standalone). Пары двусторонние (twin_sid у обоих).
    Так twin-пары СОХРАНЯЮТСЯ, но benign-микс (и hard_neg-доля) задаётся отдельно.
    """
    pool = {}
    for b in benign_list:
        pool.setdefault(b.scenario_id, deque()).append(b)
    for s in scam_list:
        twin = SCENARIOS[s.scenario_id].twin
        if twin and pool.get(twin):
            b = pool[twin].popleft()
            s.twin_sid, b.twin_sid = b.sid, s.sid


def _interleave(a, b):
    """Перемежает два списка (scam, benign, scam, benign, …) для равномерного порядка."""
    out = []
    for x, y in itertools.zip_longest(a, b):
        if x is not None:
            out.append(x)
        if y is not None:
            out.append(y)
    return out


def build_corpus_skeletons(per_lang, scam_ratio=0.5, hard_neg_ratio=0.35,
                           turn_min=6, turn_max=14, min_per_scenario=10,
                           seed=13, prefix="corpus"):
    """Скелеты полного корпуса: точные счётчики по языкам + баланс меток/сценариев.

    Для каждого языка L из ``per_lang`` строит ровно ``per_lang[L]`` скелетов:
    - scam/benign по ``scam_ratio`` (|разница| <= 1);
    - scam-сценарии — round-robin по 12 (пропорционально, с минимумами при масштабе);
    - benign-сценарии — с целевой долей hard-negative ``hard_neg_ratio`` среди benign
      (error-diffusion между hard-neg и чистыми benign-сценариями), round-robin внутри групп;
    - twin-пары связываются внутри языка (``_link_twins``).

    Детерминированно по ``seed`` (RNG + сортировка языков + фиксированный порядок сценариев) —
    sid стабилен между запусками, что и требуется для resume по skeleton_id.
    ``min_per_scenario`` — контракт вызывающего (при 900 выполняется round-robin'ом);
    возвращаемые скелеты его удовлетворяют на масштабе корпуса.
    """
    rng = random.Random(seed)
    hn = [b for b in BENIGN_SCENARIOS if b.hard_negative]
    non_hn = [b for b in BENIGN_SCENARIOS if not b.hard_negative]
    counter = 0
    out = []

    for lang in sorted(per_lang):
        target = per_lang[lang]
        n_scam = round(target * scam_ratio)
        n_benign = target - n_scam

        scam_list = []
        for i in range(n_scam):
            sc = SCAM_SCENARIOS[i % len(SCAM_SCENARIOS)]
            counter += 1
            nt = rng.randint(turn_min, turn_max)
            lo, hi = sc.onset_window
            onset_cand = max(1, min(rng.randint(lo, hi), nt - 2))
            scam_list.append(Skeleton(
                sid=f"{prefix}_{lang}_{sc.id}_{counter:04d}", lang=lang, scenario_id=sc.id,
                is_scam=1, subtypes=sc.subtypes, hard_negative=0,
                n_turns=nt, onset_cand=onset_cand))

        benign_list = []
        acc, hn_i, nonhn_i = 0.0, 0, 0
        for _ in range(n_benign):
            acc += hard_neg_ratio
            if acc >= 1.0 and hn:                     # error-diffusion -> доля hard_neg_ratio
                acc -= 1.0
                b = hn[hn_i % len(hn)]
                hn_i += 1
            else:
                b = (non_hn or hn)[nonhn_i % len(non_hn or hn)]
                nonhn_i += 1
            counter += 1
            benign_list.append(Skeleton(
                sid=f"{prefix}_{lang}_{b.id}_{counter:04d}", lang=lang, scenario_id=b.id,
                is_scam=0, subtypes=(), hard_negative=b.hard_negative,
                n_turns=rng.randint(turn_min, turn_max), onset_cand=None))

        _link_twins(scam_list, benign_list)
        out.extend(_interleave(scam_list, benign_list))
    return out
