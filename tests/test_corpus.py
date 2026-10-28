"""Тесты масштабирования корпуса (стадии 0-1) — БЕЗ реальных вызовов LLM."""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import generate_corpus as gc  # noqa: E402

from scam_onset.data.generate.generate import GenerationError  # noqa: E402
from scam_onset.data.generate.scenarios import SCENARIOS  # noqa: E402
from scam_onset.data.generate.skeleton import build_corpus_skeletons  # noqa: E402

PER_LANG = {"ru": 300, "en": 300, "zh": 300}


# ─── payload-хелперы (валидный ответ генератора под скелет) ───

def _payload(skel, T=None, onset="auto", subtypes="auto", speakers=("A", "B")):
    T = T or skel.n_turns
    turns = [{"i": i, "role": "atk" if i % 2 else "tgt",
              "speaker": speakers[0] if i % 2 else speakers[1],
              "text": f"turn {i} (AMOUNT)"} for i in range(1, T + 1)]
    if skel.is_scam:
        onset = 1 if onset == "auto" else onset
        subtypes = list(skel.subtypes) if subtypes == "auto" else subtypes
    else:
        onset = None if onset == "auto" else onset
        subtypes = [] if subtypes == "auto" else subtypes
    return {"turns": turns,
            "draft_labels": {"is_scam": skel.is_scam, "subtypes": subtypes, "onset": onset}}


class QueueClient:
    """Отдаёт заготовленные payload'ы по очереди (по одному на вызов)."""

    def __init__(self, payloads):
        self.q = [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in payloads]
        self.calls = 0

    def complete(self, model, system, user):
        p = self.q[min(self.calls, len(self.q) - 1)]
        self.calls += 1
        return p


class RaisingClient:
    """Всегда падает с ошибкой API (эмуляция троттлинга/обрыва)."""

    def __init__(self):
        self.calls = 0

    def complete(self, model, system, user):
        self.calls += 1
        raise GenerationError("HTTP 529 overloaded")


def _cfg(max_attempts=1):
    return {"generation": {"models": ["m1"], "api": {"max_attempts": max_attempts, "max_tokens": 100},
                           "turns": {"min": 6, "max": 14}},
            "corpus": {"turns_range": [6, 14], "target_n": 900}}


NOSLEEP = lambda _s: None  # noqa: E731


# ─── 1) баланс скелетов на 900: язык/метка/сценарий с минимумами ───

def test_corpus_skeletons_counts_and_balance():
    sk = build_corpus_skeletons(PER_LANG, seed=13)
    assert len(sk) == 900
    assert dict(Counter(s.lang for s in sk)) == {"ru": 300, "en": 300, "zh": 300}
    scam = sum(s.is_scam for s in sk)
    assert scam == 450 and (900 - scam) == 450                  # ровно 50/50
    for lang in PER_LANG:                                       # 50/50 в каждом языке
        ls = [s for s in sk if s.lang == lang]
        assert sum(x.is_scam for x in ls) == 150


def test_corpus_every_scenario_meets_minimum():
    sk = build_corpus_skeletons(PER_LANG, seed=13, min_per_scenario=10)
    counts = Counter(s.scenario_id for s in sk)
    assert set(counts) == set(SCENARIOS)                        # все 19 сценариев представлены
    assert min(counts.values()) >= 10, counts


def test_corpus_hard_neg_ratio_near_target():
    sk = build_corpus_skeletons(PER_LANG, hard_neg_ratio=0.35, seed=13)
    benign = [s for s in sk if not s.is_scam]
    share = sum(s.hard_negative for s in benign) / len(benign)
    assert 0.30 <= share <= 0.40, share                        # ~0.35 среди benign


def test_corpus_twin_pairs_preserved():
    sk = build_corpus_skeletons(PER_LANG, seed=13)
    by_sid = {s.sid: s for s in sk}
    linked = [s for s in sk if s.twin_sid]
    assert linked, "должны быть twin-пары"
    for s in linked:
        twin = by_sid[s.twin_sid]
        assert twin.twin_sid == s.sid                          # двусторонняя связь
        assert twin.is_scam != s.is_scam                       # scam <-> benign
        assert twin.lang == s.lang                             # тот же язык
        if s.is_scam:                                          # benign-двойник = twin-сценарий
            assert twin.scenario_id == SCENARIOS[s.scenario_id].twin


def test_corpus_skeletons_deterministic():
    a = build_corpus_skeletons(PER_LANG, seed=13)
    b = build_corpus_skeletons(PER_LANG, seed=13)
    # тот же конфиг -> полностью идентичны (sid + длины) — основа корректного resume
    assert [(s.sid, s.n_turns, s.onset_cand) for s in a] == [(s.sid, s.n_turns, s.onset_cand) for s in b]
    c = build_corpus_skeletons(PER_LANG, seed=42)
    # sid зависит только от конфига (не от seed) -> стабилен для resume;
    # seed влияет лишь на длины/онсет-кандидаты (RNG реально используется)
    assert [s.sid for s in a] == [s.sid for s in c]
    assert [s.n_turns for s in a] != [s.n_turns for s in c]


# ─── 2) resume не дублирует уже сгенерированные skeleton_id ───

def test_pending_skeletons_skips_done():
    sk = build_corpus_skeletons({"ru": 10}, seed=13)
    done = {sk[0].sid, sk[2].sid}
    todo = gc.pending_skeletons(sk, done)
    assert {s.sid for s in todo} == {s.sid for s in sk} - done
    assert len(todo) == len(sk) - 2


def test_resume_no_duplicate_ids(tmp_path):
    sk = build_corpus_skeletons({"ru": 6}, seed=13)
    corpus = tmp_path / "corpus.jsonl"
    rej = tmp_path / "corpus_rejected.jsonl"
    cfg = _cfg()

    # первый прогон — только первые 3 скелета (эмуляция обрыва): режем клиента под срез
    first = sk[:3]
    gc.run_corpus(first, cfg, QueueClient([_payload(s) for s in first]),
                  corpus, rej, sleep=NOSLEEP)
    assert len(gc.read_jsonl(corpus)) == 3

    # повторный прогон на ПОЛНОМ наборе с resume — дозаписывает недостающие 3, без дублей
    acc_new, _ = gc.run_corpus(sk, cfg, QueueClient([_payload(s) for s in sk[3:]]),
                               corpus, rej, sleep=NOSLEEP)
    recs = gc.read_jsonl(corpus)
    ids = [r["id"] for r in recs]
    assert len(acc_new) == 3                                    # только недостающие
    assert len(recs) == 6 and len(set(ids)) == 6               # без дублей
    assert set(ids) == {s.sid for s in sk}
    assert [r["item_id"] for r in recs] == [1, 2, 3, 4, 5, 6]  # item_id продолжает нумерацию


def test_only_lang_and_limit_filter_todo(tmp_path):
    sk = build_corpus_skeletons({"ru": 8, "en": 8, "zh": 8}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"
    cfg = _cfg()
    # только zh, не более 3 за запуск — план полный, но пишем 3 zh-диалога.
    # payload'ы выровнены под фактический порядок обработки (пробег todo).
    todo1 = [s for s in sk if s.lang == "zh"][:3]
    acc, _ = gc.run_corpus(sk, cfg, QueueClient([_payload(s) for s in todo1]),
                           corpus, rej, sleep=NOSLEEP, only_lang="zh", limit=3)
    recs = gc.read_jsonl(corpus)
    assert len(acc) == 3 and len(recs) == 3
    assert all(r["lang"] == "zh" for r in recs)                 # фильтр языка сработал
    zh_sids = {s.sid for s in sk if s.lang == "zh"}
    assert {r["id"] for r in recs} <= zh_sids                   # sid из ПОЛНОГО плана (resume-совместимо)

    # resume без фильтра дозаписывает остальные 21, zh-тройка не дублируется
    done3 = {r["id"] for r in recs}
    todo2 = [s for s in sk if s.sid not in done3]
    acc2, _ = gc.run_corpus(sk, cfg, QueueClient([_payload(s) for s in todo2]),
                            corpus, rej, sleep=NOSLEEP)
    all_recs = gc.read_jsonl(corpus)
    ids = [r["id"] for r in all_recs]
    assert len(acc2) == 21 and len(all_recs) == 24 and len(set(ids)) == 24


def test_unexpected_error_does_not_kill_run(tmp_path):
    """Неожиданное исключение на ОДНОМ диалоге -> брак, остальные генерируются (долгий прогон)."""
    sk = build_corpus_skeletons({"ru": 4}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"

    class BlowsUpOnce:
        def __init__(self, payloads):
            self.q = [json.dumps(p, ensure_ascii=False) for p in payloads]
            self.calls = 0

        def complete(self, model, system, user):
            self.calls += 1
            if self.calls == 2:                     # 2-й диалог падает нештатно
                raise RuntimeError("что-то совсем неожиданное")
            p = self.q[min(self.calls - 1, len(self.q) - 1)]
            return p

    client = BlowsUpOnce([_payload(s) for s in sk])
    acc, rejd = gc.run_corpus(sk, _cfg(), client, corpus, rej, sleep=NOSLEEP)
    assert len(rejd) == 1                                        # сбойный ушёл в брак
    assert any("необработанная ошибка" in r for r in rejd[0]["reasons"])
    assert client.calls == 4 and len(gc.read_jsonl(corpus)) == 3  # цикл дошёл до конца


def test_resume_when_complete_makes_no_calls(tmp_path):
    sk = build_corpus_skeletons({"ru": 4}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"
    cfg = _cfg()
    gc.run_corpus(sk, cfg, QueueClient([_payload(s) for s in sk]), corpus, rej, sleep=NOSLEEP)
    # всё готово -> клиент, который упал бы при любом вызове, НЕ должен вызываться
    guard = RaisingClient()
    acc, rejd = gc.run_corpus(sk, cfg, guard, corpus, rej, sleep=NOSLEEP)
    assert guard.calls == 0 and acc == [] and rejd == []
    assert len(gc.read_jsonl(corpus)) == 4


# ─── 3) rejected не ломает пайплайн ───

def test_bad_payload_goes_to_rejected_not_crash(tmp_path):
    sk = build_corpus_skeletons({"ru": 4}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"
    payloads = []
    for i, s in enumerate(sk):
        p = _payload(s)
        if i == 1:
            p["turns"][2]["speaker"] = "Третий лишний"          # 3 стороны -> брак
        payloads.append(p)
    acc, rejd = gc.run_corpus(sk, _cfg(), QueueClient(payloads), corpus, rej, sleep=NOSLEEP)
    assert len(acc) == 3 and len(rejd) == 1                     # один в брак, пайплайн жив
    assert gc.read_jsonl(rej)[0]["id"] == sk[1].sid
    assert sk[1].sid not in {r["id"] for r in gc.read_jsonl(corpus)}


def test_api_error_backoff_then_reject(tmp_path):
    sk = build_corpus_skeletons({"ru": 2}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"
    slept = []
    acc, rejd = gc.run_corpus(sk, _cfg(), RaisingClient(), corpus, rej,
                              sleep=lambda s: slept.append(s), max_retries=4, backoff_base=1.0)
    assert acc == [] and len(rejd) == 2                         # оба в брак, без падения
    assert all("API ошибка" in r["reasons"][0] for r in rejd)
    assert slept[:3] == [1.0, 2.0, 4.0]                         # экспоненциальный backoff


# ─── оценка стоимости и отчёт ───

def test_estimate_cost_shape():
    sk = build_corpus_skeletons({"ru": 6}, seed=13)
    est = gc.estimate_cost(sk, _cfg(max_attempts=2))
    assert est["pending"] == 6
    assert est["calls_min"] == 6 and est["calls_max"] == 12
    assert est["in_tokens_per_pass"] > 0 and est["out_tokens_max"] == 12 * 100


def test_corpus_report_composition(tmp_path):
    sk = build_corpus_skeletons({"ru": 8}, seed=13)
    corpus, rej = tmp_path / "c.jsonl", tmp_path / "c_rejected.jsonl"
    gc.run_corpus(sk, _cfg(), QueueClient([_payload(s) for s in sk]), corpus, rej, sleep=NOSLEEP)
    rep = gc.corpus_report(gc.read_jsonl(corpus), gc.read_jsonl(rej), ["m1"], _cfg())
    assert rep["n_accepted"] == 8
    assert rep["scam"] + rep["benign"] == 8
    assert rep["by_lang"] == {"ru": 8}
    assert rep["generators"] == {"m1": 8}
    assert rep["multi_generator_ok"] is False                   # один генератор
    assert sum(rep["onset_distribution"].values()) == rep["scam"]
