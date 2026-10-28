"""Тесты конвейера генерации (стадии 0-1) — БЕЗ реальных вызовов LLM (клиент мокается)."""
import json

import pytest

from scam_onset.data.generate.generate import (
    AnthropicClient, DeepSeekClient, GenerationError, RoutedClient, build_client,
    build_record, extract_json, render_dialogue, run_pipeline, validate_record,
    vendor_of,
)
from scam_onset.data.generate.prompts import build_prompt
from scam_onset.data.generate.scenarios import SCENARIOS, SUBTYPES
from scam_onset.data.generate.skeleton import Skeleton, build_skeletons

TMIN, TMAX = 6, 14


# --- хелперы -----------------------------------------------------------------------

def _skel(is_scam=1, n_turns=8, onset_cand=1):
    return Skeleton(
        sid="t_001", lang="ru",
        scenario_id="impersonation_bank" if is_scam else "real_bank_notice",
        is_scam=is_scam, subtypes=("money_request", "other") if is_scam else (),
        hard_negative=0 if is_scam else 1, n_turns=n_turns,
        onset_cand=onset_cand if is_scam else None,
    )


def _payload(skel, T=None, onset="auto", subtypes="auto", speakers=("Банк", "Клиент")):
    """Валидный ответ генератора под скелет: atk на нечётных репликах, tgt на чётных."""
    T = T or skel.n_turns
    turns = [
        {"i": i, "role": "atk" if i % 2 else "tgt",
         "speaker": speakers[0] if i % 2 else speakers[1],
         "text": f"реплика {i} (AMOUNT)"}
        for i in range(1, T + 1)
    ]
    if skel.is_scam:
        onset = 1 if onset == "auto" else onset            # реплика 1 — atk
        subtypes = list(skel.subtypes) if subtypes == "auto" else subtypes
    else:
        onset = None if onset == "auto" else onset
        subtypes = [] if subtypes == "auto" else subtypes
    return {"turns": turns,
            "draft_labels": {"is_scam": skel.is_scam, "subtypes": subtypes, "onset": onset}}


def _validate(skel, payload):
    return validate_record(build_record(skel, payload, "fake-model"), skel, TMIN, TMAX)


class FakeClient:
    """Отдаёт заготовленные ответы по очереди; последний — для всех дальнейших вызовов."""

    def __init__(self, payloads):
        self.payloads = [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
                         for p in payloads]
        self.calls = 0

    def complete(self, model, system, user):
        p = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return p


# --- таксономия и скелеты ----------------------------------------------------------

def test_taxonomy_consistent():
    for sc in SCENARIOS.values():
        if sc.is_scam:
            assert sc.subtypes and set(sc.subtypes) <= set(SUBTYPES), sc.id
            assert sc.onset_window is not None, sc.id
            if sc.twin:
                assert SCENARIOS[sc.twin].is_scam == 0, sc.id
        else:
            assert sc.subtypes == () and sc.onset_window is None, sc.id
    assert any(s.hard_negative for s in SCENARIOS.values() if not s.is_scam)


def test_skeletons_balanced_50_50():
    sk = build_skeletons(50, ["ru", "en", "zh"], TMIN, TMAX, seed=13)
    assert len(sk) == 50
    scam = sum(s.is_scam for s in sk)
    assert abs(scam - (50 - scam)) <= 2                    # баланс ~50/50
    assert {s.lang for s in sk} == {"ru", "en", "zh"}
    for s in sk:
        assert TMIN <= s.n_turns <= TMAX
        if s.is_scam:
            assert 1 <= s.onset_cand <= s.n_turns - 2      # место под эскалацию после t*
        else:
            assert s.onset_cand is None                    # benign -> ∅ уже на стадии скелета


def test_skeletons_twin_pairs_linked():
    sk = build_skeletons(50, ["ru", "en", "zh"], TMIN, TMAX, seed=13)
    by_sid = {s.sid: s for s in sk}
    linked = [s for s in sk if s.twin_sid]
    assert linked, "должны существовать пары скам<->benign-двойник"
    for s in linked:
        twin = by_sid[s.twin_sid]
        assert twin.twin_sid == s.sid                      # связь двусторонняя
        assert twin.is_scam != s.is_scam
        assert twin.lang == s.lang                         # двойник на том же языке
    # детерминированность по seed
    assert [s.sid for s in sk] == [s.sid for s in build_skeletons(50, ["ru", "en", "zh"],
                                                                  TMIN, TMAX, seed=13)]


def test_prompts_carry_invariants():
    scam_skel = _skel(is_scam=1, onset_cand=3)
    system, user = build_prompt(scam_skel)
    assert "(URL)" in system and "(CODE)" in system        # плейсхолдеры вшиты
    assert "atk" in user and "speaker" in user             # разделение сторон вшито
    assert "реплика 3" in user                             # онсет-кандидат вшит
    _, user_b = build_prompt(_skel(is_scam=0))
    assert "onset = null" in user_b or "onset\": null" in user_b


# --- парсер ------------------------------------------------------------------------

def test_extract_json_robust():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Вот JSON:\n```json\n{"a": 1}\n```\nготово') == {"a": 1}
    assert extract_json('преамбула {"a": {"b": 2}} хвост') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("никакого json тут нет")


# --- валидация инвариантов ---------------------------------------------------------

def test_valid_two_party_dialogue_passes():
    skel = _skel(is_scam=1)
    assert _validate(skel, _payload(skel)) == []


def test_benign_valid_and_benign_with_onset_rejected():
    skel = _skel(is_scam=0)
    assert _validate(skel, _payload(skel)) == []           # benign -> onset None проходит
    bad = _payload(skel, onset=2)
    assert any("benign с онсетом" in r for r in _validate(skel, bad))


def test_scam_onset_must_hit_atk_turn():
    skel = _skel(is_scam=1)
    bad = _payload(skel, onset=2)                          # реплика 2 -> tgt
    assert any("не на atk-реплике" in r for r in _validate(skel, bad))
    assert _validate(skel, _payload(skel, onset=3)) == []  # реплика 3 -> atk


def test_scam_without_onset_or_subtypes_rejected():
    skel = _skel(is_scam=1)
    assert any("невалидным онсетом" in r for r in _validate(skel, _payload(skel, onset=None)))
    assert any("консистентности" in r for r in _validate(skel, _payload(skel, subtypes=[])))


def test_three_speakers_rejected():
    skel = _skel(is_scam=1)
    bad = _payload(skel)
    bad["turns"][3]["speaker"] = "Оператор №2"             # третья сторона (та же role tgt)
    assert any("должно быть ровно 2" in r for r in _validate(skel, bad))


def test_speaker_switching_role_rejected():
    skel = _skel(is_scam=1)
    bad = _payload(skel)
    bad["turns"][3]["role"] = "atk"                        # «Клиент» вдруг стал атакующим
    assert any("сменил role" in r for r in _validate(skel, bad))


def test_length_bounds_enforced():
    skel = _skel(is_scam=1, n_turns=8)
    assert any("длина 5" in r for r in _validate(skel, _payload(skel, T=5)))
    assert any("длина 15" in r for r in _validate(skel, _payload(skel, T=15)))


def test_label_drift_vs_skeleton_rejected():
    skel = _skel(is_scam=1)
    bad = _payload(skel)
    bad["draft_labels"]["is_scam"] = 0                     # генератор «передумал»
    assert any("is_scam черновика" in r for r in _validate(skel, bad))


# --- пайплайн на моках -------------------------------------------------------------

def _pipeline_cfg(tmp_path):
    return {
        "generation": {"models": ["fake-model"],
                       "api": {"max_attempts": 1},
                       "turns": {"min": TMIN, "max": TMAX}},
        "paths": {"corpus": str(tmp_path)},
    }


def test_run_pipeline_writes_corpus_and_rejects(tmp_path):
    skels = build_skeletons(2, ["ru"], TMIN, TMAX, seed=13)   # [scam, его benign-двойник]
    bad = _payload(skels[1])
    bad["turns"][2]["speaker"] = "Третий лишний"
    client = FakeClient([_payload(skels[0]), bad])
    accepted, rejected, rep = run_pipeline(skels, _pipeline_cfg(tmp_path), client)

    assert len(accepted) == 1 and len(rejected) == 1
    assert accepted[0]["gold_src"] == "llm_draft"
    assert accepted[0]["item_id"] == 1 and accepted[0]["generator"] == "fake-model"
    assert rejected[0]["reasons"]

    lines = [json.loads(x) for x in (tmp_path / "pilot_v1.jsonl").read_text("utf-8").splitlines()]
    assert [r["id"] for r in lines] == [skels[0].sid]
    rej = [json.loads(x)
           for x in (tmp_path / "pilot_v1_rejected.jsonl").read_text("utf-8").splitlines()]
    assert rej[0]["id"] == skels[1].sid
    assert rep["n_accepted"] == 1 and rep["n_rejected"] == 1
    assert rep["multi_generator_ok"] is False              # пилот: 1 генератор — с пометкой


def test_run_pipeline_parses_fenced_json(tmp_path):
    skels = build_skeletons(1, ["en"], TMIN, TMAX, seed=13)
    fenced = "Вот диалог:\n```json\n" + json.dumps(_payload(skels[0]), ensure_ascii=False) + "\n```"
    accepted, rejected, _ = run_pipeline(skels, _pipeline_cfg(tmp_path), FakeClient([fenced]))
    assert len(accepted) == 1 and not rejected


def test_render_dialogue_shows_sides_and_onset(tmp_path):
    skels = build_skeletons(1, ["ru"], TMIN, TMAX, seed=13)
    accepted, _, _ = run_pipeline(skels, _pipeline_cfg(tmp_path), FakeClient([_payload(skels[0])]))
    text = render_dialogue(accepted[0])
    assert "[1] Банк (atk):" in text and "(tgt):" in text
    assert "← ОНСЕТ (черновик)" in text


def test_missing_api_key_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        AnthropicClient.from_env({})


# --- мульти-вендор: DeepSeek-клиент + роутер (§9.4) --------------------------------

def test_vendor_routing_by_model_name():
    assert vendor_of("claude-sonnet-5") == "anthropic"
    assert vendor_of("deepseek-chat") == "deepseek"
    assert vendor_of("deepseek-reasoner") == "deepseek"
    with pytest.raises(GenerationError, match="неизвестный вендор"):
        vendor_of("gpt-4o")


def test_deepseek_missing_key_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient.from_env({})


def test_build_client_reports_all_missing_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(GenerationError) as ei:
        build_client(["claude-sonnet-5", "deepseek-chat"])
    msg = str(ei.value)
    assert "ANTHROPIC_API_KEY" in msg and "DEEPSEEK_API_KEY" in msg


def test_build_client_routes_to_present_vendor(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-x")
    client = build_client(["claude-sonnet-5", "deepseek-chat"], {"max_tokens": 128})
    assert isinstance(client, RoutedClient)


def test_routed_client_dispatches_by_model():
    calls = []

    class Spy:
        def __init__(self, tag):
            self.tag = tag

        def complete(self, model, system, user):
            calls.append((self.tag, model))
            return f"{self.tag}:{model}"

    router = RoutedClient({"anthropic": Spy("A"), "deepseek": Spy("D")})
    assert router.complete("claude-sonnet-5", "s", "u") == "A:claude-sonnet-5"
    assert router.complete("deepseek-chat", "s", "u") == "D:deepseek-chat"
    assert calls == [("A", "claude-sonnet-5"), ("D", "deepseek-chat")]


@pytest.mark.parametrize("boom", [
    TimeoutError("read timed out"),                  # таймаут чтения
    ConnectionResetError("peer reset"),              # обрыв соединения
    ValueError("Expecting value: line 1 column 1"),  # битый JSON в ответе
])
def test_deepseek_transient_errors_wrapped_not_raw(monkeypatch, boom):
    """Транзиентный сбой -> GenerationError (его ретраит backoff), а не падение прогона."""
    def _boom(*a, **kw):
        raise boom
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(GenerationError, match="сбой соединения/ответа"):
        DeepSeekClient("k").complete("deepseek-chat", "s", "u")


def test_routed_client_missing_vendor_raises():
    router = RoutedClient({"anthropic": object()})     # deepseek-клиента нет
    with pytest.raises(GenerationError, match="нет клиента для вендора"):
        router.complete("deepseek-chat", "s", "u")
