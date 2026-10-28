"""Тесты стадии-2 верификации (чистые функции, без интерактива)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import verify_labels as vl  # noqa: E402


def _rec(item_id=1, is_scam=1, subtypes=None, onset=2, roles=("atk", "atk", "tgt")):
    turns = [{"i": i + 1, "role": r, "speaker": "S", "text": f"t{i + 1}"} for i, r in enumerate(roles)]
    return {
        "item_id": item_id, "id": f"d{item_id}", "lang": "ru", "scam_type": "phishing",
        "turns": turns, "is_scam": is_scam,
        "subtypes": subtypes if subtypes is not None else ["phishing_link"],
        "onset": onset,
    }


# ─── применение правки к записи ───

def test_make_verified_record_overwrites_and_stamps():
    rec = _rec(is_scam=1, subtypes=["phishing_link"], onset=2)
    out = vl.make_verified_record(rec, {"is_scam": 1, "subtypes": ["money_request"], "onset": 1},
                                  "alice", "onset 2→1")
    assert out["gold_src"] == "human_verified"
    assert out["verified_by"] == "alice"
    assert out["onset"] == 1
    assert out["subtypes"] == ["money_request"]
    # снимок черновика сохранён
    assert out["draft_labels"] == {"is_scam": 1, "subtypes": ["phishing_link"], "onset": 2}
    # исходные поля не связанные с меткой уцелели
    assert out["id"] == rec["id"] and out["turns"] == rec["turns"]


def test_confirm_no_edit_note():
    rec = _rec(is_scam=0, subtypes=[], onset=None)
    d = vl.draft_of(rec)
    assert vl.diff_note(d, d) == "подтверждено без правок"
    assert vl.change_types(d, d) == set()


def test_diff_note_and_change_types():
    draft = {"is_scam": 1, "subtypes": ["phishing_link"], "onset": 3}
    verified = {"is_scam": 1, "subtypes": ["money_request", "phishing_link"], "onset": 2}
    assert vl.change_types(draft, verified) == {"onset", "subtype"}
    note = vl.diff_note(draft, verified)
    assert "onset 3→2" in note and "subtypes" in note and "is_scam" not in note


def test_subtypes_order_insensitive():
    a = {"is_scam": 1, "subtypes": ["money_request", "phishing_link"], "onset": 2}
    b = {"is_scam": 1, "subtypes": ["phishing_link", "money_request"], "onset": 2}
    assert vl.change_types(a, b) == set()


# ─── продолжение с прерванного места ───

def test_pending_skips_done():
    records = [_rec(1), _rec(2), _rec(3)]
    verified = [{"item_id": 2}]
    todo = vl.pending_records(records, vl.done_item_ids(verified))
    assert [r["item_id"] for r in todo] == [1, 3]


def test_resume_roundtrip(tmp_path):
    out = tmp_path / "verified.jsonl"
    rec = vl.make_verified_record(_rec(1), vl.draft_of(_rec(1)), "bob", "ok")
    vl.append_jsonl(out, rec)
    loaded = vl.read_jsonl(out)
    assert vl.done_item_ids(loaded) == {1}
    # второй запуск: запись 1 уже не в todo
    assert vl.pending_records([_rec(1), _rec(2)], vl.done_item_ids(loaded)) == [_rec(2)]


def test_read_jsonl_missing_file(tmp_path):
    assert vl.read_jsonl(tmp_path / "nope.jsonl") == []


# ─── валидация onset на atk-реплику ───

def test_validate_onset_on_atk_ok():
    turns = _rec(roles=("atk", "atk", "tgt"))["turns"]
    ok, warn = vl.validate_onset(1, turns)
    assert ok and warn is None


def test_validate_onset_on_tgt_warns_but_ok():
    turns = _rec(roles=("atk", "atk", "tgt"))["turns"]
    ok, warn = vl.validate_onset(3, turns)  # tgt-реплика
    assert ok and warn and "tgt" in warn


def test_validate_onset_out_of_range_blocks():
    turns = _rec(roles=("atk", "atk", "tgt"))["turns"]
    ok, warn = vl.validate_onset(5, turns)
    assert not ok and "вне диапазона" in warn


def test_validate_onset_benign_empty_ok():
    turns = _rec()["turns"]
    assert vl.validate_onset(None, turns) == (True, None)
    assert vl.validate_onset("", turns) == (True, None)


# ─── консистентность (инварианты 1/3/6) ───

def test_consistency_benign_with_onset_warns():
    turns = _rec()["turns"]
    w = vl.consistency_warnings({"is_scam": 0, "subtypes": [], "onset": 2}, turns)
    assert any("инвариант 3" in x for x in w)


def test_consistency_scam_without_onset_warns():
    turns = _rec()["turns"]
    w = vl.consistency_warnings({"is_scam": 1, "subtypes": ["phishing_link"], "onset": None}, turns)
    assert any("онсет пуст" in x for x in w)


def test_consistency_clean_scam_no_warn():
    turns = _rec(roles=("atk", "atk", "tgt"))["turns"]
    assert vl.consistency_warnings({"is_scam": 1, "subtypes": ["phishing_link"], "onset": 1}, turns) == []


# ─── подсчёт согласия human-vs-draft ───

def test_agreement_all_confirmed():
    recs = []
    for i in range(4):
        r = _rec(i, is_scam=1, subtypes=["phishing_link"], onset=2)
        recs.append(vl.make_verified_record(r, vl.draft_of(r), "x", "ok"))
    ag = vl.agreement_stats(recs)
    assert ag["n"] == 4 and ag["full_frac"] == 1.0
    assert ag["field_frac"] == {"is_scam": 1.0, "subtypes": 1.0, "onset": 1.0}


def test_agreement_partial():
    r0 = _rec(0, is_scam=1, subtypes=["phishing_link"], onset=2)
    keep = vl.make_verified_record(r0, vl.draft_of(r0), "x", "ok")
    r1 = _rec(1, is_scam=1, subtypes=["phishing_link"], onset=2)
    edited = vl.make_verified_record(r1, {"is_scam": 1, "subtypes": ["phishing_link"], "onset": 1}, "x", "onset 2→1")
    ag = vl.agreement_stats([keep, edited])
    assert ag["n"] == 2
    assert ag["full_frac"] == 0.5           # одна запись правилась
    assert ag["field_frac"]["onset"] == 0.5  # onset разошёлся в одной
    assert ag["field_frac"]["is_scam"] == 1.0


def test_agreement_ignores_records_without_draft():
    ag = vl.agreement_stats([{"item_id": 9, "is_scam": 1}])
    assert ag["n"] == 0 and ag["full_frac"] == 0.0


# ─── батч-применение суждений (авто-проход) ───

def test_make_verified_record_gold_src_override():
    rec = _rec(1)
    out = vl.make_verified_record(rec, vl.draft_of(rec), "llm", "ok", gold_src="llm_prefix_verified")
    assert out["gold_src"] == "llm_prefix_verified"


def test_apply_judgments_writes_provenance_and_skips_unjudged():
    records = [_rec(1, onset=2, roles=("atk", "atk", "tgt")),
               _rec(2, is_scam=0, subtypes=[], onset=None, roles=("tgt", "atk", "tgt"))]
    judgments = [{"item_id": 1, "is_scam": 1, "subtypes": ["phishing_link"], "onset": 1}]
    out = vl.apply_judgments(records, judgments, "llm", "llm_prefix_verified", note_prefix="auto")
    assert len(out) == 1                       # запись 2 без суждения пропущена
    assert out[0]["gold_src"] == "llm_prefix_verified"
    assert out[0]["onset"] == 1
    assert out[0]["verifier_note"].startswith("auto: ")
    assert out[0]["draft_labels"]["onset"] == 2  # снимок черновика сохранён


def test_apply_judgments_dict_form():
    records = [_rec(5, roles=("atk", "atk", "tgt"))]
    out = vl.apply_judgments(records, {5: {"is_scam": 1, "subtypes": ["phishing_link"], "onset": 2}},
                             "llm", "llm_prefix_verified")
    assert out[0]["item_id"] == 5 and out[0]["onset"] == 2


def test_apply_judgments_rejects_out_of_range_onset():
    records = [_rec(1, roles=("atk", "atk", "tgt"))]
    import pytest
    with pytest.raises(ValueError):
        vl.apply_judgments(records, [{"item_id": 1, "is_scam": 1, "subtypes": [], "onset": 9}],
                           "llm", "llm_prefix_verified")
