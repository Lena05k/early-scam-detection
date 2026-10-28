"""Тесты разметочных паков: слепота, детерминизм, манифест, валидация CSV."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from scam_onset.data import annotation_pack as ap

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "corpus_v1.jsonl"


def _mk_records(n=12, split="test"):
    recs = []
    for k in range(n):
        turns = [
            {"i": 1, "role": "tgt", "speaker": "B", "text": f"привет {k}"},
            {"i": 2, "role": "atk", "speaker": "A", "text": f"это банк {k}"},
            {"i": 3, "role": "atk", "speaker": "A", "text": f"переведите {k}"},
        ]
        recs.append({
            "id": f"corpus_ru_phishing_{k:04d}", "lang": "ru",
            "scam_type": "phishing", "is_scam": 1, "subtypes": ["money_request"],
            "onset": 2, "hard_negative": 0, "split": split, "turns": turns,
            "twin_id": "", "scenario_group": f"sg_{k}", "gold_src": "llm_draft",
        })
    return recs


# ── генерация пака ──

def test_payload_is_blind():
    payload, _ = ap.build_pack_payload(_mk_records(), "test")
    assert ap.check_blindness(payload) == []
    for d in payload["dialogs"]:
        assert set(d) == {"pid", "turns"}
        for t in d["turns"]:
            assert set(t) == {"i", "role", "speaker", "text"}


def test_html_hides_real_ids_and_labels():
    payload, _ = ap.build_pack_payload(_mk_records(), "test")
    html = ap.render_pack_html(payload)
    assert "__PACK_JSON__" not in html and "__TITLE__" not in html
    assert "corpus_ru_phishing" not in html          # реальные id скрыты
    assert '"scam_type"' not in html and '"gold_src"' not in html
    assert '"hard_negative":' not in html            # черновых меток нет


def test_script_close_tag_escaped():
    recs = _mk_records(3)
    recs[0]["turns"][0]["text"] = "опасный текст </script><b>x"
    html = ap.render_pack_html(ap.build_pack_payload(recs, "test")[0])
    assert "</script><b>x" not in html


def test_shuffle_and_sample_deterministic():
    recs = _mk_records(30)
    p1, m1 = ap.build_pack_payload(recs, "test", n=10)
    p2, m2 = ap.build_pack_payload(recs, "test", n=10)
    assert [d["dialog_id"] for d in m1["dialogs"]] == \
           [d["dialog_id"] for d in m2["dialogs"]]
    assert len(p1["dialogs"]) == 10
    # другой shuffle_seed — другой порядок
    _, m3 = ap.build_pack_payload(recs, "test", n=10, shuffle_seed=1)
    assert [d["dialog_id"] for d in m3["dialogs"]] != \
           [d["dialog_id"] for d in m1["dialogs"]]


def test_shuffle_changes_presentation_order():
    recs = _mk_records(30)
    _, m = ap.build_pack_payload(recs, "test")
    assert [d["dialog_id"] for d in m["dialogs"]] != [r["id"] for r in recs]


def test_manifest_matches_payload():
    payload, manifest = ap.build_pack_payload(_mk_records(), "test")
    assert [d["pid"] for d in payload["dialogs"]] == \
           [d["pid"] for d in manifest["dialogs"]]
    for d in manifest["dialogs"]:
        assert d["T"] == 3
        assert d["atk_turns"] == [2, 3]
    assert manifest["n"] == len(payload["dialogs"])
    assert manifest["sample_seed"] == ap.DEFAULT_SAMPLE_SEED


def test_split_filter_and_missing_split():
    recs = _mk_records(5, split="train") + _mk_records(3, split="test")
    # id должны быть уникальны между сплитами
    for i, r in enumerate(recs):
        r["id"] = f"corpus_x_{i:04d}"
        r["scenario_group"] = f"sg_x_{i}"
    payload, _ = ap.build_pack_payload(recs, "train")
    assert len(payload["dialogs"]) == 5
    with pytest.raises(ValueError):
        ap.build_pack_payload(recs, "val")


# ── валидация CSV ──

def _manifest():
    _, m = ap.build_pack_payload(_mk_records(3), "test")
    return m


def _row(pid, **kw):
    row = {"dialog_id": pid, "is_scam": "1", "subtypes": "money_request",
           "onset": "2", "hard_negative": "0", "doubt": "0", "note": "",
           "time_spent_sec": "12.5", "annotated_at": "2026-07-21T10:00:00Z"}
    row.update(kw)
    return row


def _benign(pid, **kw):
    base = {"is_scam": "0", "subtypes": "", "onset": ""}
    base.update(kw)
    return _row(pid, **base)


def test_valid_full_annotation_passes():
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    assert ap.validate_annotations(rows, m)["pass"]


def test_incomplete_fails_unless_partial():
    m = _manifest()
    rows = [_row(m["dialogs"][0]["pid"])]
    assert not ap.validate_annotations(rows, m)["pass"]
    assert ap.validate_annotations(rows, m, allow_partial=True)["pass"]


def test_unknown_and_duplicate_ids():
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    assert not ap.validate_annotations(rows + [_row("nope-1")], m)["pass"]
    assert not ap.validate_annotations(rows + [rows[0]], m)["pass"]


@pytest.mark.parametrize("mut", [
    {"onset": ""},                                # scam без onset
    {"subtypes": ""},                             # scam без подтипов
    {"hard_negative": "1"},                       # hard_neg у scam
    {"onset": "9"},                               # вне [1..T]
    {"onset": "1"},                               # реплика tgt, не атакующего
    {"onset": "abc"},
    {"subtypes": "money_request|bogus"},
    {"doubt": "5"},
    {"time_spent_sec": "-1"},
])
def test_v12_violations_detected_scam(mut):
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    rows[0] = _row(m["dialogs"][0]["pid"], **mut)
    res = ap.validate_annotations(rows, m)
    assert not res["pass"], mut


@pytest.mark.parametrize("mut", [
    {"subtypes": "money_request"},                # подтип у benign — запрещённая комбинация
    {"onset": "2"},                               # onset у benign
    {"hard_negative": ""},                        # hard_negative обязателен у benign
])
def test_v12_violations_detected_benign(mut):
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    rows[0] = _benign(m["dialogs"][0]["pid"], **mut)
    res = ap.validate_annotations(rows, m)
    assert not res["pass"], mut


def test_benign_hard_negative_ok():
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    rows[0] = _benign(m["dialogs"][0]["pid"], hard_negative="1")
    assert ap.validate_annotations(rows, m)["pass"]


def test_resolve_dialog_ids():
    m = _manifest()
    rows = [_row(m["dialogs"][0]["pid"])]
    resolved = ap.resolve_dialog_ids(rows, m)
    assert resolved[0]["dialog_id"] == m["dialogs"][0]["dialog_id"]
    assert resolved[0]["dialog_id"].startswith("corpus_")


# ── режим просмотра (full / seq) ──

def test_view_mode_counted_and_not_required():
    m = _manifest()
    pids = [d["pid"] for d in m["dialogs"]]
    rows = [_row(pids[0], view_mode="full"), _row(pids[1], view_mode="seq"),
            _row(pids[2])]                      # старый CSV — колонки нет
    res = ap.validate_annotations(rows, m)
    assert res["pass"]
    vm = res["checks"]["view_mode"]
    assert (vm["n_full"], vm["n_seq"], vm["n_unknown"]) == (1, 1, 1)


def test_unknown_view_mode_fails():
    m = _manifest()
    rows = [_row(d["pid"]) for d in m["dialogs"]]
    rows[0]["view_mode"] = "peeked"
    res = ap.validate_annotations(rows, m)
    assert not res["pass"] and not res["checks"]["view_mode"]["pass"]


def test_template_exports_view_mode_and_full_default():
    tpl = ap.TEMPLATE_PATH.read_text(encoding="utf-8")
    assert '"view_mode"' in tpl                    # колонка в экспорте CSV
    assert "full_view: true" in tpl                # полный показ — режим по умолчанию


# ── сохранение и продолжение разметки ──

def test_template_has_progress_save_and_load():
    tpl = ap.TEMPLATE_PATH.read_text(encoding="utf-8")
    for marker in ("annpack-progress/1",      # формат файла прогресса
                   "function saveProgress",   # выгрузка состояния в файл
                   "function loadProgressText",
                   "function mergeAnnotations",
                   "function parseCSV",       # CSV тоже принимается как источник
                   'id="file-input"',
                   'id="resume-banner"',      # «продолжаем с места остановки»
                   'id="storage-warn"'):      # localStorage недоступен → предупредить
        assert marker in tpl, marker


def test_progress_file_carries_unfinished_work():
    """Файл прогресса — это всё состояние, а не только готовые метки (как CSV)."""
    tpl = ap.TEMPLATE_PATH.read_text(encoding="utf-8")
    start = tpl.index("function progressJSON")
    body = tpl[start:tpl.index("function saveProgress")]
    assert "annotations: state.annotations" in body and "meta: state.meta" in body
    # экспорт CSV, наоборот, отдаёт только завершённые
    csv_body = tpl[tpl.index("function exportCSVText"):tpl.index("function doExport")]
    assert "!an.done" in csv_body


def test_read_csv_requires_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("dialog_id,is_scam\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ap.read_annotations_csv(p)


# ── слепота на реальном корпусе ──

def test_real_pack_is_blind():
    if not CORPUS.exists():
        pytest.skip("corpus_v1.jsonl отсутствует")
    with open(CORPUS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if "split" not in records[0]:
        pytest.skip("нет поля split")
    payload, manifest = ap.build_pack_payload(records, "test")
    assert ap.check_blindness(payload) == []
    html = ap.render_pack_html(payload)
    # ни одного реального id (содержат имя сценария) и ни одной черновой метки
    assert "corpus_en_" not in html and "corpus_ru_" not in html \
        and "corpus_zh_" not in html
    assert '"is_scam":' not in html and '"scenario_group"' not in html
    assert len(payload["dialogs"]) == 403
