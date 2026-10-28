"""Разметочные паки для золотой разметки (HTML, один файл, без сервера).

Слепота: в HTML попадают ТОЛЬКО текст диалога и роли (atk/tgt → A/B) —
никакие черновые метки (is_scam, subtypes, onset, hard_negative, scam_type)
и даже реальные id (они содержат имя сценария: corpus_en_phishing_0001).
Диалоги получают опакные pid в перемешанном порядке; соответствие
pid → реальный id живёт в манифесте рядом с паком, аннотатор его не открывает.

Валидация CSV аннотаций: полнота по манифесту, консистентность v1.2,
диапазон t* ∈ [1..T] и попадание онсета в реплику атакующего.
Чистый stdlib (конвенция проекта).
"""
from __future__ import annotations

import csv
import json
import pathlib
import random

SUBTYPES = ("phishing_link", "money_request", "other")
TEMPLATE_PATH = pathlib.Path(__file__).parent / "annotation_pack_template.html"

# поля корпуса, которые ЗАПРЕЩЕНО раскрывать аннотатору
BLIND_FIELDS = ("is_scam", "subtypes", "onset", "onset_cand", "hard_negative",
                "scam_type", "twin_id", "scenario_group", "gold_src",
                "generator", "other_note", "id", "item_id", "lang", "split")

DEFAULT_SAMPLE_SEED = 20260722    # выбор подвыборки (--n)
DEFAULT_SHUFFLE_SEED = 20260723   # порядок предъявления


def select_records(records, split: str, n: int | None = None,
                   sample_seed: int = DEFAULT_SAMPLE_SEED):
    """Диалоги сплита; при n — случайная подвыборка с фиксированным seed."""
    pool = [r for r in records if r.get("split") == split]
    if not pool:
        raise ValueError(f"в корпусе нет диалогов сплита {split!r}")
    if n is not None and n < len(pool):
        rng = random.Random(sample_seed)
        pool = rng.sample(pool, n)
    return pool


def build_pack_payload(records, split: str, n: int | None = None,
                       sample_seed: int = DEFAULT_SAMPLE_SEED,
                       shuffle_seed: int = DEFAULT_SHUFFLE_SEED):
    """(payload, manifest): слепой пейлоад для HTML + манифест с реальными id."""
    chosen = select_records(records, split, n, sample_seed)
    rng = random.Random(shuffle_seed)
    order = list(chosen)
    rng.shuffle(order)

    pack_id = f"{split}{len(order)}_s{shuffle_seed}"
    width = max(3, len(str(len(order))))
    dialogs, manifest_dialogs = [], []
    for k, r in enumerate(order, 1):
        pid = f"{split}-{k:0{width}d}"
        dialogs.append({
            "pid": pid,
            "turns": [{"i": t["i"], "role": t["role"], "speaker": t.get("speaker", ""),
                       "text": t["text"]} for t in r["turns"]],
        })
        manifest_dialogs.append({
            "pid": pid,
            "dialog_id": r["id"],
            "T": len(r["turns"]),
            "atk_turns": [t["i"] for t in r["turns"] if t["role"] == "atk"],
        })

    payload = {"pack_id": pack_id, "split": split, "dialogs": dialogs}
    manifest = {"pack_id": pack_id, "split": split, "n": len(order),
                "sample_seed": sample_seed, "shuffle_seed": shuffle_seed,
                "dialogs": manifest_dialogs}
    return payload, manifest


def render_pack_html(payload, template_path: pathlib.Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    title = f"Разметка: пак {payload['pack_id']} ({len(payload['dialogs'])} диалогов)"
    html = template.replace("__TITLE__", title)
    blob = json.dumps(payload, ensure_ascii=False)
    # защита от преждевременного закрытия <script> текстом диалога
    blob = blob.replace("</", "<\\/")
    return html.replace("__PACK_JSON__", blob)


def check_blindness(payload) -> list[str]:
    """Ключи записей пейлоада, которых там быть не должно (пусто = чисто)."""
    leaked = set()
    for d in payload["dialogs"]:
        leaked |= set(d) & set(BLIND_FIELDS)
        for t in d["turns"]:
            leaked |= set(t) & set(BLIND_FIELDS)
    return sorted(leaked)


# ── валидация CSV аннотаций ──────────────────────────────────────────────

EXPECTED_COLUMNS = ["dialog_id", "is_scam", "subtypes", "onset", "hard_negative",
                    "doubt", "note", "time_spent_sec", "annotated_at"]
# необязательная колонка: режим просмотра на момент разметки —
# "seq" (последовательное раскрытие, инвариант 5) | "full" (весь диалог сразу).
# Отсутствует в CSV, экспортированных до появления режима.
VIEW_MODES = ("seq", "full")


def validate_annotations(rows: list[dict], manifest: dict,
                         allow_partial: bool = False) -> dict:
    """Проверки CSV против манифеста пака. Возвращает {"pass": bool, "checks": {...}}.

    Консистентность v1.2: scam → t* заполнен и >= 1 подтип;
    benign → подтипы пусты, t* пуст; hard_negative только у benign.
    """
    by_pid = {d["pid"]: d for d in manifest["dialogs"]}
    checks: dict[str, dict] = {}
    errors: list[str] = []

    ids = [r.get("dialog_id", "") for r in rows]
    unknown = sorted(set(ids) - set(by_pid))
    dups = sorted({i for i in ids if ids.count(i) > 1})
    missing = sorted(set(by_pid) - set(ids))
    checks["ids"] = {
        "pass": not unknown and not dups and (allow_partial or not missing),
        "unknown": unknown, "duplicates": dups,
        "missing": missing if not allow_partial else [],
        "n_annotated": len(ids), "n_pack": len(by_pid),
    }

    for r in rows:
        pid = r.get("dialog_id", "?")
        d = by_pid.get(pid)
        if d is None:
            continue

        if r.get("is_scam") not in ("0", "1"):
            errors.append(f"{pid}: is_scam={r.get('is_scam')!r} вне {{0,1}}")
            continue
        scam = r["is_scam"] == "1"
        subs = [s for s in (r.get("subtypes") or "").split("|") if s]
        bad_subs = sorted(set(subs) - set(SUBTYPES))
        if bad_subs:
            errors.append(f"{pid}: неизвестные подтипы {bad_subs}")
        onset_raw = (r.get("onset") or "").strip()

        if scam:
            if not onset_raw:
                errors.append(f"{pid}: scam без onset (нарушение v1.2)")
            if not subs:
                errors.append(f"{pid}: scam без подтипов (нарушение v1.2)")
            if r.get("hard_negative") not in ("", "0", None):
                errors.append(f"{pid}: hard_negative у scam (нарушение v1.2)")
        else:
            if subs:
                errors.append(f"{pid}: подтипы у benign (запрещённая комбинация v1.2)")
            if onset_raw:
                errors.append(f"{pid}: onset у benign (benign → t*=∅)")
            if r.get("hard_negative") not in ("0", "1"):
                errors.append(f"{pid}: hard_negative={r.get('hard_negative')!r} вне {{0,1}}")

        if onset_raw:
            try:
                t = int(onset_raw)
            except ValueError:
                errors.append(f"{pid}: onset={onset_raw!r} не целое")
            else:
                if not 1 <= t <= d["T"]:
                    errors.append(f"{pid}: onset={t} вне [1..{d['T']}]")
                elif t not in d["atk_turns"]:
                    errors.append(f"{pid}: onset={t} не реплика атакующего")

        if r.get("doubt") not in ("0", "1"):
            errors.append(f"{pid}: doubt={r.get('doubt')!r} вне {{0,1}}")
        ts = r.get("time_spent_sec", "")
        try:
            if float(ts) < 0:
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"{pid}: time_spent_sec={ts!r} не число >= 0")

    checks["labels"] = {"pass": not errors, "errors": errors}

    # режим просмотра: не влияет на pass, но full-text метки не являются
    # префиксной верификацией онсета (инвариант 5) — их надо видеть в отчёте
    modes = [(r.get("view_mode") or "").strip() for r in rows]
    bad_modes = sorted({m for m in modes if m and m not in VIEW_MODES})
    checks["view_mode"] = {
        "pass": not bad_modes,
        "errors": [f"неизвестный view_mode {m!r}" for m in bad_modes],
        "n_full": sum(1 for m in modes if m == "full"),
        "n_seq": sum(1 for m in modes if m == "seq"),
        "n_unknown": sum(1 for m in modes if not m),
    }

    return {"pass": all(c["pass"] for c in checks.values()), "checks": checks}


def read_annotations_csv(path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in EXPECTED_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"в CSV нет колонок: {missing}")
        return list(reader)


def resolve_dialog_ids(rows: list[dict], manifest: dict) -> list[dict]:
    """Копия строк с pid, заменёнными на реальные id корпуса (для merge)."""
    pid2id = {d["pid"]: d["dialog_id"] for d in manifest["dialogs"]}
    out = []
    for r in rows:
        rr = dict(r)
        rr["dialog_id"] = pid2id.get(r["dialog_id"], r["dialog_id"])
        out.append(rr)
    return out
