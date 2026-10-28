#!/usr/bin/env python3
"""Человеческая верификация черновых меток — стадия 2 конвейера.

Читает диалоги с ЧЕРНОВЫМИ метками (напр. data/corpus/pilot_v1.jsonl), проводит
человека по каждому диалогу (реплики по префиксам, стороны разделены), даёт
подтвердить/исправить is_scam / подтипы / онсет и сохраняет результат с
gold_src="human_verified".

Пример:
  python scripts/verify_labels.py
  python scripts/verify_labels.py --in data/corpus/pilot_v1.jsonl --out data/corpus/pilot_v1_verified.jsonl

ВАЖНО (инварианты проекта):
  - Онсет = первая atk-реплика с ТЕКСТОВО наблюдаемым обманом (гайдлайн v1.2).
    Судить по ПРЕФИКСУ C_≤t (без заглядывания вперёд) — иначе hindsight bias (инвариант 5).
  - benign → онсет пусто (инвариант 3). Подтип/онсет ⇒ is_scam=1 (правило консистентности v1.2).
  - Прогресс сохраняется ПОСЛЕ КАЖДОГО диалога; повторный запуск продолжает с непроверенных.
"""
import argparse
import json
import pathlib
import sys

SUBTYPES = ["phishing_link", "money_request", "other"]
DEFAULT_IN = "data/corpus/pilot_v1.jsonl"
DEFAULT_OUT = "data/corpus/pilot_v1_verified.jsonl"


# ─────────────────────────── чистые функции (тестируемые) ───────────────────────────

def read_jsonl(path):
    """Прочитать jsonl → список записей. Отсутствующий файл → []."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, rec):
    """Дописать одну запись в jsonl (создаёт файл и родителя при необходимости)."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def draft_of(rec):
    """Снимок черновых меток записи (нормализованный: subtypes отсортированы)."""
    return {
        "is_scam": rec.get("is_scam"),
        "subtypes": sorted(rec.get("subtypes") or []),
        "onset": rec.get("onset"),
    }


def norm_labels(labels):
    """Нормализовать метки к сравнимому виду (subtypes sorted, onset None вместо '')."""
    onset = labels.get("onset")
    if onset == "":
        onset = None
    return {
        "is_scam": labels.get("is_scam"),
        "subtypes": sorted(labels.get("subtypes") or []),
        "onset": onset,
    }


def done_item_ids(verified_records):
    """item_id уже финализированных (verified) записей."""
    return {r["item_id"] for r in verified_records}


def pending_records(records, done_ids):
    """Записи, которых ещё нет в verified (по item_id), в исходном порядке."""
    return [r for r in records if r["item_id"] not in done_ids]


def validate_onset(onset, turns):
    """Проверить онсет против диапазона и роли реплики.

    Возвращает (ok, warning): ok=False блокирует (вне диапазона); warning —
    мягкое предупреждение (онсет не на atk-реплике), человек решает сам.
    """
    if onset is None or onset == "":
        return True, None
    if not isinstance(onset, int) or not (1 <= onset <= len(turns)):
        return False, f"онсет {onset!r} вне диапазона 1..{len(turns)}"
    role = turns[onset - 1].get("role")
    if role != "atk":
        return True, f"⚠ онсет на реплике стороны '{role}', ожидается 'atk'"
    return True, None


def consistency_warnings(labels, turns):
    """Мягкие предупреждения по правилам консистентности v1.2 (инварианты 1/3/6)."""
    warns = []
    lab = norm_labels(labels)
    if lab["is_scam"] == 0:
        if lab["subtypes"]:
            warns.append(f"⚠ is_scam=0, но заданы подтипы {lab['subtypes']} (benign не должен иметь подтипов)")
        if lab["onset"] is not None:
            warns.append(f"⚠ is_scam=0, но онсет={lab['onset']} (benign → онсет пусто, инвариант 3)")
    if lab["is_scam"] == 1:
        if lab["onset"] is None:
            warns.append("⚠ is_scam=1, но онсет пуст (скам должен иметь онсет-реплику)")
        if not lab["subtypes"]:
            warns.append("⚠ is_scam=1, но не задан ни один подтип")
    ok, w = validate_onset(lab["onset"], turns)
    if not ok:
        warns.append(f"⚠ {w}")
    elif w:
        warns.append(w)
    return warns


def diff_note(draft, verified):
    """Человекочитаемая заметка о правках draft→verified."""
    d, v = norm_labels(draft), norm_labels(verified)
    parts = []
    if d["is_scam"] != v["is_scam"]:
        parts.append(f"is_scam {d['is_scam']}→{v['is_scam']}")
    if d["subtypes"] != v["subtypes"]:
        parts.append(f"subtypes {d['subtypes']}→{v['subtypes']}")
    if d["onset"] != v["onset"]:
        parts.append(f"onset {d['onset']}→{v['onset']}")
    return "; ".join(parts) if parts else "подтверждено без правок"


def change_types(draft, verified):
    """Множество типов правок: {'is_scam','onset','subtype'}. Пусто = без правок."""
    d, v = norm_labels(draft), norm_labels(verified)
    types = set()
    if d["is_scam"] != v["is_scam"]:
        types.add("is_scam")
    if d["onset"] != v["onset"]:
        types.add("onset")
    if d["subtypes"] != v["subtypes"]:
        types.add("subtype")
    return types


def make_verified_record(rec, verified_labels, verified_by, note, gold_src="human_verified"):
    """Собрать verified-запись: исходная + verified-метки + провенанс + снимок черновика.

    gold_src="human_verified" — человеческая верификация (стадия 2, интерактив).
    gold_src="llm_prefix_verified" — авто-проход LLM по префиксам (independent-проверка
    онсетов; НЕ заменяет человека, служит сигналом расхождения с черновиком).
    """
    out = dict(rec)
    v = norm_labels(verified_labels)
    out["draft_labels"] = draft_of(rec)
    out["is_scam"] = v["is_scam"]
    out["subtypes"] = v["subtypes"]
    out["onset"] = v["onset"]
    out["gold_src"] = gold_src
    out["verified_by"] = verified_by
    out["verifier_note"] = note
    return out


def apply_judgments(records, judgments, verified_by, gold_src, note_prefix=""):
    """Собрать verified-записи из батча суждений (для не-интерактивного прохода).

    judgments: список/словарь {item_id: {is_scam, subtypes, onset}}. Возвращает список
    verified-записей в порядке records. item_id без суждения пропускается.
    Каждой записи прогоняется валидация онсета; при блокирующей ошибке — ValueError.
    """
    by_id = {j["item_id"]: j for j in judgments} if isinstance(judgments, list) else judgments
    out = []
    for rec in records:
        j = by_id.get(rec["item_id"])
        if j is None:
            continue
        labels = norm_labels(j)
        ok, warn = validate_onset(labels["onset"], rec["turns"])
        if not ok:
            raise ValueError(f"item_id={rec['item_id']}: {warn}")
        note = diff_note(draft_of(rec), labels)
        if note_prefix:
            note = f"{note_prefix}: {note}"
        out.append(make_verified_record(rec, labels, verified_by, note, gold_src=gold_src))
    return out


def agreement_stats(verified_records):
    """Доля согласия human-vs-llm_draft по финализированным записям.

    Требует наличия draft_labels в записи (наши verified-записи его хранят).
    """
    n = 0
    full = 0
    per_field = {"is_scam": 0, "subtypes": 0, "onset": 0}
    for r in verified_records:
        draft = r.get("draft_labels")
        if draft is None:
            continue
        n += 1
        v = norm_labels(r)
        d = norm_labels(draft)
        if v["is_scam"] == d["is_scam"]:
            per_field["is_scam"] += 1
        if v["subtypes"] == d["subtypes"]:
            per_field["subtypes"] += 1
        if v["onset"] == d["onset"]:
            per_field["onset"] += 1
        if v == d:
            full += 1
    return {
        "n": n,
        "full_agree": full,
        "full_frac": (full / n) if n else 0.0,
        "field_frac": {k: (c / n if n else 0.0) for k, c in per_field.items()},
    }


def format_dialogue(rec, reveal_to=None):
    """Читаемый рендер диалога, реплики по префиксам, стороны разделены.

    reveal_to=None → показать все реплики; иначе показать только 1..reveal_to.
    Плейсхолдеры в тексте не трогаем.
    """
    turns = rec["turns"]
    upto = len(turns) if reveal_to is None else min(reveal_to, len(turns))
    lines = []
    for t in turns[:upto]:
        role = t.get("role", "?")
        speaker = t.get("speaker", "")
        lines.append(f"  [{t['i']}] {speaker} ({role}): {t['text']}")
    return "\n".join(lines)


# ─────────────────────────── интерактив ───────────────────────────

def _prompt(msg):
    return input(msg).strip()


def _edit_is_scam():
    """Диалог правки is_scam + подтипов. Возвращает (is_scam, subtypes)."""
    while True:
        v = _prompt("  is_scam (0/1): ")
        if v in ("0", "1"):
            is_scam = int(v)
            break
        print("    введите 0 или 1")
    subtypes = []
    if is_scam == 1:
        print(f"    подтипы из {SUBTYPES} (мультилейбл), через запятую; пусто = ни одного")
        while True:
            raw = _prompt("  subtypes: ")
            if not raw:
                subtypes = []
                break
            cand = [s.strip() for s in raw.split(",") if s.strip()]
            bad = [s for s in cand if s not in SUBTYPES]
            if bad:
                print(f"    неизвестные подтипы: {bad}; допустимо {SUBTYPES}")
                continue
            # убрать дубликаты, сохранить порядок канона
            subtypes = [s for s in SUBTYPES if s in cand]
            break
    return is_scam, subtypes


def _edit_onset(turns):
    """Диалог правки онсета. Возвращает onset (int) или None."""
    while True:
        raw = _prompt(f"  onset (1..{len(turns)}, пусто = benign/∅): ")
        if raw == "":
            return None
        if not raw.isdigit():
            print("    введите число или пусто")
            continue
        val = int(raw)
        ok, warn = validate_onset(val, turns)
        if not ok:
            print(f"    {warn}")
            continue
        if warn:
            print(f"    {warn}")
            if _prompt("    всё равно принять? (y/N): ").lower() != "y":
                continue
        return val


def _show_labels(labels, turns, title):
    lab = norm_labels(labels)
    print(f"  {title}:")
    print(f"    is_scam : {lab['is_scam']}")
    print(f"    subtypes: {lab['subtypes'] or '—'}")
    if lab["onset"] is None:
        print("    onset   : ∅ (benign)")
    else:
        idx = lab["onset"]
        txt = turns[idx - 1]["text"] if 1 <= idx <= len(turns) else "?"
        role = turns[idx - 1].get("role", "?") if 1 <= idx <= len(turns) else "?"
        print(f"    onset   : [{idx}] ({role}) {txt}")


def _reveal_prefix(rec):
    """Пошаговое раскрытие реплик (инвариант 5: онсет без заглядывания вперёд).

    Возвращает выбранный человеком онсет (int) или None (обман не наблюдался в atk-репликах),
    либо строку 'quit'. Онсет фиксируется в момент первой atk-реплики с наблюдаемым обманом;
    после фиксации остаток диалога дораскрывается (для головы is_scam, видящей весь C).
    """
    turns = rec["turns"]
    onset = None
    print("  — раскрытие по префиксам (Enter=дальше · номер/'y'=онсет здесь · '0'=обмана нет · q=выход) —")
    for t in turns:
        i, role, speaker = t["i"], t.get("role", "?"), t.get("speaker", "")
        print(f"  [{i}] {speaker} ({role}): {t['text']}")
        if onset is not None or role != "atk":
            continue  # онсет уже зафиксирован, или tgt-реплика — онсетом быть не может
        ans = _prompt(f"    обман текстово наблюдаем в [{i}]? (Enter=нет · y=да · 0=обмана нет во всём диалоге · q): ").lower()
        if ans == "q":
            return "quit"
        if ans == "0":
            return None
        if ans in ("y", str(i)):
            onset = i
    return onset


def verify_one(rec, verified_by, prefix=False):
    """Провести человека по одному диалогу. Возвращает ('save', record) / ('dispute', None) / ('quit', None)."""
    turns = rec["turns"]
    print("\n" + "=" * 78)
    print(f"item_id={rec['item_id']}  id={rec.get('id')}  lang={rec.get('lang')}  "
          f"scam_type={rec.get('scam_type')}  T={len(turns)}")
    print("-" * 78)

    cur = draft_of(rec)
    if prefix:
        pre_onset = _reveal_prefix(rec)
        if pre_onset == "quit":
            return "quit", None
        cur = dict(cur)
        cur["onset"] = pre_onset
        print("-" * 78)
    else:
        print(format_dialogue(rec))
        print("-" * 78)
    _show_labels(draft_of(rec), turns, "ЧЕРНОВАЯ метка (llm_draft)")
    if prefix:
        _show_labels(cur, turns, "онсет по префиксу (ваш)")
    print("\n  Enter=подтвердить · s=is_scam+подтип · o=onset · n=спорный(skip) · q=выход")
    while True:
        cmd = _prompt("  > ").lower()
        if cmd == "":
            for w in consistency_warnings(cur, turns):
                print(f"  {w}")
            note = diff_note(draft_of(rec), cur)
            return "save", make_verified_record(rec, cur, verified_by, note)
        if cmd == "s":
            is_scam, subtypes = _edit_is_scam()
            cur = {"is_scam": is_scam, "subtypes": subtypes, "onset": cur["onset"]}
            if is_scam == 0:
                cur["onset"] = None
            _show_labels(cur, turns, "текущее")
        elif cmd == "o":
            cur = dict(cur)
            cur["onset"] = _edit_onset(turns)
            _show_labels(cur, turns, "текущее")
        elif cmd == "n":
            return "dispute", None
        elif cmd == "q":
            return "quit", None
        else:
            print("    команды: Enter / s / o / n / q")


def run(in_path, out_path, prefix=False):
    records = read_jsonl(in_path)
    if not records:
        print(f"Нет записей в {in_path}", file=sys.stderr)
        return 1
    verified = read_jsonl(out_path)
    done = done_item_ids(verified)
    todo = pending_records(records, done)

    print("Верификация меток (стадия 2) — гайдлайн v1.2")
    print(f"вход: {in_path}  ({len(records)} записей, уже готово {len(done)})")
    print(f"выход: {out_path}  (осталось проверить {len(todo)})")
    if not todo:
        print("Все записи уже верифицированы.")
        _print_summary(read_jsonl(out_path), session=None)
        return 0

    verified_by = _prompt("\nВаше имя (verified_by): ") or "anonymous"

    session = {"confirmed": 0, "corrected": 0, "disputed": 0,
               "changes": {"is_scam": 0, "onset": 0, "subtype": 0}}
    for rec in todo:
        action, out_rec = verify_one(rec, verified_by, prefix=prefix)
        if action == "quit":
            print("\nВыход. Прогресс сохранён.")
            break
        if action == "dispute":
            session["disputed"] += 1
            print("  → помечено спорным (вернёмся при следующем запуске)")
            continue
        # save — записать СРАЗУ (прогресс после каждого диалога)
        append_jsonl(out_path, out_rec)
        ct = change_types(out_rec["draft_labels"], out_rec)
        if ct:
            session["corrected"] += 1
            for t in ct:
                session["changes"][t] += 1
            print(f"  ✓ сохранено (правки: {out_rec['verifier_note']})")
        else:
            session["confirmed"] += 1
            print("  ✓ сохранено (без правок)")

    _print_summary(read_jsonl(out_path), session)
    return 0


def _print_summary(verified_records, session):
    print("\n" + "=" * 78)
    print("СВОДКА")
    if session is not None:
        print(f"  за сессию: подтверждено без правок {session['confirmed']}, "
              f"исправлено {session['corrected']}, спорных {session['disputed']}")
        ch = session["changes"]
        print(f"    из исправлений — is_scam: {ch['is_scam']}, onset: {ch['onset']}, subtype: {ch['subtype']}")
    ag = agreement_stats(verified_records)
    print(f"  всего верифицировано: {ag['n']}")
    if ag["n"]:
        ff = ag["field_frac"]
        print(f"  согласие human-vs-llm_draft (полное совпадение метки): "
              f"{ag['full_frac']:.3f} ({ag['full_agree']}/{ag['n']})")
        print(f"    по полям — is_scam: {ff['is_scam']:.3f}, "
              f"subtypes: {ff['subtypes']:.3f}, onset: {ff['onset']:.3f}")


def run_apply(in_path, out_path, judgments_path, verified_by, gold_src, note_prefix):
    """Не-интерактивный батч: применить суждения из JSON и записать verified-jsonl целиком."""
    records = read_jsonl(in_path)
    judgments = json.loads(pathlib.Path(judgments_path).read_text(encoding="utf-8"))
    out_recs = apply_judgments(records, judgments, verified_by, gold_src, note_prefix)
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Применено {len(out_recs)} суждений → {out_path} (gold_src={gold_src}, by={verified_by})")
    _print_summary(out_recs, session=None)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Верификация черновых меток (стадия 2)")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help=f"вход jsonl (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT, help=f"выход jsonl (default: {DEFAULT_OUT})")
    ap.add_argument("--prefix", action="store_true",
                    help="интерактив с пошаговым раскрытием реплик (онсет без заглядывания вперёд)")
    ap.add_argument("--apply", dest="apply_path", default=None,
                    help="не-интерактив: JSON с суждениями [{item_id,is_scam,subtypes,onset}] → батч-запись")
    ap.add_argument("--gold-src", default="human_verified", help="провенанс для --apply (default: human_verified)")
    ap.add_argument("--by", dest="verified_by", default=None, help="verified_by для --apply")
    ap.add_argument("--note", dest="note_prefix", default="", help="префикс verifier_note для --apply")
    a = ap.parse_args()
    try:
        if a.apply_path:
            return run_apply(a.in_path, a.out_path, a.apply_path,
                             a.verified_by or "anonymous", a.gold_src, a.note_prefix)
        return run(a.in_path, a.out_path, prefix=a.prefix)
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано. Прогресс сохранён до последнего проверенного диалога.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
