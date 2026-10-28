#!/usr/bin/env python3
"""Генерация ПОЛНОГО корпуса (Путь A, стадии 0-1): масштаб пилота 50 -> ~900 диалогов.

Пример:
  ANTHROPIC_API_KEY=... python scripts/generate_corpus.py --n 900 --langs ru,en,zh \\
      --out data/corpus/corpus_v1.jsonl

Переиспользует существующий конвейер (skeleton.py + generate.py): корпус-скелеты
(build_corpus_skeletons) -> build_prompt -> extract_json -> build_record -> validate_record.
НОВОЕ поверх пайплайна — то, чего требует масштаб:
  - ЧЕКПОИНТЫ: каждый принятый диалог дописывается в jsonl СРАЗУ (append); при обрыве
    сессии/сети ничего не теряется. --resume (по умолчанию) продолжает с недостающих
    скелетов по skeleton_id (id уже в файле — пропускаем).
  - троттлинг/ретраи на ошибки API с экспоненциальным backoff; неисправимый диалог ->
    rejected-файл, пайплайн не падает.
  - ОЦЕНКА стоимости перед запуском (диалоги/вызовы/токены) + подтверждение (--yes).
  - отчёт состава -> reports/corpus_v1_report.json.

Метки — ЧЕРНОВИК (gold_src=llm_draft): дальше человеческая верификация (стадия 2,
scripts/verify_labels.py) по замороженному гайдлайну v1.2.
"""
import argparse
import json
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.data.generate import (  # noqa: E402
    GenerationError, build_client, build_corpus_skeletons, build_prompt,
    build_record, extract_json, load_config, validate_record,
)

DEFAULT_OUT = "data/corpus/corpus_v1.jsonl"
DEFAULT_REPORT = "reports/corpus_v1_report.json"


# ─────────────────────────── чекпоинт-IO (тестируемо) ───────────────────────────

def read_jsonl(path):
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
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_ids(records):
    """skeleton_id (поле id) уже сгенерированных диалогов — для resume."""
    return {r["id"] for r in records if r.get("id")}


def pending_skeletons(skeletons, done):
    """Скелеты, которых ещё нет в корпусе (по sid), в исходном порядке."""
    return [s for s in skeletons if s.sid not in done]


# ─────────────────────────── генерация одного диалога ───────────────────────────

def _complete_with_backoff(client, model, system, user, *, max_retries, backoff_base, sleep):
    """client.complete с экспоненциальным backoff на ошибках API. Возвращает (raw, None)
    или (None, reason) если API не ответил после всех попыток."""
    for attempt in range(max_retries):
        try:
            return client.complete(model, system, user), None
        except GenerationError as e:
            if attempt == max_retries - 1:
                return None, f"API ошибка после {max_retries} попыток: {e}"
            sleep(backoff_base * (2 ** attempt))    # 1,2,4,8… × base
    return None, "API недоступен"


def generate_one(skel, client, model, *, max_attempts, turn_min, turn_max,
                 max_retries, backoff_base, sleep):
    """Скелет -> валидный rec или (None, reasons). Ретраи: API-backoff + повтор на брак."""
    system, user = build_prompt(skel)
    last_reasons = ["генерация не запускалась"]
    for _ in range(max_attempts):
        raw, api_reason = _complete_with_backoff(
            client, model, system, user,
            max_retries=max_retries, backoff_base=backoff_base, sleep=sleep)
        if raw is None:
            last_reasons = [api_reason]
            break                                   # API мёртв — не крутим max_attempts впустую
        try:
            parsed = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_reasons = [f"JSON не распарсен: {e}"]
            continue
        rec = build_record(skel, parsed, model)
        last_reasons = validate_record(rec, skel, turn_min, turn_max)
        if not last_reasons:
            return rec, []
    return None, last_reasons


# ─────────────────────────── чекпоинтенный прогон ───────────────────────────

def run_corpus(skeletons, cfg, client, corpus_path, rejected_path, *,
               resume=True, log=None, sleep=time.sleep,
               max_retries=5, backoff_base=1.0, limit=None, only_lang=None):
    """Прогон с чекпоинтами: пишет каждый принятый диалог сразу; resume по skeleton_id.

    ``only_lang`` — обрабатывать только скелеты этого языка (пробная партия по языку);
    план (skeletons) и ротация моделей остаются от ПОЛНОГО плана, поэтому sid и resume
    совместимы: диалоги ложатся в тот же corpus_path и засчитываются в цель.
    ``limit`` — обработать не более N ОСТАВШИХСЯ скелетов за этот запуск (после фильтра языка).

    Возвращает (accepted_new, rejected_new) — только сгенерированные в ЭТОТ запуск.
    Полный корпус читается отдельно (для отчёта) — он на диске в corpus_path.
    """
    gen = cfg["generation"]
    cor = cfg.get("corpus", {})
    turn_min, turn_max = cor.get("turns_range", [gen["turns"]["min"], gen["turns"]["max"]])
    models = gen["models"]
    max_attempts = int(gen.get("api", {}).get("max_attempts", 2))

    existing = read_jsonl(corpus_path) if resume else []
    done = done_ids(existing)
    next_id = len(existing) + 1
    todo = pending_skeletons(skeletons, done)
    if only_lang is not None:
        todo = [s for s in todo if s.lang == only_lang]    # проба по языку; sid из полного плана
    if limit is not None:
        todo = todo[:limit]                                # пробная партия; план остаётся полным
    idx_of = {s.sid: i for i, s in enumerate(skeletons)}   # ротация моделей стабильна при resume

    accepted_new, rejected_new = [], []
    for n, skel in enumerate(todo, 1):
        model = models[idx_of[skel.sid] % len(models)]     # мульти-генератор (§9.4)
        try:
            rec, reasons = generate_one(
                skel, client, model, max_attempts=max_attempts,
                turn_min=turn_min, turn_max=turn_max,
                max_retries=max_retries, backoff_base=backoff_base, sleep=sleep)
        except Exception as e:  # noqa: BLE001 — страховка многочасового прогона:
            # ни одна неожиданная ошибка на ОДНОМ диалоге не должна убивать весь цикл
            rec, reasons = None, [f"необработанная ошибка: {type(e).__name__}: {e}"]
        if rec is not None:
            rec["item_id"] = next_id
            next_id += 1
            append_jsonl(corpus_path, rec)             # ← чекпоинт ПОСЛЕ КАЖДОГО диалога
            accepted_new.append(rec)
            status = "ok"
        else:
            rej = {"id": skel.sid, "lang": skel.lang, "scenario": skel.scenario_id,
                   "reasons": reasons}
            append_jsonl(rejected_path, rej)
            rejected_new.append(rej)
            status = f"REJECT ({'; '.join(reasons)})"
        if log:
            log(f"[{len(done) + n}/{len(skeletons)}] {skel.sid}: {status}")
    return accepted_new, rejected_new


# ─────────────────────────── оценка стоимости ───────────────────────────

def _approx_tokens(text):
    """Грубая оценка токенов (≈4 символа/токен) — для прикидки счёта, не биллинг."""
    return max(1, len(text) // 4)


def estimate_cost(pending, cfg):
    """Оценка объёма для ОСТАВШИХСЯ скелетов: вызовы (min/max) и ориентировочные токены."""
    gen = cfg["generation"]
    api = gen.get("api", {})
    max_attempts = int(api.get("max_attempts", 2))
    max_tokens = int(api.get("max_tokens", 4096))
    in_tokens = 0
    for s in pending:
        system, user = build_prompt(s)
        in_tokens += _approx_tokens(system) + _approx_tokens(user)
    n = len(pending)
    return {
        "pending": n,
        "calls_min": n,                       # успех с первой попытки
        "calls_max": n * max_attempts,        # худший случай (повторы на брак)
        "in_tokens_per_pass": in_tokens,      # промпт-токены за один проход всех скелетов
        "out_tokens_max": n * max_attempts * max_tokens,
        "in_tokens_max": in_tokens * max_attempts,
        "max_tokens_per_call": max_tokens,
        "max_attempts": max_attempts,
    }


def print_estimate(est, models):
    print("\n=== ОЦЕНКА (перед запуском) ===")
    print(f"осталось сгенерировать: {est['pending']} диалогов")
    print(f"вызовов API: ~{est['calls_min']} (успех с 1-й) … {est['calls_max']} (макс. с повторами)")
    print(f"токены (ориентир): вход ~{est['in_tokens_per_pass']:,} за проход "
          f"(до ~{est['in_tokens_max']:,} с повторами); "
          f"выход до ~{est['out_tokens_max']:,} (по {est['max_tokens_per_call']}/вызов)")
    print(f"генераторы: {models}"
          + ("" if len(models) >= 2 else "  ⚠ один генератор — полный корпус требует >=2 (§9.4)"))


# ─────────────────────────── отчёт состава ───────────────────────────

def corpus_report(accepted, rejected, models, cfg=None):
    """Полный отчёт по корпусу на диске (accepted = ВСЕ записи corpus_v1.jsonl)."""
    n = len(accepted)
    scam = sum(r.get("is_scam") == 1 for r in accepted)
    benign = n - scam
    hard_neg = sum(r.get("hard_negative") == 1 for r in accepted)
    onset_counts = Counter(r.get("onset") for r in accepted if r.get("is_scam") == 1)
    cor = (cfg or {}).get("corpus", {})
    return {
        "target_n": cor.get("target_n"),
        "n_accepted": n,
        "n_rejected": len(rejected),
        "by_lang": dict(Counter(r.get("lang") for r in accepted)),
        "by_scenario": dict(Counter(r.get("scam_type") for r in accepted)),
        "by_lang_scenario": {
            f"{lang}/{sc}": c for (lang, sc), c in
            Counter((r.get("lang"), r.get("scam_type")) for r in accepted).items()
        },
        "scam": scam,
        "benign": benign,
        "scam_share": round(scam / n, 3) if n else None,
        "hard_negatives": hard_neg,
        "hard_neg_share_of_benign": round(hard_neg / benign, 3) if benign else None,
        "mean_turns": round(sum(len(r.get("turns") or []) for r in accepted) / n, 2) if n else None,
        "onset_distribution": {str(k): v for k, v in sorted(
            onset_counts.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "benign_onset_none": sum(r.get("is_scam") == 0 for r in accepted),
        "reject_reasons": dict(Counter(x for r in rejected for x in r.get("reasons", []))),
        "generators": dict(Counter(r.get("generator") for r in accepted)),
        "multi_generator_ok": len(set(models)) >= 2,
    }


def print_report(rep):
    print("\n=== отчёт состава ===")
    print(f"принято: {rep['n_accepted']}"
          + (f" / цель {rep['target_n']}" if rep["target_n"] else "")
          + f" | отклонено: {rep['n_rejected']}")
    print(f"scam: {rep['scam']} | benign: {rep['benign']} (scam-доля {rep['scam_share']}) "
          f"| hard-neg: {rep['hard_negatives']} (доля от benign {rep['hard_neg_share_of_benign']})")
    print(f"средняя длина: {rep['mean_turns']} реплик")
    print("по языкам:  " + ", ".join(f"{k}={v}" for k, v in sorted(rep["by_lang"].items())))
    print("по сценариям:")
    for k, v in sorted(rep["by_scenario"].items()):
        print(f"  {k:24s} {v}")
    print("распределение онсетов (scam): "
          + ", ".join(f"t*={k}:{v}" for k, v in rep["onset_distribution"].items()))
    if rep["reject_reasons"]:
        print("причины брака:")
        for k, v in sorted(rep["reject_reasons"].items()):
            print(f"  {v}× {k}")
    print(f"генераторы: {rep['generators']}"
          + ("" if rep["multi_generator_ok"] else "  ⚠ <2 генераторов (§9.4)"))


# ─────────────────────────── CLI ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Генерация полного корпуса (стадии 0-1, чекпоинты+resume)")
    ap.add_argument("--n", type=int, default=None, help="целевой размер (default: corpus.target_n)")
    ap.add_argument("--langs", default=None, help="языки через запятую (default: corpus.per_lang)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"корпус jsonl (default: {DEFAULT_OUT})")
    ap.add_argument("--report", default=DEFAULT_REPORT, help=f"отчёт (default: {DEFAULT_REPORT})")
    ap.add_argument("--seed", type=int, default=None, help="default: seeds[0] из конфига")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="не продолжать: генерировать с нуля (перезапись not-append всё равно append)")
    ap.add_argument("--yes", action="store_true", help="не спрашивать подтверждение оценки")
    ap.add_argument("--limit", type=int, default=None,
                    help="сгенерировать не более N за этот запуск (пробная партия; resume добьёт остальное)")
    ap.add_argument("--only-lang", default=None,
                    help="обработать только этот язык из плана (проба по языку; sid и resume совместимы)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cor = cfg.get("corpus", {})
    gen = cfg["generation"]
    seed = a.seed if a.seed is not None else cfg["seeds"][0]

    # per_lang: из --langs (делим target поровну) или из конфига
    if a.langs:
        langs = a.langs.split(",")
        total = a.n or cor.get("target_n", 900)
        base, extra = divmod(total, len(langs))
        per_lang = {lang: base + (1 if i < extra else 0) for i, lang in enumerate(langs)}
    else:
        per_lang = dict(cor.get("per_lang", {"ru": 300, "en": 300, "zh": 300}))
        if a.n and a.n != sum(per_lang.values()):
            # масштабировать пропорционально под --n
            factor = a.n / sum(per_lang.values())
            per_lang = {k: round(v * factor) for k, v in per_lang.items()}

    tr = cor.get("turns_range", [gen["turns"]["min"], gen["turns"]["max"]])
    skeletons = build_corpus_skeletons(
        per_lang, scam_ratio=cor.get("scam_ratio", 0.5),
        hard_neg_ratio=cor.get("hard_neg_ratio", 0.35),
        turn_min=tr[0], turn_max=tr[1],
        min_per_scenario=cor.get("min_per_scenario", 10), seed=seed)
    print(f"скелетов: {len(skeletons)} | по языкам: {per_lang} | генераторы: {gen['models']}")

    corpus_path = pathlib.Path(a.out)
    rejected_path = corpus_path.with_name(corpus_path.stem + "_rejected.jsonl")
    existing = read_jsonl(corpus_path) if a.resume else []
    todo = pending_skeletons(skeletons, done_ids(existing))
    if existing:
        print(f"resume: в {corpus_path} уже {len(existing)} диалогов, осталось {len(todo)}")
    if a.only_lang is not None:
        todo = [s for s in todo if s.lang == a.only_lang]
        print(f"only-lang: за этот запуск только {a.only_lang} ({len(todo)} доступно в плане)")
    if a.limit is not None:
        todo = todo[:a.limit]
        print(f"limit: за этот запуск сгенерирую не более {a.limit} (план полный, resume добьёт остальное)")

    est = estimate_cost(todo, cfg)
    print_estimate(est, gen["models"])
    if not todo:
        print("Всё уже сгенерировано — перехожу к отчёту.")
    elif not a.yes:
        if input("\nЗапустить генерацию? [y/N]: ").strip().lower() != "y":
            print("Отменено.")
            return 0

    if todo:
        try:
            client = build_client(gen["models"], gen.get("api"))
        except GenerationError as e:
            print(f"ОШИБКА: {e}", file=sys.stderr)
            return 1
        run_corpus(skeletons, cfg, client, corpus_path, rejected_path,
                   resume=a.resume, log=print, limit=a.limit, only_lang=a.only_lang)

    accepted = read_jsonl(corpus_path)                 # ВЕСЬ корпус (с учётом прошлых запусков)
    rejected = read_jsonl(rejected_path)
    rep = corpus_report(accepted, rejected, gen["models"], cfg)
    print_report(rep)
    report_path = pathlib.Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nфайлы: {corpus_path}  |  {rejected_path}  |  отчёт: {report_path}")
    print("метки — ЧЕРНОВИК (gold_src=llm_draft); далее верификация scripts/verify_labels.py (v1.2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
