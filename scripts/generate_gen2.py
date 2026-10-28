#!/usr/bin/env python3
"""Второй генератор: 200 диалогов ДРУГИМ вендором на тех же промптах, что corpus_v1.

Зачем. Корпус v1 сделан одним вендором (deepseek-chat + deepseek-reasoner), и цена
этого измерена: TF-IDF воспроизводит silver на 98.7%, слой 2 на 98.9%, сам DeepSeek
на 99.0%. Значит, любая модель без файнтюна ловит стилевой отпечаток генератора, а не
обман. gen2 — контрольная партия, на которой это проверяется напрямую: модели, обученные
на train v1, меряются на чужом стиле (scripts/gen2_eval.py).

Что переиспользуется БЕЗ ИЗМЕНЕНИЙ (правило: исходные промпты не переписывать):
  - таксономия 19 сценариев     — src/scam_onset/data/generate/scenarios.py
  - шаблоны промптов            — .../prompts.py::build_prompt
  - схема записи и валидация    — .../generate.py::build_record / validate_record
  - роутинг вендоров и клиенты  — .../generate.py::build_client
  - чекпоинты/resume/backoff    — scripts/generate_corpus.py
Пропорции берутся из configs/default.yaml: gen2 (те же scam_ratio и hard_neg_ratio,
что у corpus), поэтому баланс scam/benign и доля hard-negative совпадают с v1.

ЕДИНСТВЕННОЕ отличие промпта — ДОБАВКА про формат ссылки (URL_ADDENDUM ниже). Шаблоны
не редактируются: добавка приписывается к системному промпту и целиком сохраняется в
отчёт прогона, чтобы отличие было видно и воспроизводимо.

  ANTHROPIC_API_KEY=... python scripts/generate_gen2.py            # генерация
  python scripts/generate_gen2.py --dry-run                        # план без сети и денег
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scam_onset.data.generate import (  # noqa: E402
    GenerationError, build_client, build_corpus_skeletons, build_prompt,
    build_record, extract_json, load_config, validate_record, vendor_of,
)
# ENV_KEY не входит в __all__ пакета — берём из модуля напрямую, пакет не трогаем
from scam_onset.data.generate.generate import ENV_KEY  # noqa: E402
from generate_corpus import (  # noqa: E402
    _complete_with_backoff, append_jsonl, done_ids, estimate_cost, pending_skeletons,
    print_estimate, read_jsonl,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = "data/corpus/gen2_v1.jsonl"
DEFAULT_REPORT = "runs/gen2/generation_report.json"

# ── единственная дельта к промптам v1 ───────────────────────────────────────────
# В v1 системный промпт требовал плейсхолдер (URL), и генератор писал ссылку СЛОВОМ:
# на 2000 диалогов ноль вхождений http/www (CLAUDE.md «Слой 0»). Из-за этого у подтипа
# phishing_link не было поверхностного якоря, и он оказался худшим классом (F1 0.4530
# при recall 0.3058). Здесь ссылка пишется обезвреженной, но структурно похожей.
# Безопасность сохраняется: hxxp не является схемой, [.] не является разделителем домена,
# .example зарезервирован RFC 2606 и не резолвится — ссылка некликабельна и никуда не ведёт.
URL_ADDENDUM = """

ДОПОЛНЕНИЕ К ПРАВИЛУ ПЛЕЙСХОЛДЕРОВ (заменяет ТОЛЬКО пункт про (URL), остальные
плейсхолдеры — (CARD), (CODE), (AMOUNT), (NAME), (NUM), (APP), (BRAND) — без изменений):
вместо плейсхолдера (URL) пиши ОБЕЗВРЕЖЕННУЮ ссылку вида hxxp://secure-verify[.]example/a8f2.
Формат жёсткий: схема ровно hxxp:// (не http и не https), домен второго уровня из
латиницы и дефисов по смыслу сообщения, затем [.] в квадратных скобках, затем example,
затем / и 4-8 символов из [a-z0-9]. Домен верхнего уровня всегда example.
Такая ссылка некликабельна и никуда не ведёт — это осознанное требование безопасности.
Настоящие ссылки (http://, https://, www.), реальные домены и сокращатели запрещены.
Ссылку вставляй только там, где по сценарию она уместна; в остальных диалогах её нет."""

# Валидатор формата: hxxp://<домен>[.]example/<4-8 [a-z0-9]>
DEFANGED_RE = re.compile(r"hxxp://[a-z0-9][a-z0-9-]*\[\.\]example/[a-z0-9]{4,8}",
                         re.IGNORECASE)
# Живая ссылка — брак. Список сокращателей тот же, что в слое 0 (runs/layer0 url_regex).
LIVE_URL_RE = re.compile(
    r"(https?://|www\.|bit\.ly|t\.co|tinyurl\.com|goo\.gl|is\.gd|cutt\.ly|ow\.ly|"
    r"rb\.gy|shorturl\.at|clck\.ru|vk\.cc|clc\.to|u\.to|dwz\.cn|t\.cn|url\.cn|"
    r"suo\.im|sourl\.cn|mrw\.so)", re.IGNORECASE)
# Остаточный плейсхолдер v1 — тоже брак: значит, добавку модель проигнорировала.
PLACEHOLDER_URL_RE = re.compile(r"\(URL\)")


def rel(p):
    """Путь относительно корня репозитория, если он внутри; иначе как есть."""
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def dialog_text(rec):
    return "\n".join(t.get("text", "") for t in (rec.get("turns") or []))


def validate_gen2(rec, skel):
    """Проверки СВЕРХ validate_record: только про ссылки, остальное — как в v1."""
    reasons = []
    text = dialog_text(rec)
    if LIVE_URL_RE.search(text):
        reasons.append(f"живая ссылка в тексте: {LIVE_URL_RE.search(text).group(0)!r}")
    if PLACEHOLDER_URL_RE.search(text):
        reasons.append("остался плейсхолдер (URL) — добавка про формат ссылки не сработала")
    bad = [m for m in re.findall(r"hxxp://\S+", text, re.IGNORECASE)
           if not DEFANGED_RE.fullmatch(m.rstrip(".,;:!?)»\"'"))]
    if bad:
        reasons.append(f"ссылка не в формате hxxp://…[.]example/xxxx: {bad[0]!r}")
    # phishing_link заявлен подтипом -> ссылка обязана быть в тексте (иначе якоря снова нет)
    if "phishing_link" in (rec.get("subtypes") or []) and not DEFANGED_RE.search(text):
        reasons.append("подтип phishing_link без обезвреженной ссылки в тексте")
    return reasons


def generate_one(skel, client, model, *, max_attempts, turn_min, turn_max,
                 max_retries, backoff_base, sleep, addendum):
    """Скелет -> валидная запись. Промпт = build_prompt(skel) + добавка про ссылку."""
    system, user = build_prompt(skel)
    system = system + addendum
    last_reasons = ["генерация не запускалась"]
    for _ in range(max_attempts):
        raw, api_reason = _complete_with_backoff(
            client, model, system, user,
            max_retries=max_retries, backoff_base=backoff_base, sleep=sleep)
        if raw is None:
            return None, [api_reason]          # API мёртв — не крутим попытки впустую
        try:
            parsed = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_reasons = [f"JSON не распарсен: {e}"]
            continue
        rec = build_record(skel, parsed, model)
        last_reasons = validate_record(rec, skel, turn_min, turn_max) + validate_gen2(rec, skel)
        if not last_reasons:
            rec["generator_batch"] = "gen2"
            return rec, []
    return None, last_reasons


def run_gen2(skeletons, models, client, corpus_path, rejected_path, *, turn_min, turn_max,
             max_attempts, addendum, resume=True, log=print, sleep=time.sleep,
             max_retries=5, backoff_base=1.0, limit=None):
    """Чекпоинтенный прогон: каждый принятый диалог дописывается сразу, resume по sid."""
    existing = read_jsonl(corpus_path) if resume else []
    done = done_ids(existing)
    next_id = len(existing) + 1
    todo = pending_skeletons(skeletons, done)
    if limit is not None:
        todo = todo[:limit]
    idx_of = {s.sid: i for i, s in enumerate(skeletons)}

    accepted, rejected = [], []
    for n, skel in enumerate(todo, 1):
        model = models[idx_of[skel.sid] % len(models)]
        try:
            rec, reasons = generate_one(
                skel, client, model, max_attempts=max_attempts,
                turn_min=turn_min, turn_max=turn_max, max_retries=max_retries,
                backoff_base=backoff_base, sleep=sleep, addendum=addendum)
        except Exception as e:  # noqa: BLE001 — один диалог не должен убивать прогон
            rec, reasons = None, [f"необработанная ошибка: {type(e).__name__}: {e}"]
        if rec is not None:
            rec["item_id"] = next_id
            next_id += 1
            append_jsonl(corpus_path, rec)
            accepted.append(rec)
            status = "ok"
        else:
            rej = {"id": skel.sid, "lang": skel.lang, "scenario": skel.scenario_id,
                   "reasons": reasons}
            append_jsonl(rejected_path, rej)
            rejected.append(rej)
            status = f"REJECT ({'; '.join(reasons)})"
        log(f"[{len(done) + n}/{len(skeletons)}] {skel.sid}: {status}")
    return accepted, rejected


def composition(records, skeletons):
    """Состав партии + сверка пропорций с планом скелетов."""
    n = len(records)
    scam = sum(r.get("is_scam") == 1 for r in records)
    with_url = sum(1 for r in records if DEFANGED_RE.search(dialog_text(r)))
    phish = [r for r in records if "phishing_link" in (r.get("subtypes") or [])]
    return {
        "n": n,
        "by_lang": dict(Counter(r.get("lang") for r in records)),
        "by_scenario": dict(Counter(r.get("scam_type") for r in records)),
        "scenarios_covered": len({r.get("scam_type") for r in records}),
        "scenarios_planned": len({s.scenario_id for s in skeletons}),
        "scam": scam,
        "benign": n - scam,
        "scam_share": round(scam / n, 3) if n else None,
        "hard_negatives": sum(r.get("hard_negative") == 1 for r in records),
        "mean_turns": round(sum(len(r.get("turns") or []) for r in records) / n, 2) if n else None,
        "onset_distribution": {
            str(k): v for k, v in sorted(
                Counter(r.get("onset") for r in records if r.get("is_scam") == 1).items(),
                key=lambda kv: (kv[0] is None, kv[0]))},
        "generators": dict(Counter(r.get("generator") for r in records)),
        "dialogs_with_defanged_url": with_url,
        "defanged_url_share": round(with_url / n, 3) if n else None,
        "phishing_link_dialogs": len(phish),
        "phishing_link_with_url": sum(1 for r in phish if DEFANGED_RE.search(dialog_text(r))),
        "live_urls": sum(1 for r in records if LIVE_URL_RE.search(dialog_text(r))),
    }


def main():
    ap = argparse.ArgumentParser(description="Второй генератор: 200 диалогов другим вендором")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--models", default=None, help="через запятую; default: gen2.models")
    ap.add_argument("--n", type=int, default=None, help="default: gen2.target_n")
    ap.add_argument("--seed", type=int, default=None, help="default: gen2.seed")
    ap.add_argument("--limit", type=int, default=None, help="не более N за этот запуск")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="построить план и промпты, показать оценку и выйти (без сети)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    g2 = cfg.get("gen2")
    if not g2:
        raise SystemExit(f"в {a.config} нет секции gen2")
    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if a.models else list(g2["models"]))
    seed = a.seed if a.seed is not None else int(g2["seed"])
    per_lang = dict(g2["per_lang"])
    if a.n and a.n != sum(per_lang.values()):
        factor = a.n / sum(per_lang.values())
        per_lang = {k: round(v * factor) for k, v in per_lang.items()}
    tr = g2["turns_range"]

    vendors = {vendor_of(m) for m in models}
    if "deepseek" in vendors:
        raise SystemExit(
            "gen2 обязан быть НЕ-DeepSeek: корпус v1 и silver-метки породил DeepSeek, "
            "и партия его же моделью проверит согласие со стилем самого себя, а не перенос. "
            f"Сейчас в списке: {models}")

    skeletons = build_corpus_skeletons(
        per_lang, scam_ratio=g2["scam_ratio"], hard_neg_ratio=g2["hard_neg_ratio"],
        turn_min=tr[0], turn_max=tr[1], seed=seed, prefix=g2["sid_prefix"])

    print(f"=== gen2: {len(skeletons)} скелетов | {per_lang} | генераторы {models} ===")
    print(f"сценариев в плане: {len({s.scenario_id for s in skeletons})} из 19 "
          f"| scam {sum(s.is_scam for s in skeletons)} "
          f"| hard-neg {sum(s.hard_negative for s in skeletons)}")
    print(f"промпты: те же, что у corpus_v1 (build_prompt), плюс добавка про формат ссылки "
          f"({len(URL_ADDENDUM)} символов)")

    corpus_path = pathlib.Path(a.out)
    rejected_path = corpus_path.with_name(corpus_path.stem + "_rejected.jsonl")
    existing = read_jsonl(corpus_path) if a.resume else []
    todo = pending_skeletons(skeletons, done_ids(existing))
    if existing:
        print(f"resume: уже есть {len(existing)}, осталось {len(todo)}")
    if a.limit is not None:
        todo = todo[:a.limit]
    est = estimate_cost(todo, cfg)
    print_estimate(est, models)
    print("  (предупреждение «>=2 генератора» относится к ПОЛНОМУ корпусу §9.4; "
          "gen2 — контрольная партия одним ВТОРЫМ вендором, это и есть замысел)")

    if a.dry_run:
        sys_p, usr_p = build_prompt(skeletons[0])
        print("\n=== ПРОБНЫЙ ПРОГОН (--dry-run): сеть не трогалась ===")
        print(f"пример промпта для {skeletons[0].sid} "
              f"(сценарий {skeletons[0].scenario_id}, {skeletons[0].n_turns} реплик, "
              f"t*_cand={skeletons[0].onset_cand}):")
        print("--- добавка к системному промпту ---")
        print(URL_ADDENDUM.strip())
        print(f"--- системный промпт: {len(sys_p)} симв. + добавка {len(URL_ADDENDUM)} ---")
        print(f"--- пользовательский промпт: {len(usr_p)} симв. (шаблон v1, не изменён) ---")
        print("\nсамопроверка валидатора ссылок:")
        for txt, want in (("зайдите hxxp://secure-verify[.]example/a8f2 сейчас", True),
                          ("зайдите https://secure-verify.example/a8f2", False),
                          ("перейдите по ссылке (URL)", False),
                          ("hxxp://pay[.]example/QQ", False)):
            rec = {"turns": [{"text": txt}], "subtypes": ["phishing_link"]}
            ok = not validate_gen2(rec, None)
            mark = "OK  " if ok == want else "FAIL"
            print(f"  {mark} {'принять' if want else 'отклонить'}: {txt[:52]}")
        report = {"task": "gen2_plan", "dry_run": True, "models": models, "seed": seed,
                  "per_lang": per_lang, "n_skeletons": len(skeletons),
                  "scenarios_planned": sorted({s.scenario_id for s in skeletons}),
                  "url_addendum": URL_ADDENDUM, "estimate": est,
                  "env_key_required": sorted(ENV_KEY[v] for v in vendors)}
        out = pathlib.Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nплан сохранён: {rel(out)}")
        print(f"для запуска нужен ключ: {', '.join(sorted(ENV_KEY[v] for v in vendors))}")
        return 0

    if todo and not a.yes:
        if input("\nЗапустить генерацию? [y/N]: ").strip().lower() != "y":
            print("Отменено.")
            return 0

    if todo:
        try:
            client = build_client(models, cfg["generation"].get("api"))
        except GenerationError as e:
            print(f"ОШИБКА: {e}", file=sys.stderr)
            return 1
        run_gen2(skeletons, models, client, corpus_path, rejected_path,
                 turn_min=tr[0], turn_max=tr[1],
                 max_attempts=int(cfg["generation"].get("api", {}).get("max_attempts", 2)),
                 addendum=URL_ADDENDUM, resume=a.resume, limit=a.limit)

    records = read_jsonl(corpus_path)
    rejected = read_jsonl(rejected_path)
    comp = composition(records, skeletons)
    report = {"task": "gen2", "models": models, "seed": seed, "per_lang": per_lang,
              "url_addendum": URL_ADDENDUM, "composition": comp,
              "n_rejected": len(rejected),
              "reject_reasons": dict(Counter(x for r in rejected
                                             for x in r.get("reasons", [])))}
    out = pathlib.Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== состав gen2 ===")
    print(f"принято {comp['n']} | отклонено {len(rejected)} | "
          f"scam {comp['scam']} / benign {comp['benign']} (доля {comp['scam_share']})")
    print(f"языки {comp['by_lang']} | сценариев {comp['scenarios_covered']}/"
          f"{comp['scenarios_planned']} | hard-neg {comp['hard_negatives']}")
    print(f"обезвреженных ссылок: {comp['dialogs_with_defanged_url']} диалогов "
          f"({comp['defanged_url_share']}); живых ссылок: {comp['live_urls']} (должно быть 0)")
    print(f"phishing_link: {comp['phishing_link_dialogs']} диалогов, из них со ссылкой "
          f"{comp['phishing_link_with_url']}")
    print(f"\nфайлы: {corpus_path} | {rejected_path} | отчёт: {rel(out)}")
    print("метки — ЧЕРНОВИК генератора (gold_src=llm_draft), как в v1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
