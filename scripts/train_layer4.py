#!/usr/bin/env python3
"""Слой 4: LLM few-shot бейзлайн — сильная модель БЕЗ обучения.

Закрывает вопрос «зачем файнтюн, если LLM умеет это из коробки». Модель за один
вызов возвращает строго JSON {"is_scam":0|1,"subtypes":[...],"onset":int|null}.

Три режима промпта, все на одном и том же наборе:
  zero       — только инструкция
  few        — инструкция + 6 фиксированных примеров из TRAIN (2 на язык, 3/3)
  few_rules  — few + краткое изложение правил гайдлайна v1.2

Режим выбирается по gold-dev 100; финальный замер — gold-test v2 (303). Тест в
подборе промпта не участвует. temperature=0, но прогонов три: LLM недетерминированы
и при нуле (кэш-ключ включает номер прогона, иначе три прогона были бы копией).

⚠ ЦИКЛИЧНОСТЬ: корпус и silver-метки породил DeepSeek. Он НЕ может быть основным
бейзлайном — будет соглашаться сам с собой. Основная модель обязана быть другого
вендора; DeepSeek запускается отдельно (--control) как проверка цикличности.

  python scripts/train_layer4.py --provider openai --model gpt-4o-mini
  python scripts/train_layer4.py --provider deepseek --model deepseek-chat --control
  python scripts/train_layer4.py --mock            # прогон на заглушке, без сети и денег
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_fewshot_examples import dialog_text  # noqa: E402
from train_binary import read_corpus, read_gold_csv  # noqa: E402

SUBTYPES = ["phishing_link", "money_request", "other"]
MODES = ["zero", "few", "few_rules"]
N_RUNS = 3

# Роутинг по вендорам. Ключи берутся ТОЛЬКО из окружения.
PROVIDERS = {
    "openai":     {"base": "https://api.openai.com/v1", "env": "OPENAI_API_KEY",
                   "style": "openai", "json_mode": True},
    "deepseek":   {"base": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY",
                   "style": "openai", "json_mode": True},
    "anthropic":  {"base": "https://api.anthropic.com/v1", "env": "ANTHROPIC_API_KEY",
                   "style": "anthropic", "json_mode": False},
    "dashscope":  {"base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                   "env": "DASHSCOPE_API_KEY", "style": "openai", "json_mode": True},
    "together":   {"base": "https://api.together.xyz/v1", "env": "TOGETHER_API_KEY",
                   "style": "openai", "json_mode": True},
    "groq":       {"base": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY",
                   "style": "openai", "json_mode": True},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY",
                   "style": "openai", "json_mode": True},
    "fireworks":  {"base": "https://api.fireworks.ai/inference/v1",
                   "env": "FIREWORKS_API_KEY", "style": "openai", "json_mode": True},
}

INSTRUCTION = """Ты размечаешь диалоги на предмет мошенничества (скама). Диалог дан построчно
в формате [номер|роль] текст, где роль atk — потенциально атакующая сторона, tgt — её собеседник.

Верни РОВНО один JSON-объект и ничего больше — без пояснений, без markdown-ограждений:
{"is_scam": 0 или 1, "subtypes": [...], "onset": целое число или null}

Поля:
- is_scam: 1, если хотя бы одна сторона обманывает, чтобы подтолкнуть другую к действию во вред ей
  (отдать деньги, назвать код/карту/пароль, перейти по вредной ссылке, установить ПО); иначе 0.
- subtypes: подмножество ["phishing_link", "money_request", "other"]; несколько значений допустимо.
  Пустой список, если is_scam = 0.
- onset: номер ПЕРВОЙ реплики атакующей стороны, где обман наблюдается В ТЕКСТЕ. null, если is_scam = 0."""

RULES = """Правила разметки (краткое изложение замороженного гайдлайна v1.2):

ЧТО СЧИТАЕТСЯ СКАМОМ. Обман не обязан быть грубым, техническим или мгновенным.
Медленная манипуляция через выдуманные отношения или сценарий — тоже скам.
Скам: романтический развод (строит доверие на непроверяемой легенде, затем просит денег);
инвестиционный развод и «ошиблись номером» с нереальной доходностью; фейк-вакансия со
взносом за «активацию»; выдача себя за поддержку/банк/госорган; выдача себя за родственника
или начальника («мам, новый номер, срочно переведи»); «вы выиграли приз» с комиссией или
ссылкой; tech-support («ваш ПК заражён, установите ПО»).

ЧТО НЕ СКАМ (hard negatives — главный источник ложных тревог). Правдивое уведомление банка,
реальная доставка или чек, честный маркетинг («распродажа −40%»), обычная просьба денег без
обмана («займёшь до пятницы?»), честное предупреждение о мошенничестве. Признак benign:
НЕТ ложной легенды И не просят сделать опасное в пользу отправителя — даже если в диалоге
есть деньги, ссылка, срочность или тревожный тон.

ПОДТИПЫ. phishing_link — ведут по ссылке за данными или оплатой. money_request — просят
перевод, оплату, комиссию. other — обман через роль или сценарий: банк, госорган, родственник,
начальник, романтика, инвестиции, фейк-вакансия, tech-support. Если деньги просят ЧЕРЕЗ такую
роль — ставь и money_request, и other.

ОНСЕТ. Номер первой реплики атакующей стороны, где обман присутствует в тексте.
Правило 1: ложная личность как факт («это служба безопасности банка») — уже онсет, ДО просьбы.
Правило 2: при медленном заходе онсет — первая непроверяемая легенда («я на платформе в море»),
а не эмоция в начале и не просьба денег в конце.
Приветствие и эмоции сами по себе онсетом НЕ являются. У benign онсет пустой (null).

КОНСИСТЕНТНОСТЬ. is_scam=1 ⇒ хотя бы один подтип. is_scam=0 ⇒ подтипы пусты и onset null."""


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# ─────────────────────────────── промпты ───────────────────────────────
def build_messages(mode, dialog, examples):
    """(system, messages) под выбранный режим."""
    system = INSTRUCTION if mode == "zero" else INSTRUCTION
    if mode == "few_rules":
        system = INSTRUCTION + "\n\n" + RULES
    msgs = []
    if mode in ("few", "few_rules"):
        for e in examples:
            msgs.append({"role": "user", "content": e["dialog"]})
            msgs.append({"role": "assistant",
                         "content": json.dumps(e["answer"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": dialog})
    return system, msgs


# ──────────────────────────────── клиент ────────────────────────────────
class LLMClient:
    """Минимальный HTTP-клиент: OpenAI-совместимые API и Anthropic. Только stdlib."""

    def __init__(self, provider, model, temperature=0.0, max_tokens=200,
                 timeout=120, max_attempts=6, key_file=None):
        if provider not in PROVIDERS:
            raise SystemExit(f"неизвестный вендор {provider}; есть: {list(PROVIDERS)}")
        self.cfg = PROVIDERS[provider]
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout, self.max_attempts = timeout, max_attempts
        # ключ: из окружения либо из файла вне репозитория (--key-file).
        # Файл нужен потому, что оболочка агента стартует из снимка окружения:
        # export, сделанный в другом терминале, до неё не доходит.
        self.key = os.environ.get(self.cfg["env"])
        if not self.key and key_file:
            p = pathlib.Path(key_file).expanduser()
            if not p.exists():
                raise SystemExit(f"файл ключа не найден: {p}")
            self.key = p.read_text(encoding="utf-8").strip()
        if not self.key:
            raise SystemExit(
                f"нет ключа: ни переменной {self.cfg['env']}, ни --key-file. "
                f"Положи ключ в файл вне репозитория, например:\n"
                f"  printf %s 'КЛЮЧ' > ~/.{provider}_key && chmod 600 ~/.{provider}_key\n"
                f"и запусти с --key-file ~/.{provider}_key")

    def _request(self, system, messages):
        if self.cfg["style"] == "anthropic":
            url = f"{self.cfg['base']}/messages"
            body = {"model": self.model, "system": system, "messages": messages,
                    "max_tokens": self.max_tokens, "temperature": self.temperature}
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
        else:
            url = f"{self.cfg['base']}/chat/completions"
            body = {"model": self.model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "max_tokens": self.max_tokens, "temperature": self.temperature}
            if self.cfg["json_mode"]:
                body["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {self.key}",
                       "content-type": "application/json"}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, system, messages):
        """→ (текст, usage). Ретраи с экспоненциальной паузой на 429/5xx."""
        delay = 2.0
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                data = self._request(system, messages)
                if self.cfg["style"] == "anthropic":
                    text = "".join(b.get("text", "") for b in data.get("content", []))
                    u = data.get("usage", {})
                    usage = {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
                else:
                    text = data["choices"][0]["message"]["content"]
                    u = data.get("usage", {})
                    usage = {"in": u.get("prompt_tokens", 0),
                             "out": u.get("completion_tokens", 0)}
                return text, usage
            except urllib.error.HTTPError as e:
                last = e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if e.code in (408, 409, 429) or e.code >= 500:
                    pause = float(retry_after) if retry_after else delay
                    time.sleep(pause + random.uniform(0, 0.5))
                    delay = min(delay * 2, 60)
                    continue
                raise SystemExit(f"HTTP {e.code} от {self.provider}: "
                                 f"{e.read().decode('utf-8', 'replace')[:400]}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
                last = e
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60)
        raise RuntimeError(f"не удалось получить ответ за {self.max_attempts} попыток: {last!r}")


class MockClient:
    """Заглушка для смоук-теста: без сети и денег, ответ детерминирован по тексту."""

    provider, model = "mock", "mock"

    def complete(self, system, messages):
        dialog = messages[-1]["content"]
        h = int(hashlib.sha256(dialog.encode()).hexdigest(), 16)
        if h % 17 == 0:                       # ~6% нарочно битых ответов
            return "конечно! вот ответ: {is_scam: да}", {"in": 100, "out": 10}
        is_scam = int(h % 3 != 0)
        n_turns = max(1, dialog.count("\n") + 1)
        ans = {"is_scam": is_scam,
               "subtypes": ([SUBTYPES[h % 3]] if is_scam else []),
               "onset": (h % n_turns) + 1 if is_scam else None}
        return json.dumps(ans, ensure_ascii=False), {"in": 100, "out": 10}


# ──────────────────────────── парсинг ответа ────────────────────────────
def parse_answer(text, n_turns):
    """Строка → (разобранный ответ | None, причина отказа).

    Терпит markdown-ограждение и преамбулу, но требует валидную схему:
    is_scam ∈ {0,1}; subtypes ⊆ трёх известных; onset ∈ [1..T] либо null.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "пустой ответ"
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
            t = t.split("\n", 1)[1] if t[:4].lower().startswith("json") else t
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi <= lo:
        return None, "не найден JSON-объект"
    try:
        obj = json.loads(t[lo:hi + 1])
    except json.JSONDecodeError as e:
        return None, f"невалидный JSON: {e.msg}"
    if not isinstance(obj, dict) or "is_scam" not in obj:
        return None, "нет поля is_scam"
    try:
        is_scam = int(obj["is_scam"])
    except (TypeError, ValueError):
        return None, "is_scam не число"
    if is_scam not in (0, 1):
        return None, f"is_scam={obj['is_scam']} вне {{0,1}}"

    subs = obj.get("subtypes") or []
    if isinstance(subs, str):
        subs = [subs]
    if not isinstance(subs, list):
        return None, "subtypes не список"
    subs = [s for s in subs if s in SUBTYPES]

    onset = obj.get("onset", None)
    if onset in ("", "null", "None"):
        onset = None
    if onset is not None:
        try:
            onset = int(onset)
        except (TypeError, ValueError):
            return None, "onset не целое"
        if not (1 <= onset <= n_turns):
            onset = None                      # вне диапазона реплик → считаем «не указан»
    if is_scam == 0:
        subs, onset = [], None
    return {"is_scam": is_scam, "subtypes": sorted(subs), "onset": onset}, None


# ──────────────────────────── прогон режима ────────────────────────────
def run_mode(client, mode, recs, examples, run_idx, cache_dir, concurrency, tag,
             split_name):
    """Один прогон режима по набору. Кэш на диске: повтор ничего не стоит.

    Имя файла включает СПЛИТ: dev и test держатся раздельно. Иначе один файл
    содержит оба набора, и любой разбор «по файлу» молча берёт 403 вместо 303.
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{tag}_{mode}_{split_name}_run{run_idx}.jsonl"
    done = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["dialog_id"]] = rec
    todo = [r for r in recs if r["id"] not in done]
    print(f"    [{mode} run{run_idx}] из кэша {len(done)}, к вызову {len(todo)}")

    lock = threading.Lock()
    fh = open(cache_file, "a", encoding="utf-8")

    def work(r):
        dialog = dialog_text(r)
        system, messages = build_messages(mode, dialog, examples)
        n_turns = len(r["turns"])
        raw, usage = client.complete(system, messages)
        parsed, err = parse_answer(raw, n_turns)
        retries = 0
        # одна повторная попытка на непарсящийся ответ — фиксируем ОБА показателя
        if parsed is None:
            retries = 1
            raw2, usage2 = client.complete(system, messages)
            usage = {"in": usage["in"] + usage2["in"], "out": usage["out"] + usage2["out"]}
            parsed2, err2 = parse_answer(raw2, n_turns)
            if parsed2 is not None:
                parsed, err, raw = parsed2, None, raw2
            else:
                raw, err = raw2, err2
        return {"dialog_id": r["id"], "lang": r["lang"], "raw": raw, "parsed": parsed,
                "parse_error": err, "retries": retries, "usage": usage}

    if todo:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for i, rec in enumerate(ex.map(work, todo), 1):
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[rec["dialog_id"]] = rec
                if i % 50 == 0:
                    print(f"      … {i}/{len(todo)}")
    fh.close()
    return [done[r["id"]] for r in recs]


# ─────────────────────────────── метрики ───────────────────────────────
def binary_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    k = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(k if np.isfinite(k) else 0.0)}


def subtype_metrics(gold_sets, pred_sets):
    from sklearn.metrics import precision_recall_fscore_support
    per_label, f1s = {}, []
    for s in SUBTYPES:
        yt = [int(s in g) for g in gold_sets]
        yp = [int(s in p) for p in pred_sets]
        p, r, f, _ = precision_recall_fscore_support(
            yt, yp, average="binary", pos_label=1, zero_division=0)
        per_label[s] = {"precision": float(p), "recall": float(r), "f1": float(f),
                        "n_gold": int(sum(yt)), "n_pred": int(sum(yp))}
        f1s.append(float(f))
    exact = float(np.mean([set(g) == set(p) for g, p in zip(gold_sets, pred_sets)]))
    return {"macro_f1": float(np.mean(f1s)), "per_label": per_label,
            "subset_accuracy": exact}


def onset_metrics(pairs):
    """pairs — список (t_gold, t_pred) на диалогах, где И gold, И LLM сказали scam."""
    if not pairs:
        return {"n": 0}
    t = np.array([a for a, _ in pairs], dtype=float)
    p = np.array([b for _, b in pairs], dtype=float)
    d = np.abs(p - t)
    return {"n": len(pairs), "oad": float(d.mean()), "exact": float((d == 0).mean()),
            "within1": float((d <= 1).mean()), "within2": float((d <= 2).mean()),
            "median_abs": float(np.median(d)), "bias": float((p - t).mean())}


def evaluate(answers, recs, gold, gold_sub, gold_onset, silver, langs):
    """Один прогон одного режима на одном наборе → все метрики."""
    parsed = [(a, r) for a, r in zip(answers, recs) if a["parsed"] is not None]
    n_all, n_ok = len(answers), len(parsed)
    first_pass_ok = sum(1 for a in answers if a["parsed"] is not None and a["retries"] == 0)

    y_true = np.array([gold[r["id"]] for _, r in parsed])
    y_pred = np.array([a["parsed"]["is_scam"] for a, _ in parsed])
    m = binary_metrics(y_true, y_pred)
    per_lang = {}
    for lg in langs:
        idx = [i for i, (_, r) in enumerate(parsed) if r["lang"] == lg]
        if idx:
            per_lang[lg] = binary_metrics(y_true[idx], y_pred[idx]) | {"n": len(idx)}
    sil = float(np.mean([a["parsed"]["is_scam"] == silver[r["id"]] for a, r in parsed]))

    # консервативный вариант: непарсящиеся считаем benign (нижняя граница)
    y_true_all = np.array([gold[r["id"]] for r in recs])
    y_pred_all = np.array([a["parsed"]["is_scam"] if a["parsed"] else 0 for a in answers])
    m_all = binary_metrics(y_true_all, y_pred_all)

    subs = subtype_metrics([set(gold_sub[r["id"]]) for _, r in parsed],
                           [set(a["parsed"]["subtypes"]) for a, _ in parsed])
    pairs = [(gold_onset[r["id"]], a["parsed"]["onset"]) for a, r in parsed
             if gold[r["id"]] == 1 and a["parsed"]["is_scam"] == 1
             and gold_onset.get(r["id"]) and a["parsed"]["onset"]]
    return {"n_total": n_all, "n_parsed": n_ok,
            "parse_rate": n_ok / n_all if n_all else 0.0,
            "parse_rate_first_attempt": first_pass_ok / n_all if n_all else 0.0,
            "n_retried": sum(a["retries"] for a in answers),
            "parse_errors": [a["parse_error"] for a in answers if a["parsed"] is None],
            "is_scam": m, "is_scam_unparsed_as_benign": m_all,
            "per_language": per_lang, "silver_agreement": sil,
            "subtypes": subs, "onset": onset_metrics(pairs)}


def mstd(vals):
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "values": v}


def agg(runs, path):
    """Среднее ± std по прогонам для метрики, заданной путём ключей."""
    out = []
    for r in runs:
        cur = r
        for k in path:
            cur = cur[k]
        out.append(cur)
    return mstd(out)


# ──────────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--gold", default="data/gold/gold_test_v2.csv")
    ap.add_argument("--gold-dev", default="data/gold/gold_dev_split.json")
    ap.add_argument("--examples", default="runs/layer4/few_shot_examples.json")
    ap.add_argument("--out-dir", default="runs/layer4")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--control", action="store_true",
                    help="прогон-контроль цикличности (DeepSeek): пишется отдельно")
    ap.add_argument("--test-best-only", action="store_true",
                    help="на тесте гонять только режим, выбранный по gold-dev (дешевле)")
    ap.add_argument("--key-file", default=None,
                    help="файл с ключом вне репозитория, если переменной окружения нет")
    ap.add_argument("--mock", action="store_true", help="заглушка вместо API")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    records = read_corpus(args.corpus)
    gold = read_gold_csv(args.gold)

    import csv
    gold_sub, gold_onset = {}, {}
    with open(args.gold, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            did = (row.get("dialog_id") or "").strip()
            if not did:
                continue
            raw = (row.get("subtypes") or "").strip()
            gold_sub[did] = [s.strip() for s in raw.split("|") if s.strip()] if raw else []
            o = (row.get("onset") or "").strip()
            gold_onset[did] = int(o) if o else None

    all_test = [r for r in records if r.get("split") == "test" and r["id"] in gold]
    if len(all_test) != len(gold):
        sys.exit(f"❌ джойн неполный: {len(all_test)} из {len(gold)}")
    sm = json.loads(pathlib.Path(args.gold_dev).read_text(encoding="utf-8"))
    assign = sm["assignment"]
    dev_recs = [r for r in all_test if assign.get(r["id"]) == "dev"]
    test_recs = [r for r in all_test if assign.get(r["id"]) != "dev"]
    assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"], \
        "разбиение gold не совпало с gold_dev_split.json"
    gd = {r.get("scenario_group") for r in dev_recs}
    gt = {r.get("scenario_group") for r in test_recs}
    assert not (gd & gt), "группы scenario_group разорваны между dev и test"

    silver = {r["id"]: int(r["is_scam"]) for r in records}
    langs = sorted({r["lang"] for r in test_recs})
    ex_payload = json.loads(pathlib.Path(args.examples).read_text(encoding="utf-8"))
    examples = ex_payload["examples"]
    assert len(examples) == 6, "ожидалось 6 few-shot примеров"

    client = (MockClient() if args.mock
              else LLMClient(args.provider, args.model, key_file=args.key_file))
    tag = "mock" if args.mock else f"{args.provider}_{args.model}".replace("/", "_")
    if args.control:
        tag += "_control"
    print(f"\n=== слой 4: {tag} | режимы {modes} | прогонов {args.runs} ===")
    print(f"разбиение gold: seed={sm['seed']} dev={len(dev_recs)} test={len(test_recs)} "
          f"(групп {len(gd)}/{len(gt)}, пересечение 0)")
    if not args.mock and args.provider == "deepseek" and not args.control:
        print("⚠ DeepSeek породил корпус и silver-метки. Как ОСНОВНОЙ бейзлайн он "
              "некорректен (согласие с самим собой). Запускай его с --control.")

    t0 = time.time()
    results = {"dev": {}, "test": {}}
    usage_total = {"in": 0, "out": 0, "calls": 0}

    def do(mode, split_name, recs):
        runs = []
        for k in range(1, args.runs + 1):
            answers = run_mode(client, mode, recs, examples, k,
                               out_dir / "cache", args.concurrency, tag, split_name)
            for a in answers:
                usage_total["in"] += a["usage"]["in"]
                usage_total["out"] += a["usage"]["out"]
                usage_total["calls"] += 1 + a["retries"]
            runs.append(evaluate(answers, recs, gold, gold_sub, gold_onset, silver, langs))
        results[split_name][mode] = {"runs": runs, "f1": agg(runs, ["is_scam", "f1"]),
                                     "parse_rate": agg(runs, ["parse_rate"])}
        f = results[split_name][mode]["f1"]
        pr = results[split_name][mode]["parse_rate"]
        print(f"  {mode:<10} {split_name:<5} F1 {f['mean']:.4f}±{f['std']:.4f}  "
              f"разбор {pr['mean']:.4f}")

    # СНАЧАЛА весь dev — режим выбирается только по нему, тест ещё не тронут
    for mode in modes:
        do(mode, "dev", dev_recs)

    # ── ВЫБОР РЕЖИМА — ТОЛЬКО ПО GOLD-DEV ──
    best_mode = max(modes, key=lambda m: results["dev"][m]["f1"]["mean"])
    print(f"\nлучший режим по gold-dev: {best_mode} "
          f"(dev F1 {results['dev'][best_mode]['f1']['mean']:.4f})")

    test_modes = [best_mode] if args.test_best_only else modes
    if args.test_best_only:
        print(f"--test-best-only: на тесте гоняется только {best_mode} "
              f"({len(test_recs) * args.runs} вызовов вместо "
              f"{len(test_recs) * args.runs * len(modes)})")
    for mode in test_modes:
        do(mode, "test", test_recs)

    bt = results["test"][best_mode]["runs"]
    summary = {k: agg(bt, ["is_scam", k]) for k in
               ("accuracy", "precision", "recall", "f1", "cohen_kappa")}
    summary["silver_agreement"] = agg(bt, ["silver_agreement"])
    summary["parse_rate"] = agg(bt, ["parse_rate"])
    summary["parse_rate_first_attempt"] = agg(bt, ["parse_rate_first_attempt"])
    summary["subtypes_macro_f1"] = agg(bt, ["subtypes", "macro_f1"])
    summary["onset_oad"] = agg(bt, ["onset", "oad"]) if bt[0]["onset"]["n"] else None

    print(f"\ngold-test v2 (n={len(test_recs)}), режим {best_mode}, "
          f"среднее ± std по {args.runs} прогонам:")
    for k, v in summary.items():
        if v:
            print(f"{k:>28} {v['mean']:>10.4f} {v['std']:>9.4f}")

    print(f"\nразбор ответов: {summary['parse_rate']['mean']:.4f} после ретрая, "
          f"{summary['parse_rate_first_attempt']['mean']:.4f} с первой попытки")
    errs = {}
    for r in bt:
        for e in r["parse_errors"]:
            errs[e] = errs.get(e, 0) + 1
    if errs:
        print("причины отказов разбора:")
        for e, c in sorted(errs.items(), key=lambda kv: -kv[1]):
            print(f"    {c:>4} × {e}")

    print("\nпо языкам (лучший режим):")
    for lg in langs:
        row = agg(bt, ["per_language", lg, "f1"])
        rec = agg(bt, ["per_language", lg, "recall"])
        pr = agg(bt, ["per_language", lg, "precision"])
        print(f"  {lg}: F1 {row['mean']:.4f}±{row['std']:.4f}  "
              f"P {pr['mean']:.4f}  R {rec['mean']:.4f}")

    print("\nподтипы (лучший режим, macro-F1 "
          f"{summary['subtypes_macro_f1']['mean']:.4f}):")
    for s in SUBTYPES:
        f1 = agg(bt, ["subtypes", "per_label", s, "f1"])
        p = agg(bt, ["subtypes", "per_label", s, "precision"])
        r_ = agg(bt, ["subtypes", "per_label", s, "recall"])
        print(f"  {s:<16} F1 {f1['mean']:.4f}±{f1['std']:.4f}  "
              f"P {p['mean']:.4f}  R {r_['mean']:.4f}")

    if bt[0]["onset"]["n"]:
        print(f"\nонсет (только там, где и gold, и LLM сказали scam; "
              f"n={bt[0]['onset']['n']}):")
        for k in ("oad", "exact", "within1", "bias"):
            v = agg(bt, ["onset", k])
            print(f"  {k:<10} {v['mean']:>8.4f} ± {v['std']:.4f}")

    elapsed = time.time() - t0
    print(f"\nвызовов {usage_total['calls']}, токенов вход {usage_total['in']}, "
          f"выход {usage_total['out']}, время {elapsed / 60:.1f} мин")

    payload = {
        "task": "layer 4 — LLM few-shot без обучения",
        "provider": client.provider, "model": client.model, "tag": tag,
        "is_control_run": bool(args.control),
        "temperature": 0.0, "n_runs": args.runs, "modes": modes,
        "gold_file": args.gold,
        "split": {"file": args.gold_dev, "seed": sm["seed"],
                  "n_dev": len(dev_recs), "n_test": len(test_recs)},
        "few_shot": {"file": args.examples, "n": len(examples),
                     "label_source": ex_payload["label_source"],
                     "warning": ex_payload["warning"],
                     "ids": [e["id"] for e in examples]},
        "best_mode_by_gold_dev": best_mode,
        "summary": summary,
        "dev": {m: {"f1": results["dev"][m]["f1"],
                    "parse_rate": results["dev"][m]["parse_rate"]} for m in modes},
        "test": {m: {"f1": results["test"][m]["f1"],
                     "parse_rate": results["test"][m]["parse_rate"],
                     "runs": results["test"][m]["runs"]} for m in results["test"]},
        "test_modes_run": list(results["test"]),
        "per_language": {lg: {k: agg(bt, ["per_language", lg, k])
                              for k in ("f1", "precision", "recall")} for lg in langs},
        "usage": usage_total, "elapsed_sec": elapsed,
    }
    name = "results_control.json" if args.control else "results.json"
    (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"\nсохранено: {out_dir / name}")

    if not args.no_plots and not args.control:
        try:
            from make_layer4_plots import make_plots
            make_plots(payload, out_dir / "plots")
        except FileNotFoundError as e:
            print(f"графики пропущены: {e}")


if __name__ == "__main__":
    main()
