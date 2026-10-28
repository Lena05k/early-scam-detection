"""Стадия 1: управляемый синтез диалогов из скелетов (методичка §9.2).

Пишет data/corpus/pilot_v1.jsonl (схема проекта + gold_src="llm_draft" — черновик
для human-LLM верификации, стадия 2) и pilot_v1_rejected.jsonl (брак с причинами).

Пилот: один генератор допустим. ПОЛНЫЙ корпус требует >=2 генераторов (методичка
§9.4: «генераторный артефакт») — список generation.models в configs/default.yaml,
ротация по списку уже встроена.

Слой чистый stdlib + pyyaml (конвенция CLAUDE.md): Anthropic API — через urllib,
без SDK и ML-зависимостей.
"""
import json
import os
import pathlib
import urllib.error
import urllib.request
from collections import Counter

import yaml

from .prompts import build_prompt
from .scenarios import SCENARIOS, SUBTYPES

API_URL = "https://api.anthropic.com/v1/messages"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Ключ окружения по вендору + список имён-моделей вендора для маршрутизации.
ENV_KEY = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


class GenerationError(RuntimeError):
    pass


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class AnthropicClient:
    """Минимальный клиент /v1/messages на stdlib."""

    def __init__(self, api_key, api_cfg=None):
        api_cfg = api_cfg or {}
        self.api_key = api_key
        self.version = api_cfg.get("version", "2023-06-01")
        self.max_tokens = int(api_cfg.get("max_tokens", 4096))
        self.temperature = float(api_cfg.get("temperature", 1.0))
        self.timeout = float(api_cfg.get("timeout_s", 180))

    @classmethod
    def from_env(cls, api_cfg=None):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise GenerationError(
                "ANTHROPIC_API_KEY не задан — генерация зовёт Anthropic API и без ключа "
                "невозможна. Экспортируй ключ (export ANTHROPIC_API_KEY=sk-ant-...) и "
                "перезапусти. Тесты и валидация работают без ключа (на моках)."
            )
        return cls(key, api_cfg)

    def complete(self, model, system, user):
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise GenerationError(f"Anthropic API HTTP {e.code}: {e.read().decode()[:500]}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"Anthropic API недоступен: {e.reason}") from e
        except (OSError, ValueError) as e:
            # таймаут чтения / обрыв соединения / битый JSON — ТРАНЗИЕНТНО.
            # Оборачиваем в GenerationError, чтобы решал backoff, а не падал весь прогон.
            raise GenerationError(
                f"Anthropic API: сбой соединения/ответа: {type(e).__name__}: {e}") from e
        return "".join(b.get("text", "") for b in data.get("content", []))


class DeepSeekClient:
    """Минимальный клиент DeepSeek /chat/completions (OpenAI-совместимый) на stdlib.

    Тот же интерфейс complete(model, system, user), что у AnthropicClient, — чтобы
    вставать в ту же ротацию генераторов (§9.4). Auth: Bearer; system идёт отдельной
    репликой role=system. Модели: deepseek-chat, deepseek-reasoner.
    """

    def __init__(self, api_key, api_cfg=None):
        api_cfg = api_cfg or {}
        self.api_key = api_key
        self.max_tokens = int(api_cfg.get("max_tokens", 4096))
        self.temperature = float(api_cfg.get("temperature", 1.0))
        self.timeout = float(api_cfg.get("timeout_s", 180))

    @classmethod
    def from_env(cls, api_cfg=None):
        key = os.environ.get(ENV_KEY["deepseek"])
        if not key:
            raise GenerationError(
                "DEEPSEEK_API_KEY не задан — генерация через DeepSeek без ключа невозможна. "
                "Экспортируй ключ (export DEEPSEEK_API_KEY=sk-...) и перезапусти. Тесты и "
                "валидация работают без ключа (на моках)."
            )
        return cls(key, api_cfg)

    def complete(self, model, system, user):
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            DEEPSEEK_API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise GenerationError(f"DeepSeek API HTTP {e.code}: {e.read().decode()[:500]}") from e
        except urllib.error.URLError as e:
            raise GenerationError(f"DeepSeek API недоступен: {e.reason}") from e
        except (OSError, ValueError) as e:
            # таймаут чтения / обрыв соединения / битый JSON — ТРАНЗИЕНТНО.
            # Оборачиваем в GenerationError, чтобы решал backoff, а не падал весь прогон.
            raise GenerationError(
                f"DeepSeek API: сбой соединения/ответа: {type(e).__name__}: {e}") from e
        choices = data.get("choices") or []
        if not choices:
            raise GenerationError(f"DeepSeek API: пустой ответ {str(data)[:300]}")
        return choices[0].get("message", {}).get("content", "") or ""


def vendor_of(model):
    """Вендор по имени модели — для маршрутизации в мульти-генераторной ротации."""
    m = str(model).lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("deepseek"):
        return "deepseek"
    raise GenerationError(
        f"неизвестный вендор для модели {model!r}: ожидались имена claude-* или deepseek-*"
    )


_CLIENT_CLS = {"anthropic": AnthropicClient, "deepseek": DeepSeekClient}


class RoutedClient:
    """Роутер: по имени модели зовёт нужного вендор-клиента (Claude ↔ DeepSeek).

    Держит по одному клиенту на вендор; complete(model, …) выбирает клиента через
    vendor_of(model). Так один список generation.models с моделями разных вендоров
    крутится в общей ротации пайплайна без изменений run_pipeline/run_corpus.
    """

    def __init__(self, clients):
        self._clients = dict(clients)              # vendor -> client

    def complete(self, model, system, user):
        vendor = vendor_of(model)
        client = self._clients.get(vendor)
        if client is None:
            raise GenerationError(
                f"нет клиента для вендора {vendor!r} (модель {model!r}); "
                f"проверь ключ {ENV_KEY.get(vendor, '?')}"
            )
        return client.complete(model, system, user)


def build_client(models, api_cfg=None):
    """Собирает клиент под список моделей: один вендор -> его клиент, несколько -> RoutedClient.

    Ключи берутся ТОЛЬКО из окружения (ANTHROPIC_API_KEY / DEEPSEEK_API_KEY). Если для
    какого-то нужного вендора ключа нет — единая понятная ошибка со списком недостающих.
    """
    vendors = {vendor_of(m) for m in models}       # vendor_of поднимет ошибку на неизвестной модели
    clients, missing = {}, []
    for v in sorted(vendors):
        key = os.environ.get(ENV_KEY[v])
        if not key:
            missing.append(ENV_KEY[v])
            continue
        clients[v] = _CLIENT_CLS[v](key, api_cfg)
    if missing:
        raise GenerationError(
            "не заданы ключи окружения: " + ", ".join(missing) + ". "
            "Мульти-генератор (§9.4) требует ключ на КАЖДОГО вендора из generation.models. "
            "Экспортируй недостающие (export ИМЯ=...) и перезапусти."
        )
    return RoutedClient(clients)


def extract_json(text):
    """Достаёт JSON-объект из ответа LLM (терпит преамбулы и ```-ограждения)."""
    t = text.strip()
    if "```" in t:
        # берём содержимое первого ограждения, срезав возможный маркер языка
        t = t.split("```", 2)[1]
        t = t.split("\n", 1)[1] if t.startswith(("json", "JSON")) else t
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError("в ответе нет JSON-объекта")
    return json.loads(t[lo:hi + 1])


def build_record(skel, parsed, model):
    """Ответ LLM -> запись в схеме проекта (как data/synth) + провенанс черновика."""
    labels = parsed.get("draft_labels") or {}
    onset = labels.get("onset")
    if isinstance(onset, str):
        onset = int(onset) if onset.isdigit() else None
    turns = [
        {
            "i": t.get("i"),
            "role": t.get("role"),
            "speaker": str(t.get("speaker", "")),
            "text": str(t.get("text", "")),
        }
        for t in (parsed.get("turns") or [])
    ]
    sc = SCENARIOS[skel.scenario_id]
    return {
        "item_id": None,                       # проставляется по порядку при записи
        "id": skel.sid,
        "lang": skel.lang,
        "scam_type": skel.scenario_id,
        "turns": turns,
        "is_scam": labels.get("is_scam"),
        "subtypes": list(labels.get("subtypes") or []),
        "other_note": sc.other_note,
        "onset": onset,
        "hard_negative": skel.hard_negative,
        "onset_cand": skel.onset_cand,         # план стадии 0; НЕ золото (методичка §9.4)
        "twin_id": skel.twin_sid,
        "gold_src": "llm_draft",               # черновик; человек верифицирует (стадия 2)
        "generator": model,
    }


def validate_record(rec, skel, turn_min, turn_max):
    """Инварианты диалога; возвращает список причин брака (пусто = принят)."""
    reasons = []
    turns = rec.get("turns") or []
    T = len(turns)
    if not turn_min <= T <= turn_max:
        reasons.append(f"длина {T} вне [{turn_min}, {turn_max}]")
    if [t.get("i") for t in turns] != list(range(1, T + 1)):
        reasons.append("нумерация реплик не сплошная 1..T")
    bad_roles = {t.get("role") for t in turns} - {"atk", "tgt"}
    if bad_roles:
        reasons.append(f"недопустимые role: {sorted(map(str, bad_roles))}")

    # разделение сторон: ровно 2 speaker, маппинг speaker -> role постоянен
    speaker_role = {}
    for t in turns:
        sp, role = t.get("speaker"), t.get("role")
        if sp in speaker_role and speaker_role[sp] != role:
            reasons.append(f"speaker «{sp}» сменил role ({speaker_role[sp]} -> {role})")
            break
        speaker_role.setdefault(sp, role)
    if turns and len(speaker_role) != 2:
        reasons.append(f"уникальных speaker {len(speaker_role)}, должно быть ровно 2")

    if rec.get("is_scam") != skel.is_scam:
        reasons.append(f"is_scam черновика ({rec.get('is_scam')}) != сценария ({skel.is_scam})")
    unknown = set(rec.get("subtypes") or []) - set(SUBTYPES)
    if unknown:
        reasons.append(f"неизвестные подтипы: {sorted(unknown)}")

    onset = rec.get("onset")
    if skel.is_scam:
        if not any(t.get("role") == "atk" for t in turns):
            reasons.append("scam без единой atk-реплики")
        if not rec.get("subtypes"):
            reasons.append("scam без подтипов (правило консистентности v1.2)")
        if not isinstance(onset, int) or not 1 <= onset <= T:
            reasons.append(f"scam с невалидным онсетом: {onset!r}")
        elif turns[onset - 1].get("role") != "atk":
            reasons.append(f"онсет t*={onset} не на atk-реплике")
    else:
        if onset is not None:
            reasons.append(f"benign с онсетом {onset!r} (инвариант: benign -> ∅)")
        if rec.get("subtypes"):
            reasons.append("benign с подтипами (правило консистентности v1.2)")
    return reasons


def render_dialogue(rec):
    """Человекочитаемый рендер для верификатора: сразу видно, кто отвечает и где черновой t*."""
    subs = ", ".join(rec["subtypes"]) or "—"
    head = (f"— {rec['id']} | {rec['lang']} | {rec['scam_type']} | is_scam={rec['is_scam']} "
            f"| subtypes: {subs} | onset={rec['onset']} (ЧЕРНОВИК, gold_src={rec['gold_src']})")
    lines = [head]
    for t in rec["turns"]:
        mark = "  ← ОНСЕТ (черновик)" if rec["onset"] == t["i"] else ""
        lines.append(f"[{t['i']}] {t['speaker']} ({t['role']}): {t['text']}{mark}")
    return "\n".join(lines)


def run_pipeline(skeletons, cfg, client, out_dir=None, log=None):
    """Скелеты -> LLM -> валидация -> pilot_v1.jsonl / pilot_v1_rejected.jsonl + отчёт."""
    gen = cfg["generation"]
    turn_min, turn_max = gen["turns"]["min"], gen["turns"]["max"]
    models = gen["models"]
    max_attempts = int(gen.get("api", {}).get("max_attempts", 2))
    out = pathlib.Path(out_dir or cfg["paths"]["corpus"])
    out.mkdir(parents=True, exist_ok=True)

    accepted, rejected = [], []
    for idx, skel in enumerate(skeletons):
        model = models[idx % len(models)]      # ротация мульти-генератора (§9.4)
        system, user = build_prompt(skel)
        last_reasons, raw = ["генерация не запускалась"], ""
        for _ in range(max_attempts):
            raw = client.complete(model, system, user)
            try:
                parsed = extract_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last_reasons = [f"JSON не распарсен: {e}"]
                continue
            rec = build_record(skel, parsed, model)
            last_reasons = validate_record(rec, skel, turn_min, turn_max)
            if not last_reasons:
                accepted.append(rec)
                break
        else:
            rejected.append({
                "id": skel.sid, "lang": skel.lang, "scenario": skel.scenario_id,
                "reasons": last_reasons, "raw": raw[:2000],
            })
        if log:
            log(f"[{idx + 1}/{len(skeletons)}] {skel.sid}: "
                + ("ok" if not last_reasons else f"REJECT ({'; '.join(last_reasons)})"))

    for i, rec in enumerate(accepted, 1):
        rec["item_id"] = i
    corpus_path = out / "pilot_v1.jsonl"
    rejected_path = out / "pilot_v1_rejected.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return accepted, rejected, _report(accepted, rejected, models)


def _report(accepted, rejected, models):
    n = len(accepted)
    scam = sum(r["is_scam"] for r in accepted)
    return {
        "n_accepted": n,
        "n_rejected": len(rejected),
        "by_scenario": dict(Counter(r["scam_type"] for r in accepted)),
        "by_lang": dict(Counter(r["lang"] for r in accepted)),
        "scam": scam,
        "benign": n - scam,
        "benign_share": round((n - scam) / n, 3) if n else None,
        "hard_negatives": sum(r["hard_negative"] for r in accepted),
        "mean_turns": round(sum(len(r["turns"]) for r in accepted) / n, 2) if n else None,
        "reject_reasons": dict(Counter(x for r in rejected for x in r["reasons"])),
        "models": list(models),
        "multi_generator_ok": len(models) >= 2,  # False допустимо ТОЛЬКО для пилота (§9.4)
    }
