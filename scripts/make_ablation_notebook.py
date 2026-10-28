#!/usr/bin/env python3
"""Генератор ноутбука `notebooks/ablation_context_kaggle.ipynb` (ablation контекста).

Ноутбук пишется скриптом, а не руками: так его легко переигрывать и видно диффом,
что поменялось. Логику разметки реплик и подвыборки он НЕ повторяет — читает готовые
списки из `train_manifest.json`, собранного `scripts/ablation_context.py`.

  python scripts/make_ablation_notebook.py
"""
import argparse
import json
import pathlib

MD_INTRO = """# Ablation диалогового контекста: диалог против одиночной реплики

**Зачем.** Первый заявленный вклад работы (tech_report §1) — детекция на уровне
ДИАЛОГА, а не отдельного сообщения. Он заявлен, но не обоснован: не показано, что
контекст вообще что-то даёт. Здесь меняется РОВНО ОДНО — единица обучения и
предсказания.

**Конфигурации.**

| имя | обучение | метка |
|---|---|---|
| `dialogue` | 1392 диалога | is_scam диалога — **эталон, здесь НЕ обучается** (0.9344 ± 0.0097) |
| `turn_naive_full` | 14003 реплики | метка диалога переносится на каждую реплику |
| `turn_naive_n1392` | 1392 реплики | то же, подвыборка до размера диалогового train |
| `turn_onset_full` | 14003 реплики | реплика scam, если её индекс ≥ t* диалога |
| `turn_onset_n1392` | 1392 реплики | то же, подвыборка |

**Что нельзя нарушать.**

- Гиперпараметры те же: lr 2e-5, wd 0.01, warmup 0.2, batch 16, max_length 512,
  5 эпох, patience 2, сиды 20260721/22/23.
- Отбор ТОЛЬКО по gold-dev 100 (`metric_for_best_model` смотрит на gold_dev).
  gold-test подключён третьим eval-набором ИСКЛЮЧИТЕЛЬНО ради снятия предсказаний
  по эпохам и в отборе не участвует.
- Сплиты не переразбиваются, gold-файлы только читаются.
- Контроль размера обязателен: без варианта на 1392 реплики эффект единицы обучения
  смешается с эффектом объёма (14003 против 1392, разница в 10 раз).

**Агрегация реплик в диалог.** Правило «хотя бы одна реплика scam» смещено к
высокому recall — чем длиннее диалог, тем выше шанс ложной тревоги. Поэтому
считаются три правила (`any`, доля `k` из T, максимум вероятности с порогом θ),
и лучшее выбирается ТОЛЬКО по gold-dev. Метрика отбора эпохи — лучшая по dev
агрегация, так что сохранённый чекпоинт по построению совпадает с выбранной эпохой.

**Отличие от диалоговых прогонов, вынужденное:** динамический паддинг по батчу
вместо `padding="max_length"`. Реплика — это ~30 токенов, а не 326; паддинг до 512
раздул бы прогон в разы. На математику это не влияет (attention mask тот же).

**Датасет Kaggle:** `scam-onset-ablation` (corpus_v1.jsonl, gold_test_v2.csv,
gold_dev_split.json, train_manifest.json). **Ускоритель: GPU T4 x2** — на P100
предустановленный PyTorch падает, на одной GPU меняется эффективный батч.
**Время:** ~1.5 ч.
"""

CELL_CFG = '''import os, sys, json, csv, glob, pathlib, time, copy, shutil
import numpy as np
import torch

def find_one(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"не найден {name} под /kaggle/input — подключён ли датасет?"
    return hits[0]

CORPUS   = find_one("corpus_v1.jsonl")
GOLD     = find_one("gold_test_v2.csv")
GOLD_DEV = find_one("gold_dev_split.json")
MANIFEST = find_one("train_manifest.json")

OUT    = pathlib.Path("/kaggle/working/ablation"); (OUT / "data").mkdir(parents=True, exist_ok=True)
MODELS = pathlib.Path("/kaggle/working/models");   MODELS.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "xlm-roberta-base"
SEEDS        = [20260721, 20260722, 20260723]
LR, WEIGHT_DECAY, WARMUP_RATIO = 2e-5, 0.01, 0.2
EPOCHS, PATIENCE, BATCH_SIZE, MAX_LENGTH = 5, 2, 16, 512
USE_FP16 = torch.cuda.is_available()

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет"
print("GPU:", gpu, "| устройств:", torch.cuda.device_count())
assert "T4" in gpu, f"нужен T4 x2, а выдали {gpu} — числа будут несопоставимы"
'''

CELL_DATA = '''def read_corpus(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def turn_text(t):
    return f"[{t['i']}|{t['speaker']}] {t['text']}"

records = read_corpus(CORPUS)
by_id   = {r["id"]: r for r in records}

gold = {}
with open(GOLD, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        did = (r.get("dialog_id") or "").strip()
        if did: gold[did] = int(r["is_scam"])

val_recs = [r for r in records if r.get("split") == "val"]
all_test = []
for r in records:
    if r.get("split") != "test": continue
    r2 = {k: v for k, v in r.items() if k != "is_scam"}   # silver-метку выбрасываем
    if r2["id"] in gold:
        r2["gold_is_scam"] = gold[r2["id"]]
        all_test.append(r2)
assert len(all_test) == len(gold), f"джойн неполный: {len(all_test)} из {len(gold)}"

sm = json.load(open(GOLD_DEV, encoding="utf-8")); assign = sm["assignment"]
dev_recs  = [r for r in all_test if assign.get(r["id"]) == "dev"]
test_recs = [r for r in all_test if assign.get(r["id"]) != "dev"]
assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"], "разбиение не совпало"
assert not ({r.get("scenario_group") for r in dev_recs} &
            {r.get("scenario_group") for r in test_recs}), "группы dev/test разорваны"

manifest = json.load(open(MANIFEST, encoding="utf-8"))
CONFIGS  = manifest["configs"]
assert manifest["seeds"] == SEEDS

# ── eval-наборы: ВСЕ реплики диалогов, метка реплики = метка её диалога ──
def flat_turns(recs, label_key):
    texts, labels, dids, idxs = [], [], [], []
    for r in sorted(recs, key=lambda x: x["id"]):
        for t in r["turns"]:
            texts.append(turn_text(t)); labels.append(int(r[label_key]))
            dids.append(r["id"]); idxs.append(int(t["i"]))
    return {"texts": texts, "labels": labels, "dialog_ids": dids, "turn_idx": idxs}

EVAL = {
    "gold_dev":   flat_turns(dev_recs,  "gold_is_scam"),
    "silver_val": flat_turns(val_recs,  "is_scam"),
    "gold_test":  flat_turns(test_recs, "gold_is_scam"),
}
sizes = {k: len(v["texts"]) for k, v in EVAL.items()}
assert len(set(sizes.values())) == 3, f"размеры eval-наборов совпали: {sizes}"
print("реплик в eval-наборах:", sizes)

DLG_ORDER = {k: sorted({d for d in v["dialog_ids"]}) for k, v in EVAL.items()}
DLG_GOLD  = {"gold_dev":  {r["id"]: r["gold_is_scam"] for r in dev_recs},
             "gold_test": {r["id"]: r["gold_is_scam"] for r in test_recs},
             "silver_val": {r["id"]: int(r["is_scam"]) for r in val_recs}}
for cfg in CONFIGS:
    c = cfg["composition"]
    print(f"  {cfg['name']:<22} реплик {c['n']:>6}  доля scam {c['scam_share']:.3f}")
'''

CELL_AGG = '''# ── правила агрегации: реплики → диалог. Выбор параметра только по gold-dev ──
FRAC_GRID    = [round(x, 3) for x in np.arange(0.02, 1.001, 0.02)]
MAXPROB_GRID = [round(x, 2) for x in np.arange(0.01, 1.0, 0.01)]

def rule_grid():
    return [("any", None)] + [("frac", k) for k in FRAC_GRID] + \\
           [("maxprob", t) for t in MAXPROB_GRID]

def aggregate(p, rule, param):
    a = np.asarray(p, dtype=float)
    if rule == "any":     return int((a >= 0.5).any())
    if rule == "frac":    return int(float((a >= 0.5).mean()) >= param)
    if rule == "maxprob": return int(float(a.max()) >= param)
    raise ValueError(rule)

def f1_np(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

def group_probs(setname, probs):
    """Плоский массив вероятностей реплик → {dialog_id: [p_1..p_T]}."""
    out = {}
    for did, p in zip(EVAL[setname]["dialog_ids"], probs):
        out.setdefault(did, []).append(float(p))
    return out

def agg_f1(setname, probs, rule, param):
    g = group_probs(setname, probs)
    ids = DLG_ORDER[setname]
    y = [DLG_GOLD[setname][d] for d in ids]
    return f1_np(y, [aggregate(g[d], rule, param) for d in ids])

def best_agg(setname, probs):
    """Лучшее (правило, параметр) по F1 на этом наборе. Тай-брейк — порядок сетки."""
    g = group_probs(setname, probs)
    ids = DLG_ORDER[setname]
    y = np.array([DLG_GOLD[setname][d] for d in ids])
    best = (-1.0, None, None)
    for rule, param in rule_grid():
        f = f1_np(y, np.array([aggregate(g[d], rule, param) for d in ids]))
        if f > best[0]:
            best = (f, rule, param)
    return best
'''

CELL_TRAIN = '''from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)   # динамический паддинг

class TurnDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = [int(x) for x in labels]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]; return item

DS_EVAL = {k: TurnDataset(v["texts"], v["labels"]) for k, v in EVAL.items()}
SIZE2SET = {len(v["texts"]): k for k, v in EVAL.items()}

SNAP = {}          # {имя набора: [вероятности по эпохам]}

def compute_metrics(ep):
    logits, labels = ep[0], ep[1]
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
    setname = SIZE2SET[len(probs)]
    SNAP.setdefault(setname, []).append(np.round(probs, 6).tolist())
    turn_f1 = f1_np(np.asarray(labels), (probs >= 0.5).astype(int))
    f_best, rule, param = best_agg(setname, probs)
    return {"turn_f1": float(turn_f1), "any_f1": float(agg_f1(setname, probs, "any", None)),
            "bestagg_f1": float(f_best),
            "best_rule": {"any": 0.0, "frac": 1.0, "maxprob": 2.0}[rule],
            "best_param": float(param if param is not None else -1.0)}

def build_train_ds(cfg):
    texts, labels = [], []
    for did, i, lab in cfg["turns"]:
        t = next(x for x in by_id[did]["turns"] if int(x["i"]) == i)
        texts.append(turn_text(t)); labels.append(int(lab))
    return TurnDataset(texts, labels)

def run(cfg, seed, save_weights=False):
    """Один прогон. save_weights=False → чекпоинты не пишутся, эпоха выбирается
    постфактум из снимков; True → лучший чекпоинт по gold-dev сохраняется в models/.

    ⚠ В логе дважды за эпоху будет "early stopping required metric_for_best_model,
    but did not find eval_gold_dev_bestagg_f1 so early stopping is disabled". Это ШУМ:
    коллбэк вызывается после КАЖДОГО из трёх eval-наборов и молча выходит на
    silver_val и gold_test, где своей метрики нет. На gold_dev он отрабатывает."""
    global SNAP
    SNAP = {}
    set_seed(seed)
    t0 = time.time()
    ds_train = build_train_ds(cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32)
    ckpt = OUT / f"ckpt_{cfg['name']}_{seed}"
    targs = TrainingArguments(
        output_dir=str(ckpt),
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=64,
        seed=seed, data_seed=seed, fp16=USE_FP16,
        eval_strategy="epoch",
        save_strategy="epoch" if save_weights else "no",
        load_best_model_at_end=save_weights, save_total_limit=2,
        metric_for_best_model="eval_gold_dev_bestagg_f1", greater_is_better=True,
        logging_steps=50, report_to="none", disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train,
                      eval_dataset=DS_EVAL, processing_class=tok, data_collator=collator,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
    trainer.train()
    hist = copy.deepcopy(trainer.state.log_history)
    n_ep = len(SNAP["gold_dev"])
    assert all(len(v) == n_ep for v in SNAP.values()), \\
        f"снимки рассинхронизированы: {[(k, len(v)) for k, v in SNAP.items()]}"
    dev_f1 = [float(h["eval_gold_dev_bestagg_f1"]) for h in hist
              if "eval_gold_dev_bestagg_f1" in h]
    assert len(dev_f1) == n_ep, f"эпох в логе {len(dev_f1)}, снимков {n_ep}"
    best_epoch = int(np.argmax(dev_f1)) + 1              # тай-брейк — ранняя эпоха
    rec = {"config": cfg["name"], "seed": seed, "epochs": n_ep,
           "dev_bestagg_f1_by_epoch": dev_f1, "best_epoch": best_epoch,
           "minutes": round((time.time() - t0) / 60, 2),
           "n_train_turns": len(ds_train), "history": hist,
           "snapshots": {k: v for k, v in SNAP.items()}}
    if save_weights:
        dest = MODELS / f"{cfg['name']}_seed{seed}"
        trainer.save_model(str(dest)); tok.save_pretrained(str(dest))
        rec["weights"] = str(dest)
    shutil.rmtree(ckpt, ignore_errors=True)
    del model, trainer; torch.cuda.empty_cache()
    print(f"  {cfg['name']} сид {seed}: эпох {n_ep}, лучшая {best_epoch}, "
          f"dev bestagg F1 {dev_f1[best_epoch-1]:.4f}, {rec['minutes']} мин")
    return rec
'''

CELL_PHASE1 = '''# ── фаза 1: все 12 прогонов без сохранения весов ──
runs = []
t_all = time.time()
for cfg in CONFIGS:
    print(f"\\n=== {cfg['name']} ({cfg['composition']['n']} реплик) ===")
    for seed in SEEDS:
        r = run(cfg, seed, save_weights=False)
        runs.append(r)
        with open(OUT / "data" / f"{cfg['name']}_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False)
print(f"\\nфаза 1: {len(runs)} прогонов за {(time.time()-t_all)/60:.1f} мин")
'''

CELL_PICK = '''# ── какая turn-конфигурация лучшая ПО GOLD-DEV (тест в отборе не участвует) ──
by_cfg = {}
for r in runs:
    dev = r["dev_bestagg_f1_by_epoch"][r["best_epoch"] - 1]
    by_cfg.setdefault(r["config"], []).append(dev)
dev_mean = {k: float(np.mean(v)) for k, v in by_cfg.items()}
for k, v in sorted(dev_mean.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} gold-dev F1 {v:.4f}  (по сидам {[round(x,4) for x in by_cfg[k]]})")
BEST_CFG = max(dev_mean, key=lambda k: (dev_mean[k], k))
print("\\nлучшая turn-конфигурация по gold-dev:", BEST_CFG)
'''

CELL_PHASE2 = '''# ── фаза 2: переобучение лучшей конфигурации С СОХРАНЕНИЕМ ВЕСОВ ──
# (конвенция CLAUDE.md: прогон, чьи числа идут в отчёт, обязан оставить веса на диске)
cfg_best = next(c for c in CONFIGS if c["name"] == BEST_CFG)
retrain = []
for seed in SEEDS:
    r = run(cfg_best, seed, save_weights=True)
    ref = next(x for x in runs if x["config"] == BEST_CFG and x["seed"] == seed)
    r["phase1_best_epoch"] = ref["best_epoch"]
    r["phase1_dev_f1"] = ref["dev_bestagg_f1_by_epoch"][ref["best_epoch"] - 1]
    r["reproduced"] = bool(abs(r["dev_bestagg_f1_by_epoch"][r["best_epoch"] - 1]
                               - r["phase1_dev_f1"]) <= 0.01)
    retrain.append(r)
    with open(OUT / "data" / f"retrain_{BEST_CFG}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False)
    print(f"    фаза 1: эпоха {r['phase1_best_epoch']}, dev {r['phase1_dev_f1']:.4f} → "
          f"фаза 2: эпоха {r['best_epoch']} "
          f"{'воспроизвелось' if r['reproduced'] else 'РАСХОЖДЕНИЕ'}")
print("\\nвеса:", [p.name for p in sorted(MODELS.iterdir())])
'''

CELL_SAVE = '''payload = {
    "model": MODEL_NAME, "seeds": SEEDS,
    "hyperparams": {"lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                    "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                    "max_length": MAX_LENGTH, "padding": "dynamic", "fp16": USE_FP16},
    "eval_sizes": sizes,
    "dialog_order": DLG_ORDER,
    "dialog_gold": DLG_GOLD,
    "turn_index": {k: EVAL[k]["turn_idx"] for k in EVAL},
    "turn_dialog_ids": {k: EVAL[k]["dialog_ids"] for k in EVAL},
    "configs": [{k: v for k, v in c.items() if k != "turns"} for c in CONFIGS],
    "dev_mean_by_config": dev_mean, "best_config": BEST_CFG,
    "runs": [{k: v for k, v in r.items() if k != "snapshots"} for r in runs],
    "retrain": [{k: v for k, v in r.items() if k != "snapshots"} for r in retrain],
    "gpu": gpu, "device_count": torch.cuda.device_count(),
    "versions": {"torch": torch.__version__},
}
with open(OUT / "results_raw.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)
print("сохранено:", OUT / "results_raw.json")
print("файлы:", sorted(p.name for p in (OUT / "data").iterdir()))
'''


def nb(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"},
                     "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}


MD_WEIGHTS = """# Веса конфигурации `{cfg}` (добор к основному прогону ablation)

**Зачем.** Основной прогон (`ablation_context_kaggle.ipynb`) сохранил веса той
конфигурации, которую выбрал gold-dev. Но в CLAUDE.md как результат модели идут
числа и другой конфигурации, а конвенция проекта требует, чтобы прогон, чьи числа
попадают в отчёт, оставил веса на диске. Здесь ровно те же 3 сида той же
конфигурации переобучаются с `save_strategy="epoch"`, `load_best_model_at_end=True`
и явным `save_model`.

**Фазы 1 нет:** конфигурация задана жёстко, отбор эпохи — тот же критерий
(лучшая агрегация на gold-dev). Ожидается воспроизведение эпох и dev-F1 основного
прогона; расхождение печатается явно.
"""

CELL_FORCE = '''# ── конфигурация задана жёстко, фаза 1 не нужна ──
BEST_CFG = "{cfg}"
assert any(c["name"] == BEST_CFG for c in CONFIGS), BEST_CFG
runs, dev_mean = [], {{}}
print("сохраняем веса для:", BEST_CFG)
'''

CELL_PHASE2_SOLO = '''cfg_best = next(c for c in CONFIGS if c["name"] == BEST_CFG)
retrain = []
for seed in SEEDS:
    r = run(cfg_best, seed, save_weights=True)
    retrain.append(r)
    with open(OUT / "data" / f"weights_{BEST_CFG}_seed{seed}.json", "w",
              encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False)
print("\\nвеса:", [p.name for p in sorted(MODELS.iterdir())])
'''

CELL_SAVE_SOLO = '''payload = {
    "model": MODEL_NAME, "seeds": SEEDS, "config": BEST_CFG,
    "hyperparams": {"lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                    "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                    "max_length": MAX_LENGTH, "padding": "dynamic", "fp16": USE_FP16},
    "runs": [{k: v for k, v in r.items() if k != "snapshots"} for r in retrain],
    "gpu": gpu, "device_count": torch.cuda.device_count(),
}
with open(OUT / "weights_results.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)
print("сохранено:", OUT / "weights_results.json")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notebooks/ablation_context_kaggle.ipynb")
    ap.add_argument("--save-config", default=None,
                    help="только добор весов для одной конфигурации, без фазы 1")
    args = ap.parse_args()
    if args.save_config:
        cells = [md(MD_WEIGHTS.format(cfg=args.save_config)), code(CELL_CFG),
                 code(CELL_DATA), code(CELL_AGG), code(CELL_TRAIN),
                 code(CELL_FORCE.format(cfg=args.save_config)),
                 code(CELL_PHASE2_SOLO), code(CELL_SAVE_SOLO)]
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"записан {out} (только веса {args.save_config})")
        return
    cells = [md(MD_INTRO), code(CELL_CFG), code(CELL_DATA), code(CELL_AGG),
             code(CELL_TRAIN),
             md("## Фаза 1 — 12 прогонов (4 конфигурации × 3 сида), веса не пишутся\n"),
             code(CELL_PHASE1), code(CELL_PICK),
             md("## Фаза 2 — лучшая конфигурация переобучается с сохранением весов\n"),
             code(CELL_PHASE2), code(CELL_SAVE)]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записан {out}")


if __name__ == "__main__":
    main()
