#!/usr/bin/env python3
"""Генератор `notebooks/retrain_save_kaggle.ipynb`.

Восстанавливает веса основной бинарной модели (раздел «Три сида, отбор по gold-dev»)
и сразу считает на ней префиксные вероятности для раздела «Метрики ранней детекции».

Зачем: раздел про раннюю детекцию был посчитан на единственном уцелевшем чекпоинте —
эпохе 3 первого baseline (отбор по silver-val, gold-test F1 0.8680), потому что веса
модели с 0.9344 не сохранились (`save_total_limit=1`, затем `save_strategy="no"`).

Конфигурация повторяет «Три сида» ТОЧНО, кроме сохранения:
  save_strategy="epoch", load_best_model_at_end=True, save_total_limit=2.

  python scripts/make_retrain_notebook.py
"""
import argparse
import json
import pathlib

MD_INTRO = """# Переобучение с сохранением весов + префиксные вероятности

**Зачем.** Веса основной бинарной модели (раздел «Три сида, отбор по gold-dev»,
gold-test F1 0.9375 ± 0.0086 на v1) не сохранились: тот прогон шёл с
`save_total_limit=1`, а более поздние — с `save_strategy="no"`. Из-за этого раздел
«Метрики ранней детекции» пришлось считать на чужом чекпоинте — эпохе 3 первого
baseline с F1 0.8680. Здесь модель восстанавливается и на ней же считаются
префиксные вероятности.

**Конфигурация повторяет «Три сида» точно:** lr 2e-5, wd 0.01, warmup 0.2, batch 16,
max_length 512, 5 эпох, patience=2, отбор по `eval_gold_dev_f1`, сиды
20260721/22/23. Единственное отличие — сохранение: `save_strategy="epoch"`,
`load_best_model_at_end=True`, `save_total_limit=2`.

**Ожидаемое (эталон из CLAUDE.md, gold v1):** лучшая эпоха 1 на всех трёх сидах,
gold-test F1 0.9415 / 0.9276 / 0.9433. Здесь замер идёт на **gold v2**, где среднее
по трём сидам было 0.9344 против 0.9375 на v1 — то есть ждём примерно 0.938 / 0.925
/ 0.940. Расхождение больше 0.01 от эталона означает, что прогон не воспроизвёлся.

⚠ **Ускоритель — GPU T4 x2.** На P100 предустановленный PyTorch падает
(`cudaErrorNoKernelImageForDevice`), а на одной GPU меняется эффективный батч
(16 против 32) и числа перестают быть сопоставимыми. Выставить в UI: Session
options → Accelerator → GPU T4 x2.

**Датасет:** `scam-onset-crosslingual` (corpus_v1.jsonl, gold_test_v2.csv,
gold_dev_split.json). **Время:** ~5 мин обучения на сид + ~1 мин префиксов.
"""

CELL_CFG = '''import os, sys, json, csv, glob, pathlib, time, copy, shutil
import torch

def find_one(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"не найден {name} под /kaggle/input"
    return hits[0]

CORPUS   = find_one("corpus_v1.jsonl")
GOLD     = find_one("gold_test_v2.csv")
GOLD_DEV = find_one("gold_dev_split.json")

OUT    = pathlib.Path("/kaggle/working/retrain"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = pathlib.Path("/kaggle/working/models");  MODELS.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "xlm-roberta-base"
SEEDS        = [20260721, 20260722, 20260723]
LR, WEIGHT_DECAY, WARMUP_RATIO = 2e-5, 0.01, 0.2
EPOCHS, PATIENCE, BATCH_SIZE, MAX_LENGTH = 5, 2, 16, 512
USE_FP16 = torch.cuda.is_available()

# эталон из CLAUDE.md (gold v1) — для проверки воспроизводимости
REFERENCE_V1 = {20260721: 0.9415, 20260722: 0.9276, 20260723: 0.9433}
TOLERANCE = 0.01

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

def build_text(rec):
    return "\\n".join(f"[{t['i']}|{t['speaker']}] {t['text']}" for t in rec["turns"])

def prefix_text(rec, upto):
    return "\\n".join(f"[{t['i']}|{t['speaker']}] {t['text']}" for t in rec["turns"][:upto])

records = read_corpus(CORPUS)
gold, gold_onset = {}, {}
with open(GOLD, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        did = (r.get("dialog_id") or "").strip()
        if did:
            gold[did] = int(r["is_scam"])
            o = (r.get("onset") or "").strip()
            gold_onset[did] = int(o) if o else None

train_recs = [r for r in records if r.get("split") == "train"]
val_recs   = [r for r in records if r.get("split") == "val"]
all_test = []
for r in records:
    if r.get("split") != "test": continue
    r2 = {k: v for k, v in r.items() if k != "is_scam"}
    if r2["id"] in gold:
        r2["gold_is_scam"] = gold[r2["id"]]
        all_test.append(r2)
assert len(all_test) == len(gold), f"джойн неполный: {len(all_test)} из {len(gold)}"

sm = json.load(open(GOLD_DEV, encoding="utf-8")); assign = sm["assignment"]
dev_recs  = [r for r in all_test if assign.get(r["id"]) == "dev"]
test_recs = [r for r in all_test if assign.get(r["id"]) != "dev"]
assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"]
assert not ({r.get("scenario_group") for r in dev_recs} &
            {r.get("scenario_group") for r in test_recs})
print(f"train {len(train_recs)} / val {len(val_recs)} / dev {len(dev_recs)} / test {len(test_recs)}")
'''

CELL_TRAIN = '''import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import (accuracy_score, cohen_kappa_score, roc_auc_score,
                             average_precision_score, precision_recall_fscore_support)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def metrics_from(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}

class DlgDataset(torch.utils.data.Dataset):
    def __init__(self, recs, label_key):
        self.enc = tok([build_text(r) for r in recs], truncation=True,
                       max_length=MAX_LENGTH, padding="max_length")
        self.labels = [int(r[label_key]) for r in recs]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i]); return item

ds_train = DlgDataset(train_recs, "is_scam")
ds_val   = DlgDataset(val_recs,   "is_scam")
ds_dev   = DlgDataset(dev_recs,   "gold_is_scam")
ds_test  = DlgDataset(test_recs,  "gold_is_scam")

def train_seed(seed):
    set_seed(seed)
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    targs = TrainingArguments(
        output_dir=str(OUT / f"ckpt_seed{seed}"),
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        seed=seed, data_seed=seed, fp16=USE_FP16,
        eval_strategy="epoch",
        # ← ЕДИНСТВЕННОЕ отличие от прогона «Три сида»: веса лучшего чекпоинта
        # обязаны пережить прогон
        save_strategy="epoch", load_best_model_at_end=True, save_total_limit=2,
        metric_for_best_model="eval_gold_dev_f1", greater_is_better=True,
        logging_steps=10, report_to="none", disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train,
                      eval_dataset={"silver_val": ds_val, "gold_dev": ds_dev,
                                    "gold_test": ds_test},
                      processing_class=tok,
                      compute_metrics=lambda ep: metrics_from(ep[1], np.argmax(ep[0], -1)),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
    trainer.train()
    hist = copy.deepcopy(trainer.state.log_history)

    per_epoch = []
    for h in hist:
        if "eval_gold_dev_f1" in h:
            per_epoch.append({"epoch": int(round(h["epoch"])),
                              "gold_dev_f1": float(h["eval_gold_dev_f1"])})
    best_epoch = max(per_epoch, key=lambda x: x["gold_dev_f1"])["epoch"]

    po = trainer.predict(ds_test)
    probs = torch.softmax(torch.tensor(po.predictions), dim=-1).numpy()[:, 1]
    y_pred = po.predictions.argmax(axis=-1)
    y_true = np.array([r["gold_is_scam"] for r in test_recs])
    m = metrics_from(y_true, y_pred)
    m["roc_auc"] = float(roc_auc_score(y_true, probs))
    m["average_precision"] = float(average_precision_score(y_true, probs))

    dest = MODELS / f"binary_xlmr_seed{seed}"
    trainer.save_model(str(dest)); tok.save_pretrained(str(dest))
    shutil.rmtree(OUT / f"ckpt_seed{seed}", ignore_errors=True)   # промежуточные не нужны

    ref = REFERENCE_V1[seed]
    print(f"\\nсид {seed}: лучшая эпоха {best_epoch} (эталон 1) | "
          f"gold-test v2 F1 {m['f1']:.4f} (эталон на v1 {ref:.4f}, "
          f"|Δ| {abs(m['f1']-ref):.4f}, допуск {TOLERANCE})")
    print(f"  веса: {dest} | {(time.time()-t0)/60:.1f} мин")
    return {"seed": seed, "best_epoch": best_epoch, "per_epoch": per_epoch,
            "test_metrics": m, "reference_v1": ref,
            "abs_diff_vs_reference": abs(m["f1"] - ref),
            "reproduced": bool(best_epoch == 1 and abs(m["f1"] - ref) <= TOLERANCE),
            "model_dir": str(dest), "elapsed_sec": time.time() - t0,
            "y_pred": y_pred.tolist(), "prob_scam": [float(x) for x in probs]}

RUNS = [train_seed(s) for s in SEEDS]
'''

CELL_CHECK = '''ok = [r for r in RUNS if r["reproduced"]]
print("=" * 70)
for r in RUNS:
    print(f"сид {r['seed']}: эпоха {r['best_epoch']}, F1 {r['test_metrics']['f1']:.4f}, "
          f"|Δ| {r['abs_diff_vs_reference']:.4f} → "
          f"{'ВОСПРОИЗВЕЛОСЬ' if r['reproduced'] else 'РАСХОЖДЕНИЕ'}")
import numpy as np
f1s = [r["test_metrics"]["f1"] for r in RUNS]
print(f"\\ngold-test v2: {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f} "
       f"(в CLAUDE.md на v2: 0.9344 ± 0.0097)")
if len(ok) < len(RUNS):
    print("\\n⚠ НЕ ВСЕ СИДЫ ВОСПРОИЗВЕЛИСЬ — префиксы считаются, но решение о том, "
          "годятся ли они для раздела, принимается вручную по этим числам.")
'''

CELL_PREFIX = '''def prefix_probs_for(model_dir, recs, batch=32):
    """P(scam) на каждом префиксе каждого диалога. Динамический паддинг."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_dir)
    md = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    jobs = [(r["id"], k, prefix_text(r, k))
            for r in recs for k in range(1, len(r["turns"]) + 1)]
    jobs.sort(key=lambda j: len(j[2]))
    out, t0 = {}, time.time()
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            ch = jobs[i:i + batch]
            enc = tk([c[2] for c in ch], truncation=True, max_length=MAX_LENGTH,
                     padding=True, return_tensors="pt").to("cuda")
            pr = torch.softmax(md(**enc).logits, dim=-1)[:, 1].float().cpu().numpy()
            for (did, k, _), p in zip(ch, pr):
                out.setdefault(did, {})[k] = float(p)
    del md; torch.cuda.empty_cache()
    print(f"  {len(jobs)} префиксов за {time.time()-t0:.0f} c")
    return {d: [v[k] for k in sorted(v)] for d, v in out.items()}

PREFIX = {}
for r in RUNS:
    print(f"префиксы, сид {r['seed']}:")
    PREFIX[str(r["seed"])] = prefix_probs_for(r["model_dir"], test_recs)
'''

CELL_SAVE = '''payload = {
    "task": "переобучение с сохранением весов + префиксные вероятности",
    "model": MODEL_NAME, "seeds": SEEDS,
    "hyperparams": {"lr": LR, "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO, "epochs": EPOCHS,
                    "patience": PATIENCE, "batch_size": BATCH_SIZE,
                    "max_length": MAX_LENGTH, "fp16": USE_FP16},
    "save_policy": "save_strategy=epoch, load_best_model_at_end=True, save_total_limit=2",
    "gold": "gold_test_v2.csv", "reference_v1": REFERENCE_V1, "tolerance": TOLERANCE,
    "gpu": gpu, "n_test": len(test_recs),
    "test_ids": [r["id"] for r in test_recs],
    "test_langs": [r["lang"] for r in test_recs],
    "runs": [{k: v for k, v in r.items() if k != "prob_scam"} for r in RUNS],
    "prob_scam": {str(r["seed"]): r["prob_scam"] for r in RUNS},
}
(OUT / "retrain_results.json").write_text(json.dumps(payload, ensure_ascii=False),
                                          encoding="utf-8")
(OUT / "prefix_probs_by_seed.json").write_text(json.dumps(PREFIX, ensure_ascii=False),
                                               encoding="utf-8")
print("сохранено:", OUT / "retrain_results.json", OUT / "prefix_probs_by_seed.json")

shutil.make_archive("/kaggle/working/retrain_small", "zip", OUT)
print("маленький архив (метрики + префиксы):", "/kaggle/working/retrain_small.zip")
for d in sorted(MODELS.iterdir()):
    print(" веса:", d, f"{sum(f.stat().st_size for f in d.rglob('*'))/1e9:.2f} ГБ")
'''


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notebooks/retrain_save_kaggle.ipynb")
    args = ap.parse_args()
    nb = {
        "cells": [
            cell("markdown", MD_INTRO),
            cell("markdown", "## Конфигурация и данные"),
            cell("code", CELL_CFG),
            cell("code", CELL_DATA),
            cell("markdown", "## Обучение трёх сидов с сохранением весов"),
            cell("code", CELL_TRAIN),
            cell("markdown", "## Проверка воспроизводимости"),
            cell("code", CELL_CHECK),
            cell("markdown", "## Префиксные вероятности на восстановленных моделях"),
            cell("code", CELL_PREFIX),
            cell("markdown", "## Сохранение"),
            cell("code", CELL_SAVE),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ноутбук: {p} ({len(nb['cells'])} ячеек)")


if __name__ == "__main__":
    main()
