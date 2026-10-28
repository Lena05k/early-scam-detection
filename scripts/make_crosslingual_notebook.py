#!/usr/bin/env python3
"""Генератор ноутбука `notebooks/crosslingual_kaggle.ipynb` (87 прогонов xlm-r).

Ноутбук пишется скриптом, а не руками: так его легко переигрывать и видно диффом,
что именно поменялось. Логику отбора конфигураций он НЕ повторяет — читает готовые
списки id из `train_manifest.json`, поэтому выборки побитово те же, что у слоёв 1 и 2.

  python scripts/make_crosslingual_notebook.py
"""
import argparse
import json
import pathlib

MD_INTRO = """# Кросс-язычный перенос: xlm-roberta-base, 87 прогонов

**Зачем.** Третий заявленный вклад работы (tech_report §1, RQ3) — инвариантность к
языку — до сих пор не измерен: есть только in-language разбивка внутри all-3.
Здесь считается полная матрица «обучали на X → тест на Y».

**Что нельзя нарушать.**

- **Целевой язык не участвует НИ В ЧЁМ при zero-shot.** Отбор эпохи идёт по gold-dev,
  отфильтрованному по языкам-источникам (`dev_sel`, 33–34 при одном источнике,
  66–67 при двух). Полный gold-dev в отборе не участвует.
- **Контроль размера.** Один язык даёт ~464 примера, три — 1392. Для leave-one-out и
  all-3 есть вариант, выровненный подвыборкой до 464, иначе эффект переноса
  смешается с эффектом объёма.
- **Тест всегда полный gold-test v2 (303)**, метрики отдельно по каждому языку.
- Сплиты не переразбиваются, gold-файлы только читаются.

**Как устроен отбор.** Чекпоинты НЕ сохраняются (`save_strategy="no"`): вместо этого
на каждой эпохе снимаются пер-диалоговые предсказания на `dev_sel`, `val_sel` и
`gold_test`. Выбор эпохи и все его варианты (по gold-dev источников, по silver-val
источников) считаются ПОСТФАКТУМ из снимков — без переобучения и без перезагрузки
весов. Ранняя остановка работает по `eval_dev_sel_f1`.

⚠ Отбор на 33 диалогах шумный: ДИ шире в 1.71×, а для en-источника выбор совпадает
с выбором полного dev лишь в 19.9% бутстрап-выборок (`scripts/crosslingual_dev_ci.py`).
Поэтому вариант «по silver-val источников» считается наравне и сравнивается.

**Датасет Kaggle:** `corpus_v1.jsonl`, `gold_test_v2.csv`, `gold_dev_split.json`,
`train_manifest.json`. **Ожидаемое время:** ~4 ч 15 мин на T4 x2.
"""

CELL_CFG = '''import os, sys, json, csv, glob, pathlib, time, copy
import torch

def find_one(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"не найден {name} под /kaggle/input — подключён ли датасет?"
    return hits[0]

CORPUS   = find_one("corpus_v1.jsonl")
GOLD     = find_one("gold_test_v2.csv")
GOLD_DEV = find_one("gold_dev_split.json")
MANIFEST = find_one("train_manifest.json")

OUT = pathlib.Path("/kaggle/working/crosslingual"); (OUT / "data").mkdir(parents=True, exist_ok=True)

MODEL_NAME    = "xlm-roberta-base"
SEEDS         = [20260721, 20260722, 20260723]
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.2
EPOCHS        = 5
PATIENCE      = 2
BATCH_SIZE    = 16
MAX_LENGTH    = 512
USE_FP16      = torch.cuda.is_available()
LANGS         = ["en", "ru", "zh"]

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет")
print("устройств:", torch.cuda.device_count())
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

def read_gold_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if did: out[did] = int(r["is_scam"])
    return out

records  = read_corpus(CORPUS)
gold     = read_gold_csv(GOLD)
manifest = json.load(open(MANIFEST, encoding="utf-8"))
by_id    = {r["id"]: r for r in records}

train_all = [r for r in records if r.get("split") == "train"]
val_all   = [r for r in records if r.get("split") == "val"]

all_test = []
for r in records:
    if r.get("split") != "test": continue
    r2 = {k: v for k, v in r.items() if k != "is_scam"}   # silver-метку выбрасываем
    if r2["id"] in gold:
        r2["gold_is_scam"] = gold[r2["id"]]
        all_test.append(r2)
assert len(all_test) == len(gold), f"джойн неполный: {len(all_test)} из {len(gold)}"

sm     = json.load(open(GOLD_DEV, encoding="utf-8"))
assign = sm["assignment"]
dev_recs  = [r for r in all_test if assign.get(r["id"]) == "dev"]
test_recs = [r for r in all_test if assign.get(r["id"]) != "dev"]
assert len(dev_recs) == sm["n_dev"] and len(test_recs) == sm["n_test"], "разбиение не совпало"
assert not ({r.get("scenario_group") for r in dev_recs} &
            {r.get("scenario_group") for r in test_recs}), "группы разорваны"

CONFIGS = manifest["configs"]
assert manifest["seeds"] == SEEDS, "сиды манифеста разошлись с ноутбуком"
print(f"конфигураций {len(CONFIGS)}, сидов {len(SEEDS)} → прогонов {len(CONFIGS)*len(SEEDS)}")
print(f"dev {len(dev_recs)} / test {len(test_recs)} / train-пул {len(train_all)} / val-пул {len(val_all)}")
'''

CELL_RUN = '''import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def metrics_from(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 pos_label=1, zero_division=0)
    k = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f), "cohen_kappa": float(k)}

class DlgDataset(torch.utils.data.Dataset):
    def __init__(self, recs, label_key):
        self.enc = tok([build_text(r) for r in recs], truncation=True,
                       max_length=MAX_LENGTH, padding="max_length")
        self.labels = [int(r[label_key]) for r in recs]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i]); return item

# токенизируем один раз то, что не зависит от конфигурации
DS_TEST = DlgDataset(test_recs, "gold_is_scam")
TEST_LANGS = [r["lang"] for r in test_recs]
_train_cache, _dev_cache, _val_cache = {}, {}, {}

def ds_train_for(cfg_name):
    if cfg_name not in _train_cache:
        ids = CONFIGS[cfg_name]["train_ids"]
        _train_cache[cfg_name] = DlgDataset([by_id[i] for i in ids], "is_scam")
    return _train_cache[cfg_name]

def sel_sets_for(sources):
    key = "+".join(sorted(sources))
    if key not in _dev_cache:
        d = [r for r in dev_recs if r["lang"] in sources]
        v = [r for r in val_all  if r["lang"] in sources]
        _dev_cache[key] = (d, DlgDataset(d, "gold_is_scam"))
        _val_cache[key] = (v, DlgDataset(v, "is_scam"))
    return _dev_cache[key], _val_cache[key]

def run_one(cfg_name, seed):
    """Один прогон. Возвращает пер-эпохные снимки на dev_sel / val_sel / gold_test."""
    cfg = CONFIGS[cfg_name]["config"]
    sources = cfg["sources"]
    (dev_sel, ds_dev), (val_sel, ds_val) = sel_sets_for(sources)
    ds_train = ds_train_for(cfg_name)

    # снимки опознаются по длине набора — размеры обязаны различаться
    sizes = [len(dev_sel), len(val_sel), len(test_recs)]
    assert len(set(sizes)) == 3, f"размеры eval-наборов совпали: {sizes}"

    set_seed(seed)
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2,
                                                               torch_dtype=torch.float32)
    targs = TrainingArguments(
        output_dir=str(OUT / "tmp"),
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        seed=seed, data_seed=seed, fp16=USE_FP16,
        eval_strategy="epoch",
        save_strategy="no",              # чекпоинты не нужны: отбор идёт по снимкам
        load_best_model_at_end=False,
        metric_for_best_model="eval_dev_sel_f1", greater_is_better=True,
        logging_steps=50, report_to="none", disable_tqdm=True,
    )

    snaps = {"dev_sel": [], "val_sel": [], "gold_test": []}
    def compute_metrics(ep):
        logits, labels = ep
        preds = np.argmax(logits, axis=-1)
        n = len(labels)
        name = ("dev_sel" if n == len(dev_sel) else
                "val_sel" if n == len(val_sel) else
                "gold_test" if n == len(test_recs) else None)
        if name:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            snaps[name].append({"y_true": np.asarray(labels).tolist(),
                                "y_pred": preds.tolist(),
                                "prob_scam": [float(x) for x in probs]})
        return metrics_from(labels, preds)

    # ⚠ В логе будет дважды за эпоху: "early stopping required metric_for_best_model,
    # but did not find eval_dev_sel_f1 so early stopping is disabled". Это ШУМ, а не
    # поломка: коллбэк вызывается после КАЖДОГО из трёх eval-наборов и на val_sel и
    # gold_test молча выходит, потому что там своих ключей нет. На dev_sel метрика
    # присутствует, и остановка работает — виден по epochs_run < EPOCHS.
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train,
                      eval_dataset={"dev_sel": ds_dev, "val_sel": ds_val,
                                    "gold_test": DS_TEST},
                      processing_class=tok, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
    trainer.train()
    hist = copy.deepcopy(trainer.state.log_history)
    n_ep = len(snaps["gold_test"])
    assert len(snaps["dev_sel"]) == len(snaps["val_sel"]) == n_ep, "снимки разошлись"

    del model, trainer
    torch.cuda.empty_cache()
    return {"config": cfg_name, "seed": seed, "epochs_run": n_ep,
            "elapsed_sec": time.time() - t0,
            "n_train": len(CONFIGS[cfg_name]["train_ids"]),
            "sizes": {"dev_sel": len(dev_sel), "val_sel": len(val_sel),
                      "gold_test": len(test_recs)},
            "dev_sel_langs": [r["lang"] for r in dev_sel],
            "val_sel_langs": [r["lang"] for r in val_sel],
            "snapshots": snaps,
            "log_history": [h for h in hist if "loss" in h or "eval_dev_sel_f1" in h]}
'''

CELL_LOOP = '''ALL, t_start = [], time.time()
names = list(CONFIGS)
done_path = OUT / "data" / "runs.jsonl"
done = set()
if done_path.exists():                      # докачка после обрыва сессии
    for line in done_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); done.add((r["config"], r["seed"])); ALL.append(r)
    print(f"из кэша: {len(done)} прогонов")

fh = open(done_path, "a", encoding="utf-8")
total = len(names) * len(SEEDS)
for ci, name in enumerate(names, 1):
    for seed in SEEDS:
        if (name, seed) in done:
            continue
        r = run_one(name, seed)
        fh.write(json.dumps(r, ensure_ascii=False) + "\\n"); fh.flush()
        ALL.append(r)
        el = (time.time() - t_start) / 60
        k = len(ALL)
        print(f"[{k:>2}/{total}] {name:<26} сид {seed} | эпох {r['epochs_run']} | "
              f"{r['elapsed_sec']/60:.1f} мин | всего {el:.0f} мин | "
              f"осталось ~{el/max(1,k-len(done))*(total-k):.0f} мин", flush=True)
fh.close()
print(f"\\nГОТОВО: {len(ALL)} прогонов за {(time.time()-t_start)/60:.0f} мин")
'''

CELL_SAVE = '''payload = {
    "task": "кросс-язычный перенос, xlm-roberta-base",
    "model": MODEL_NAME, "seeds": SEEDS,
    "hyperparams": {"lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                    "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                    "max_length": MAX_LENGTH, "fp16": USE_FP16},
    "selection": ("эпоха выбирается ПОСТФАКТУМ по снимкам: основной вариант — gold-dev "
                  "языков-источников (dev_sel), контроль — silver-val источников (val_sel). "
                  "Целевой язык в отборе не участвует."),
    "test_langs": TEST_LANGS,
    "test_ids": [r["id"] for r in test_recs],
    "n_runs": len(ALL),
    "runs": ALL,
}
out = OUT / "results_raw.json"
out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print("сохранено:", out, f"{out.stat().st_size/1e6:.1f} МБ")

import shutil
shutil.make_archive("/kaggle/working/crosslingual_artifacts", "zip", OUT)
print("архив: /kaggle/working/crosslingual_artifacts.zip")
'''


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notebooks/crosslingual_kaggle.ipynb")
    args = ap.parse_args()

    nb = {
        "cells": [
            cell("markdown", MD_INTRO),
            cell("markdown", "## Конфигурация и данные"),
            cell("code", CELL_CFG),
            cell("code", CELL_DATA),
            cell("markdown", "## Один прогон как функция"),
            cell("code", CELL_RUN),
            cell("markdown", "## 87 прогонов (докачиваемо: при обрыве перезапустить ячейку)"),
            cell("code", CELL_LOOP),
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
    print(f"ноутбук: {p} ({len(nb['cells'])} ячеек, {p.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
