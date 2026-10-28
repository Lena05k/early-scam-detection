#!/usr/bin/env python3
"""Бинарный классификатор is_scam на xlm-roberta-base.

Обучение на train (silver-метки LLM), валидация на val (silver), тест — ТОЛЬКО
по ручной золотой разметке: исходная колонка is_scam из test-сплита
выбрасывается, метки джойнятся из gold-CSV по dialog_id.

Сплиты берутся из поля `split` корпуса и НЕ переразбиваются.

  python scripts/train_binary.py                      # полный прогон
  python scripts/train_binary.py --smoke              # дымовой: 64/32/32, 1 эпоха, len 128

Ожидаемый потолок качества ~0.86–0.90: согласие silver-меток с gold по
is_scam = 0.859, обучающий сигнал зашумлён относительно тестовой правды.
"""
import argparse
import os
import json
import pathlib
import sys

import numpy as np

SEED = 20260721
MODEL_NAME = "xlm-roberta-base"


# ─────────────────────────── данные (чистые функции) ───────────────────────────

def read_corpus(path):
    """corpus jsonl → список записей."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_text(rec, anonymize_speakers=False):
    """Реплики → текст: одна реплика на строку, формат "[индекс|говорящий] текст".

    anonymize_speakers=True → вместо имени подставляется роль (`atk`/`tgt`).
    Нужно для проверки утечки: 92.1% имён atk-говорящих встречаются только в
    scam либо только в benign, то есть имя почти эквивалентно метке. Сама роль
    меткой не является — `atk` есть во всех 2000 диалогов, и scam, и benign.
    """
    key = "role" if anonymize_speakers else "speaker"
    return "\n".join(
        f"[{t['i']}|{t[key]}] {t['text']}" for t in rec["turns"]
    )


def read_gold_csv(path):
    """Золотая разметка → {dialog_id: is_scam}. Файл только читается."""
    import csv
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            did = (r.get("dialog_id") or "").strip()
            if did:
                out[did] = int(r["is_scam"])
    return out


def make_test_set(records, gold, verbose=True):
    """test-сплит + gold-метки по dialog_id. Исходный is_scam ВЫБРАСЫВАЕТСЯ.

    Возвращает (список записей с полем gold_is_scam, отчёт о джойне).
    """
    test = [r for r in records if r.get("split") == "test"]
    stripped = []
    for r in test:
        r2 = {k: v for k, v in r.items() if k != "is_scam"}  # silver-метку выбрасываем
        stripped.append(r2)

    matched, unmatched = [], []
    for r in stripped:
        if r["id"] in gold:
            r["gold_is_scam"] = gold[r["id"]]
            matched.append(r)
        else:
            unmatched.append(r["id"])
    gold_only = sorted(set(gold) - {r["id"] for r in stripped})

    report = {
        "n_test_split": len(test),
        "n_gold_rows": len(gold),
        "n_matched": len(matched),
        "unmatched_test_ids": unmatched,
        "gold_ids_not_in_test": gold_only,
    }
    if verbose:
        print(f"джойн gold ↔ test-сплит: смэтчилось {len(matched)} из {len(gold)} "
              f"строк gold (test-сплит: {len(test)})")
    return matched, report


# ─────────────────────────── метрики ───────────────────────────

def compute_metrics_from(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


# ─────────────────────────── обучение ───────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--gold", default="data/gold/gold_test_v1.csv")
    p.add_argument("--out-dir", default="runs/binary_xlmr")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--max-length", type=int, default=None,
                   help="по умолчанию берётся из runs/binary_xlmr/token_lengths.json")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.2)
    p.add_argument("--tb-root", default="runs/tb",
                   help="корень логов TensorBoard; прогон ляжет в <корень>/<модель>_seed<..>_e<..>")
    p.add_argument("--gold-dev", default="data/gold/gold_dev_split.json",
                   help="маппинг gold-dev/gold-test для отбора чекпоинта; "
                        "пустая строка — отключить и отбирать по silver-val")
    p.add_argument("--anonymize-speakers", action="store_true",
                   help="подставлять роль atk/tgt вместо имени говорящего "
                        "(проверка утечки через имена)")
    p.add_argument("--smoke", action="store_true",
                   help="дымовой прогон: 64 train / 32 val / 32 test, 1 эпоха, max_length=128")
    args = p.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              EarlyStoppingCallback, Trainer, TrainingArguments,
                              set_seed)

    set_seed(SEED)

    if torch.cuda.is_available():
        device_name, use_fp16 = "cuda", True
    elif torch.backends.mps.is_available():
        device_name, use_fp16 = "mps", False
    else:
        device_name, use_fp16 = "cpu", False

    # ── гиперпараметры длины ──
    if args.smoke:
        max_length, batch_size, epochs = 128, 16, 1
    else:
        max_length, batch_size = args.max_length, args.batch_size
        if max_length is None or batch_size is None:
            tl = pathlib.Path(args.out_dir) / "token_lengths.json"
            if not tl.exists():
                sys.exit(f"❌ нет {tl} — сначала запусти scripts/token_length_stats.py "
                         f"(или задай --max-length/--batch-size явно)")
            st = json.loads(tl.read_text(encoding="utf-8"))
            max_length = max_length or st["chosen_max_length"]
            batch_size = batch_size or st["chosen_batch_size"]
        epochs = args.epochs

    print(f"устройство: {device_name}  fp16={use_fp16}  "
          f"max_length={max_length}  batch_size={batch_size}  epochs={epochs}  seed={SEED}")

    # ── данные ──
    records = read_corpus(args.corpus)
    gold = read_gold_csv(args.gold)

    train_recs = [r for r in records if r.get("split") == "train"]
    val_recs = [r for r in records if r.get("split") == "val"]
    test_recs, join_report = make_test_set(records, gold)

    if join_report["n_matched"] < join_report["n_gold_rows"]:
        print(f"\n❌ смэтчилось {join_report['n_matched']} < {join_report['n_gold_rows']} строк gold.")
        if join_report["gold_ids_not_in_test"]:
            print("  id из gold, которых нет в test-сплите (до 10):")
            for i in join_report["gold_ids_not_in_test"][:10]:
                print(f"    {i}")
        if join_report["unmatched_test_ids"]:
            print("  id из test-сплита без gold-метки (до 10):")
            for i in join_report["unmatched_test_ids"][:10]:
                print(f"    {i}")
        sys.exit(1)

    # gold-dev: часть золотого теста уходит на ОТБОР чекпоинта (человеческие метки),
    # остальное — финальная оценка. Отбор по silver-val бессмыслен: recall там
    # насыщен в 1.0 с первой эпохи, разница между эпохами — шум.
    dev_recs = []
    gd_path = pathlib.Path(args.gold_dev) if args.gold_dev else None
    if gd_path and gd_path.exists():
        assign = json.loads(gd_path.read_text(encoding="utf-8"))["assignment"]
        dev_recs = [r for r in test_recs if assign.get(r["id"]) == "dev"]
        test_recs = [r for r in test_recs if assign.get(r["id"]) != "dev"]
        print(f"gold-dev включён ({gd_path}): отбор чекпоинта по {len(dev_recs)} "
              f"золотым диалогам, финальный тест — {len(test_recs)}")
    else:
        print("gold-dev ВЫКЛЮЧЕН — отбор чекпоинта пойдёт по silver-val "
              "(recall там насыщен, отбор фактически случаен)")

    if args.smoke:
        train_recs, val_recs = train_recs[:64], val_recs[:32]
        dev_recs, test_recs = dev_recs[:16], test_recs[:32]

    print(f"train {len(train_recs)} (silver) / val {len(val_recs)} (silver) / "
          f"gold-dev {len(dev_recs)} (gold) / test {len(test_recs)} (gold)")

    tok = AutoTokenizer.from_pretrained(args.model)

    class DlgDataset(torch.utils.data.Dataset):
        def __init__(self, recs, label_key):
            self.enc = tok([build_text(r, args.anonymize_speakers) for r in recs],
                           truncation=True,
                           max_length=max_length, padding="max_length")
            self.labels = [int(r[label_key]) for r in recs]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    ds_train = DlgDataset(train_recs, "is_scam")
    ds_val = DlgDataset(val_recs, "is_scam")
    ds_test = DlgDataset(test_recs, "gold_is_scam")
    ds_dev = DlgDataset(dev_recs, "gold_is_scam") if dev_recs else None

    # Два eval-набора сразу: кривая по silver-val остаётся видимой, но отбор
    # чекпоинта идёт по gold-dev. Метрики получают префикс имени набора.
    if ds_dev is not None:
        eval_sets = {"silver_val": ds_val, "gold_dev": ds_dev}
        best_metric = "eval_gold_dev_f1"
    else:
        eval_sets = ds_val
        best_metric = "f1"

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics_from(labels, preds)

    # Каждый прогон — в свою именованную подпапку, чтобы `tensorboard --logdir runs/tb`
    # показывал все прогоны одним сравнительным видом с осмысленной легендой.
    run_name = f"{args.model.split('/')[-1]}_seed{SEED}_e{epochs}"
    if args.anonymize_speakers:
        run_name += "_noNames"
    if args.smoke:
        run_name += "_smoke"
    tb_dir = pathlib.Path(args.tb_root) / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    # transformers >= 5.1 ИГНОРИРУЕТ logging_dir и пишет в <output_dir>/runs/<дата>_<хост>.
    # Переменную окружения понимает новый код, logging_dir — старый (5.0.0 на Kaggle);
    # ставим оба, иначе кривые Trainer'а уедут мимо именованной папки.
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        weight_decay=0.01,
        # 0.2, а не 0.1: сырой grad_norm доходил до 191 на шагах 50–110 при
        # max_grad_norm=1.0, то есть направление шага там срезалось почти вслепую.
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=SEED,
        data_seed=SEED,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        report_to="tensorboard",
        logging_dir=str(tb_dir),
    )
    print(f"TensorBoard: {tb_dir}  | отбор чекпоинта по: {best_metric}  | "
          f"warmup_ratio={args.warmup_ratio}")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=eval_sets,
        processing_class=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── история ──
    history = trainer.state.log_history
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\nлучший чекпоинт: {best_ckpt}  ({best_metric}={trainer.state.best_metric})")

    # ── тест на gold ──
    pred_out = trainer.predict(ds_test)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = logits.argmax(axis=-1)
    y_true = np.array([r["gold_is_scam"] for r in test_recs])

    from sklearn.metrics import classification_report, confusion_matrix
    m = compute_metrics_from(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n" + "=" * 60)
    print(f"ТЕСТ на ручной золотой разметке (n={len(y_true)})")
    print("=" * 60)
    for k in ("accuracy", "precision", "recall", "f1", "cohen_kappa"):
        print(f"  {k:12s} {m[k]:.4f}")
    print("\nматрица ошибок (строки — gold, столбцы — предсказание):")
    print("            pred=0  pred=1")
    print(f"  gold=0    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  gold=1    {cm[1,0]:6d}  {cm[1,1]:6d}")
    print("\n" + classification_report(y_true, y_pred, labels=[0, 1],
                                       target_names=["benign", "scam"], digits=4,
                                       zero_division=0))

    # ── разбивка по языкам ──
    per_lang = {}
    langs = sorted({r.get("lang") for r in test_recs if r.get("lang")})
    if langs:
        print("качество по языкам (поле lang):")
        for lg in langs:
            idx = [i for i, r in enumerate(test_recs) if r.get("lang") == lg]
            mm = compute_metrics_from(y_true[idx], y_pred[idx])
            mm["n"] = len(idx)
            per_lang[lg] = mm
            print(f"  {lg}: n={mm['n']:3d}  acc={mm['accuracy']:.4f}  f1={mm['f1']:.4f}  "
                  f"κ={mm['cohen_kappa']:.4f}")
    else:
        print("⚠ поля языка в данных нет — разбивка по языкам не считается")

    # ── артефакты ──
    import csv as _csv
    pred_path = out_dir / "test_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dialog_id", "lang", "gold_is_scam", "pred_is_scam", "prob_scam"])
        for r, yt, yp, pr in zip(test_recs, y_true, y_pred, probs):
            w.writerow([r["id"], r.get("lang", ""), int(yt), int(yp), f"{pr:.6f}"])

    results = {
        "device": device_name, "model": args.model, "seed": SEED,
        "max_length": max_length, "batch_size": batch_size,
        "epochs_requested": epochs,
        "best_model_checkpoint": best_ckpt, "best_metric_value": trainer.state.best_metric,
        "selection_metric": best_metric,
        "n_gold_dev": len(dev_recs), "n_gold_test": len(test_recs),
        "warmup_ratio": args.warmup_ratio, "tb_dir": str(tb_dir),
        "anonymize_speakers": args.anonymize_speakers,
        "join_report": {k: v for k, v in join_report.items()
                        if k in ("n_test_split", "n_gold_rows", "n_matched")},
        "test_metrics": m,
        "confusion_matrix_gold_x_pred": cm.tolist(),
        "per_language": per_lang,
        "smoke": args.smoke,
    }
    (out_dir / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── итог по gold в TensorBoard ──
    # Без этого в дашборде только eval/* — то есть val на silver-метках, ровно та
    # выборка, которая ничего не различает. Главные числа проекта пишем туда же.
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(tb_dir))
    for k, v in m.items():
        w.add_scalar(f"gold_test/{k}", v, 0)
    for k, v in (pred_out.metrics or {}).items():
        if isinstance(v, (int, float)):
            w.add_scalar(f"gold_test/{k.replace('test_', '')}", v, 0)
    for lg, mm in per_lang.items():
        w.add_scalar(f"gold_test_f1/{lg}", mm["f1"], 0)
        w.add_scalar(f"gold_test_recall/{lg}", mm["recall"], 0)
    w.add_scalar("gold_test/n", len(y_true), 0)
    w.flush()
    w.close()
    print(f"итог по gold записан в TensorBoard: {tb_dir}")

    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))

    print(f"\nартефакты:\n  {out_dir/'history.json'}\n  {pred_path}\n  "
          f"{out_dir/'test_results.json'}\n  {out_dir/'best'}")


if __name__ == "__main__":
    main()
