#!/usr/bin/env python3
"""Перенос history.json прогона в события TensorBoard.

Нужен для прогонов, сделанных до включения report_to="tensorboard" (например
на Kaggle): скаляры берутся из уже записанной истории, ничего не досчитывается
и не сглаживается. Ось X — шаг обучения, как в самом Trainer.

  python scripts/history_to_tb.py --run-dir runs/binary_xlmr
"""
import argparse
import json
import pathlib
import sys

# что писать: ключ в history.json → тег в TensorBoard
TRAIN_KEYS = {
    "loss": "train/loss",
    "learning_rate": "train/learning_rate",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
}
EVAL_KEYS = {
    "eval_loss": "eval/loss",
    "eval_accuracy": "eval/accuracy",
    "eval_precision": "eval/precision",
    "eval_recall": "eval/recall",
    "eval_f1": "eval/f1",
    "eval_cohen_kappa": "eval/cohen_kappa",
    "eval_runtime": "eval/runtime",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/binary_xlmr")
    p.add_argument("--tb-dir", default=None, help="по умолчанию <run-dir>/tb")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        sys.exit(f"❌ нет {hist_path}")
    hist = json.loads(hist_path.read_text(encoding="utf-8"))

    tb_dir = pathlib.Path(args.tb_dir) if args.tb_dir else run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(tb_dir))

    n = 0
    for rec in hist:
        step = rec.get("step")
        if step is None:
            continue
        keys = EVAL_KEYS if "eval_loss" in rec else TRAIN_KEYS
        for k, tag in keys.items():
            if k in rec and isinstance(rec[k], (int, float)):
                w.add_scalar(tag, rec[k], step)
                n += 1
    w.flush()
    w.close()

    ev = sum(1 for r in hist if "eval_loss" in r)
    tr = sum(1 for r in hist if "loss" in r and "eval_loss" not in r)
    print(f"записано {n} скаляров в {tb_dir}")
    print(f"  train-точек: {tr}, эпох валидации: {ev}")
    print(f"\n  .venv/bin/tensorboard --logdir {tb_dir} --port 6006")


if __name__ == "__main__":
    main()
