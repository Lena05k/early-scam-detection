#!/usr/bin/env python3
"""Распределение длин диалогов в токенах xlm-roberta-base + выбор max_length.

Считает по ВСЕМ 2000 диалогам корпуса (текст собирается ровно тем же
build_text, что и в обучении). Правило выбора: доля примеров длиннее 512
токенов > 25% → max_length=1024, batch_size=8; иначе 512 / 16.

  python scripts/token_length_stats.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_binary import build_text, read_corpus  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--out-plot", default="runs/binary_xlmr/plots/token_lengths.png")
    p.add_argument("--out-json", default="runs/binary_xlmr/token_lengths.json")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    records = read_corpus(args.corpus)
    texts = [build_text(r) for r in records]
    # add_special_tokens=True — считаем ровно то, что уедет в модель
    lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])

    frac_over_512 = float((lens > 512).mean())
    if frac_over_512 > 0.25:
        max_length, batch_size = 1024, 8
    else:
        max_length, batch_size = 512, 16

    stats = {
        "n": int(lens.size),
        "median": float(np.median(lens)),
        "p90": float(np.percentile(lens, 90)),
        "p95": float(np.percentile(lens, 95)),
        "max": int(lens.max()),
        "min": int(lens.min()),
        "mean": float(lens.mean()),
        "frac_over_512": frac_over_512,
        "frac_over_1024": float((lens > 1024).mean()),
        "chosen_max_length": max_length,
        "chosen_batch_size": batch_size,
    }

    print(f"диалогов: {stats['n']}  (токенизатор {args.model})")
    print(f"  медиана        {stats['median']:.0f}")
    print(f"  90-й перцентиль {stats['p90']:.0f}")
    print(f"  максимум       {stats['max']}")
    print(f"  доля > 512 токенов: {frac_over_512:.2%}")
    print(f"\nправило: доля>512 {'>' if frac_over_512 > 0.25 else '≤'} 25% → "
          f"max_length={max_length}, batch_size={batch_size}")
    if max_length == 512:
        print(f"  → усечётся {int((lens > 512).sum())} из {stats['n']} диалогов "
              f"({frac_over_512:.2%}); 90-й перцентиль {stats['p90']:.0f} укладывается в окно")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outp = pathlib.Path(args.out_plot)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lens, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(max_length, color="#C44E52", linestyle="--", linewidth=2,
               label=f"выбранный max_length = {max_length}")
    ax.axvline(stats["median"], color="#55A868", linestyle=":", linewidth=1.6,
               label=f"медиана = {stats['median']:.0f}")
    ax.axvline(stats["p90"], color="#8172B2", linestyle=":", linewidth=1.6,
               label=f"90-й перцентиль = {stats['p90']:.0f}")
    ax.set_xlabel("длина диалога, токенов xlm-roberta-base")
    ax.set_ylabel("число диалогов")
    ax.set_title(f"Распределение длин диалогов корпуса (n={stats['n']}), "
                 f"доля длиннее 512 токенов — {frac_over_512:.1%}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outp, dpi=150)
    plt.close(fig)

    jp = pathlib.Path(args.out_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nгистограмма: {outp}\nстатистика:  {jp}")


if __name__ == "__main__":
    main()
