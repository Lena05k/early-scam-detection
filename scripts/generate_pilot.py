#!/usr/bin/env python3
"""Пилотная генерация корпуса (Путь A, стадии 0-1): ~50 диалогов ru/en/zh с черновыми метками.

Пример:
  ANTHROPIC_API_KEY=... python scripts/generate_pilot.py --n 50 --langs ru,en,zh --preview 3

Выход: data/corpus/pilot_v1.jsonl (+ pilot_v1_rejected.jsonl) и отчёт качества.
Метки — ЧЕРНОВИК (gold_src=llm_draft): дальше человеческая верификация по гайдлайну v1.2.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.data.generate import (  # noqa: E402
    AnthropicClient, GenerationError, build_skeletons, load_config,
    render_dialogue, run_pipeline,
)


def main():
    ap = argparse.ArgumentParser(description="Пилотная генерация корпуса (стадии 0-1)")
    ap.add_argument("--n", type=int, default=None, help="число диалогов (default: generation.pilot_n)")
    ap.add_argument("--langs", default=None, help="языки через запятую (default: languages из конфига)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="каталог вывода (default: paths.corpus)")
    ap.add_argument("--seed", type=int, default=None, help="default: seeds[0] из конфига")
    ap.add_argument("--preview", type=int, default=0, help="показать k диалогов в рендере для человека")
    a = ap.parse_args()

    cfg = load_config(a.config)
    gen = cfg["generation"]
    n = a.n or gen.get("pilot_n", 50)
    langs = a.langs.split(",") if a.langs else cfg["languages"]
    seed = a.seed if a.seed is not None else cfg["seeds"][0]

    try:
        client = AnthropicClient.from_env(gen.get("api"))
    except GenerationError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        sys.exit(1)

    skeletons = build_skeletons(n, langs, gen["turns"]["min"], gen["turns"]["max"], seed=seed)
    print(f"скелетов: {len(skeletons)} | языки: {langs} | генераторы: {gen['models']}")
    if len(gen["models"]) < 2:
        print("⚠ один генератор — допустимо для ПИЛОТА; полный корпус требует >=2 (методичка §9.4)")

    accepted, rejected, rep = run_pipeline(skeletons, cfg, client, out_dir=a.out, log=print)

    print("\n=== отчёт качества ===")
    print(f"принято: {rep['n_accepted']} | отклонено валидацией: {rep['n_rejected']}")
    print(f"scam: {rep['scam']} | benign: {rep['benign']} (доля benign: {rep['benign_share']}) "
          f"| hard-negatives: {rep['hard_negatives']}")
    print(f"средняя длина: {rep['mean_turns']} реплик")
    print("по языкам:  " + ", ".join(f"{k}={v}" for k, v in sorted(rep["by_lang"].items())))
    print("по сценариям:")
    for k, v in sorted(rep["by_scenario"].items()):
        print(f"  {k:24s} {v}")
    if rep["reject_reasons"]:
        print("причины брака:")
        for k, v in sorted(rep["reject_reasons"].items()):
            print(f"  {v}× {k}")
    out = pathlib.Path(a.out or cfg["paths"]["corpus"])
    print(f"\nфайлы: {out / 'pilot_v1.jsonl'}  |  {out / 'pilot_v1_rejected.jsonl'}")
    print("метки — ЧЕРНОВИК (gold_src=llm_draft); далее человеческая верификация по v1.2.")

    for rec in accepted[: a.preview]:
        print("\n" + render_dialogue(rec))


if __name__ == "__main__":
    main()
