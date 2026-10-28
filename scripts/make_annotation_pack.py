#!/usr/bin/env python3
"""Генератор разметочного пака: самодостаточный HTML + манифест.

Черновые метки корпуса в пак НЕ попадают (слепая разметка); диалоги
перемешаны с фиксированным seed, реальные id заменены на опакные pid
(реальные id содержат имя сценария и раскрыли бы ответ). Соответствие
pid → id — в манифесте <пак>.manifest.json (аннотатору не открывать).

Примеры:
  python scripts/make_annotation_pack.py --split test
  python scripts/make_annotation_pack.py --split train --n 150
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from scam_onset.data import annotation_pack as ap  # noqa: E402

DEFAULT_IN = "data/corpus/corpus_v1.jsonl"
DEFAULT_OUTDIR = "data/annotations/packs"
DEFAULT_CONFIG = "configs/default.yaml"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default=DEFAULT_IN)
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=None,
                   help="случайная подвыборка (seed фиксирован в конфиге)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    args = p.parse_args()

    cfg = {}
    cfg_path = pathlib.Path(args.config)
    if cfg_path.exists():
        cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get(
            "annotation", {})
    sample_seed = cfg.get("sample_seed", ap.DEFAULT_SAMPLE_SEED)
    shuffle_seed = cfg.get("shuffle_seed", ap.DEFAULT_SHUFFLE_SEED)

    with open(args.inp, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if "split" not in records[0]:
        sys.exit("в корпусе нет поля split — сначала scripts/make_splits.py")

    payload, manifest = ap.build_pack_payload(
        records, args.split, args.n, sample_seed, shuffle_seed)

    leaked = ap.check_blindness(payload)
    if leaked:
        sys.exit(f"❌ пейлоад раскрывает поля {leaked} — пак не записан")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"pack_{manifest['pack_id']}"
    html_path = outdir / f"{stem}.html"
    manifest_path = outdir / f"{stem}.manifest.json"
    html_path.write_text(ap.render_pack_html(payload), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"пак: {html_path} ({manifest['n']} диалогов, split={args.split}, "
          f"sample_seed={sample_seed}, shuffle_seed={shuffle_seed})")
    print(f"манифест: {manifest_path}  ← аннотатору НЕ открывать")
    print("слепота: черновые метки и реальные id в HTML отсутствуют ✅")


if __name__ == "__main__":
    main()
