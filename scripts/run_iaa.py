#!/usr/bin/env python3
"""Тонкая обёртка: запуск IAA-харнесса из пакета. Пути можно брать из configs/default.yaml."""
import sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.iaa.harness import run

def main():
    ap = argparse.ArgumentParser(description="Запуск IAA-калибровки")
    ap.add_argument("--gold", default="data/annotations/calibration_items_gold.csv")
    ap.add_argument("--annotations", default="data/annotations/annotator_demo.csv")
    ap.add_argument("--save-json", default=None, help="куда сохранить результаты (JSON)")
    a = ap.parse_args()
    run(a.gold, a.annotations, save_json=a.save_json)

if __name__ == "__main__":
    main()
