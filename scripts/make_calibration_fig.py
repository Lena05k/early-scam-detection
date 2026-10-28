#!/usr/bin/env python3
"""Фигура прогресса калибровки из JSON-результатов харнесса (run_iaa.py --save-json).

Пример:
    python scripts/make_calibration_fig.py \
        --round "раунд 1" reports/iaa_r1.json \
        --round "раунд 2" reports/iaa_r2.json \
        --round "раунд 3" reports/iaa_r3.json
"""
import sys, argparse, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    from scam_onset.viz.calibration_progress import plot_calibration_progress
except ImportError:
    sys.exit("Нужен matplotlib: pip install -e '.[viz]'")


def main():
    ap = argparse.ArgumentParser(description="Фигура прогресса калибровки по раундам")
    ap.add_argument("--round", nargs=2, action="append", metavar=("LABEL", "JSON"),
                    required=True, help="метка раунда и путь к JSON (повторяемый)")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--out", default="reports/figures/calibration_progress.png")
    a = ap.parse_args()
    ft = yaml.safe_load(pathlib.Path(a.config).read_text())["calibration"]["freeze_thresholds"]
    thresholds = {"kappa_is_scam": ft["kappa_is_scam"], "onset_alpha": ft["onset_alpha"]}
    print(plot_calibration_progress([tuple(r) for r in a.round], a.out, thresholds))


if __name__ == "__main__":
    main()
