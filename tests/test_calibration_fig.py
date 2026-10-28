import json
import sys, pathlib
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.viz.calibration_progress import plot_calibration_progress

THRESHOLDS = {"kappa_is_scam": 0.80, "onset_alpha": 0.67}

ROUNDS = [
    {"is_scam_kappa": 0.79, "subtype_macro": 0.70, "sub_other": 0.44,
     "onset_alpha": 0.33, "onset_agree_pm1": 0.60, "onset_mae": 1.5,
     "n_coscam": 14, "n_items": 24, "n_annotators": 3},
    {"is_scam_kappa": 0.82, "subtype_macro": 0.76, "sub_other": 0.50,
     "onset_alpha": 0.55, "onset_agree_pm1": 0.75, "onset_mae": 1.0,
     "n_coscam": 15, "n_items": 24, "n_annotators": 3},
    {"is_scam_kappa": 0.857, "subtype_macro": 0.794, "sub_other": 0.441,
     "onset_alpha": 0.708, "onset_agree_pm1": 0.875, "onset_mae": 0.54,
     "n_coscam": 16, "n_items": 24, "n_annotators": 3},
]


def _write_rounds(tmp_path, rounds):
    pairs = []
    for i, d in enumerate(rounds, 1):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(d))
        pairs.append((f"раунд {i}", str(p)))
    return pairs


def _check_png(path):
    p = pathlib.Path(path)
    assert p.exists()
    assert p.stat().st_size > 1024


def test_progress_three_rounds(tmp_path):
    pairs = _write_rounds(tmp_path, ROUNDS)
    _check_png(plot_calibration_progress(pairs, tmp_path / "prog.png", THRESHOLDS))


def test_progress_two_rounds(tmp_path):
    pairs = _write_rounds(tmp_path, ROUNDS[:2])
    _check_png(plot_calibration_progress(pairs, tmp_path / "prog2.png", THRESHOLDS))


def test_progress_null_alpha(tmp_path):
    # харнесс пишет null вместо NaN (напр. нет со-скам items) — точка пропускается, фигура строится
    rounds = [dict(ROUNDS[0], onset_alpha=None), ROUNDS[2]]
    pairs = _write_rounds(tmp_path, rounds)
    _check_png(plot_calibration_progress(pairs, tmp_path / "prog_null.png", THRESHOLDS))


def test_progress_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_calibration_progress([], tmp_path / "empty.png", THRESHOLDS)
