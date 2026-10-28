import sys, pathlib
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.viz.plots import (
    plot_iaa_bars, plot_onset_matrix, plot_agreement_matrix, plot_before_after,
    pairwise_agreement,
)

SCORES = {"is_scam": 0.79, "phishing": 1.0, "money": 1.0, "other": 0.44, "onset": 0.33}
THRESHOLDS = {"is_scam": 0.80, "phishing": 0.75, "money": 0.75, "other": 0.75, "onset": 0.67}

ANN = {
    1: {"A": dict(is_scam=1, subs=(1, 0, 0), onset=1),
        "B": dict(is_scam=1, subs=(1, 0, 0), onset=3)},
    2: {"A": dict(is_scam=0, subs=(0, 0, 0), onset=None),
        "B": dict(is_scam=0, subs=(0, 0, 0), onset=None)},
    3: {"A": dict(is_scam=1, subs=(0, 1, 0), onset=2),
        "B": dict(is_scam=0, subs=(0, 0, 0), onset=None)},
}


def _check_png(path):
    p = pathlib.Path(path)
    assert p.exists()
    assert p.stat().st_size > 1024


def test_iaa_bars(tmp_path):
    _check_png(plot_iaa_bars(SCORES, THRESHOLDS, tmp_path / "bars.png"))


def test_iaa_bars_negative_alpha(tmp_path):
    # α может быть отрицательной (систематическое расхождение) — фигура не должна падать
    scores = dict(SCORES, onset=-0.15)
    _check_png(plot_iaa_bars(scores, THRESHOLDS, tmp_path / "bars_neg.png"))


def test_onset_matrix(tmp_path):
    onsets = {1: {"A": 1, "B": 3, "C": 1},
              2: {"A": 2, "B": 2, "C": None},   # пропуск аннотатора -> клетка ∅
              5: {"A": 5, "B": 4, "C": 5}}
    gold = {1: 1, 2: 2, 5: 5}
    _check_png(plot_onset_matrix(onsets, gold, tmp_path / "onset.png"))


def test_onset_matrix_skips_benign(tmp_path):
    # benign -> t*=∅: строка не рисуется, фигура строится по остальным
    onsets = {1: {"A": 1, "B": 1}, 3: {"A": None, "B": None}}
    gold = {1: 1, 3: None}
    _check_png(plot_onset_matrix(onsets, gold, tmp_path / "onset_benign.png"))


def test_onset_matrix_all_benign_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_onset_matrix({3: {"A": None}}, {3: None}, tmp_path / "empty.png")


def test_agreement_matrix(tmp_path):
    _check_png(plot_agreement_matrix(ANN, tmp_path / "agree.png"))


def test_pairwise_agreement_value():
    # 3 items x 5 решений (is_scam + 3 подтипа + onset): расходятся onset у Пр.1
    # и is_scam+money+onset у Пр.3 -> 11/15 совпадений
    assert pairwise_agreement(ANN, "A", "B") == pytest.approx(11 / 15)


def test_before_after(tmp_path):
    after = {k: min(1.0, v + 0.2) for k, v in SCORES.items()}
    _check_png(plot_before_after(SCORES, after, tmp_path / "ba.png",
                                 thresholds=THRESHOLDS))


def test_before_after_without_thresholds(tmp_path):
    _check_png(plot_before_after(SCORES, SCORES, tmp_path / "ba_nothr.png"))
