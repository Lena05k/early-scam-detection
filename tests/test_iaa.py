import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.iaa.harness import (
    cohen_kappa, krippendorff_alpha_interval, median_lower,
)

def test_cohen_perfect():
    assert cohen_kappa([1,1,0,0],[1,1,0,0]) == 1.0

def test_cohen_zero_when_chance():
    # оба всегда "1" -> pe=1 -> по соглашению 1.0
    assert cohen_kappa([1,1,1],[1,1,1]) == 1.0

def test_alpha_perfect_agreement():
    assert abs(krippendorff_alpha_interval([[1,1],[3,3],[5,5]]) - 1.0) < 1e-9

def test_alpha_detects_systematic_disagreement():
    # систематический сдвиг -> alpha заметно ниже 1 (может быть <0)
    a = krippendorff_alpha_interval([[1,4],[3,1],[1,3]])
    assert a < 0.5

def test_median_lower_even_picks_lower_central():
    assert median_lower([1,3]) == 1
    assert median_lower([1,2,3,4]) == 2

def test_median_lower_odd():
    assert median_lower([1,2,3]) == 2
