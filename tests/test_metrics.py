import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.metrics import (
    onset_kernel, oad_score, binary_prf, f_latency, erde, onset_distance,
)

# ----------------------------------------------------------------- onset_kernel
def test_kernel_zero_is_one():
    assert onset_kernel(0, 1.0, 2.0) == 1.0

def test_kernel_asymmetry_late_penalised_more():
    # λ_late < λ_early ⟹ одинаковое |Δ| позднее штрафуется сильнее ⟹ q меньше
    q_late = onset_kernel(2, 1.0, 2.0)     # опоздание
    q_early = onset_kernel(-2, 1.0, 2.0)   # раннее
    assert q_late < q_early
    assert 0.0 < q_late < 1.0 and 0.0 < q_early < 1.0

# -------------------------------------------------------------------------- OAD
def test_oad_perfect_run_is_one():
    y = [1, 1, 0, 0]; p = [1, 1, 0, 0]
    tt = [1, 3, None, None]; tp = [1, 3, None, None]
    assert abs(oad_score(y, p, tt, tp) - 1.0) < 1e-12

def test_oad_beta1_equals_balanced_accuracy():
    # 2 benign (1 верный, 1 FP), 2 scam (1 TP, 1 FN)
    y = [0, 0, 1, 1]; p = [0, 1, 1, 0]
    tt = [None, None, 2, 4]; tp = [None, None, 5, None]  # онсет TP «неважен» при β=1
    tnr = 1 / 2  # верных benign
    tpr = 1 / 2  # верных scam
    bal_acc = 0.5 * (tnr + tpr)
    assert abs(oad_score(y, p, tt, tp, beta=1.0) - bal_acc) < 1e-12

def test_oad_false_alarm_on_hard_negative_penalised():
    base_y = [0, 1]; base_p = [0, 1]; tt = [None, 1]; tp = [None, 1]
    clean = oad_score(base_y, base_p, tt, tp)
    # тот же прогон, но hard-negative benign ошибочно помечен scam
    fa = oad_score([0, 1], [1, 1], [None, 1], [None, 1])
    assert fa < clean
    assert abs(clean - 1.0) < 1e-12  # без ложной тревоги — идеал

def test_oad_late_onset_floors_at_beta():
    # scam детектирован, но онсет сильно поздно ⟹ s→β (не 0)
    s = oad_score([1], [1], [1], [999], beta=0.5, lam_late=1.0)
    assert abs(s - 0.5) < 1e-6

# ------------------------------------------------------------------- F_latency
def test_f_latency_le_f1():
    y = [1, 1, 0, 0]; p = [1, 1, 0, 0]
    tt = [1, 2, None, None]; tp = [3, 4, None, None]  # поздние онсеты
    _, _, f1 = binary_prf(y, p)
    fl = f_latency(y, p, tt, tp)
    assert fl <= f1 + 1e-12
    assert f1 > 0.0 and fl < f1  # есть штраф за задержку

# ------------------------------------------------------------------------ ERDE
def test_erde_fn_costs_cfn():
    # один диалог, пропуск скама ⟹ ERDE = c_fn
    assert abs(erde([1], [0], [1], [None], c_fn=1.0) - 1.0) < 1e-12

def test_erde_early_tp_is_low():
    # ранний верный TP ⟹ низкий ERDE (много меньше стоимости пропуска)
    early = erde([1], [1], [3], [1], o=3.0, c_tp=1.0, c_fn=1.0)  # k = 1−3+1 = −1
    fn = erde([1], [0], [3], [None], c_fn=1.0)
    assert early < fn
    assert early < 0.3

# --------------------------------------------------------------- onset_distance
def test_onset_distance_mae_and_tolerance():
    # deltas: 0, +1, −2  (все на верно-детектированных scam)
    y = [1, 1, 1]; p = [1, 1, 1]
    tt = [5, 5, 5]; tp = [5, 6, 3]
    r = onset_distance(y, p, tt, tp)
    assert r["n"] == 3
    assert abs(r["mae"] - 1.0) < 1e-12          # (0+1+2)/3
    assert abs(r["tolerance"][1] - 2 / 3) < 1e-12
    assert r["early"] == 1 and r["exact"] == 1 and r["late"] == 1

def test_onset_distance_only_on_true_positives():
    # FN и FP не участвуют в onset-distance (t̂* не определён)
    y = [1, 0]; p = [0, 1]; tt = [2, None]; tp = [None, 3]
    r = onset_distance(y, p, tt, tp)
    assert r["n"] == 0
