#!/usr/bin/env python3
"""Earliness- и onset-distance-метрики — методичка §5.2, §5.3.

Обе именованные метрики (F_latency, ERDE) в оригинале меряют задержку ОТ НАЧАЛА
ПОТОКА; у нас риск текстово появляется только на t*, поэтому k считаем ОТ ОНСЕТА:

    k = t̂* − t* + 1        (§5.2, [адапт., новое соглашение])

Онсет-метрики определены только на верно-детектированных scam (y=1 ∧ ŷ=1) —
иначе t̂* не определён (инвариант CLAUDE.md).
"""
from math import exp
from statistics import median

from ._params import section


def binary_prf(y_true, y_pred):
    """Precision/Recall/F1 по классу scam (§5.1)."""
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def _k_from_onset(t_true, t_pred, clamp=True):
    """k = t̂* − t* + 1 (от онсета). None ⟹ +∞ (максимальный штраф).

    clamp=True: k ≥ 1 — «быстрее онсета» не бывает; сохраняет свойство F_latency ≤ F1
    для раннего t̂*<t*. Ранний онсет как таковой штрафуется в OAD (λ_early), не здесь.
    """
    if t_true is None or t_pred is None:
        return float("inf")
    k = t_pred - t_true + 1
    return max(1, k) if clamp else k


def f_latency(y_true, y_pred, t_true, t_pred, p=None):
    """Latency-weighted F1 (Sadeque et al. 2018), §5.2.

    penalty(k) = −1 + 2/(1+exp(−p·(k−1)));  speed = 1 − median_{TP} penalty(k_u);
    F_latency = F1 · speed.  k — от онсета (см. _k_from_onset).

    ⚠ Расхождение с методичкой: формула допускает k<1 при раннем t̂* (penalty<0 ⟹
    speed>1 ⟹ F_latency>F1), что противоречит заявленному свойству F_latency ≤ F1.
    Разрешаем клампом k≥1 (ранний онсет — забота OAD/λ_early). Помечено [адапт.].
    """
    if p is None:
        p = section("earliness")["p"]
    _, _, f1 = binary_prf(y_true, y_pred)
    pens = []
    for y, yh, t, th in zip(y_true, y_pred, t_true, t_pred):
        if y == 1 and yh == 1:                       # TP
            k = _k_from_onset(t, th)
            if k == float("inf"):
                pens.append(1.0)
            else:
                pens.append(-1.0 + 2.0 / (1.0 + exp(-p * (k - 1))))
    speed = 1.0 - median(pens) if pens else 1.0
    return f1 * speed


def erde(y_true, y_pred, t_true, t_pred, o=None, c_fp=None, c_fn=None, c_tp=None):
    """ERDE_o (Losada & Crestani 2016), §5.2 — МЕНЬШЕ лучше.

    lc_o(k) = 1 − 1/(1+e^{k−o});  cost = c_fp | c_fn | lc_o(k)·c_tp | 0 (FP|FN|TP|TN).
    Возвращает среднее по диалогам. k — от онсета.
    """
    e = section("earliness")
    o = e["o"] if o is None else o
    c_fp = e["c_fp"] if c_fp is None else c_fp
    c_fn = e["c_fn"] if c_fn is None else c_fn
    c_tp = e["c_tp"] if c_tp is None else c_tp

    costs = []
    for y, yh, t, th in zip(y_true, y_pred, t_true, t_pred):
        if y == 0 and yh == 0:
            costs.append(0.0)                        # TN
        elif y == 0 and yh == 1:
            costs.append(c_fp)                       # FP
        elif y == 1 and yh == 0:
            costs.append(c_fn)                       # FN
        else:                                        # TP
            k = _k_from_onset(t, th, clamp=False)
            lc = 1.0 - 1.0 / (1.0 + exp(k - o)) if k != float("inf") else 1.0
            costs.append(lc * c_tp)
    return sum(costs) / len(costs) if costs else 0.0


def onset_distance(y_true, y_pred, t_true, t_pred, tolerances=(1, 2)):
    """Onset-distance на верно-детектированных scam (§5.3).

    Δ_i = t̂*_i − t*_i (знаковое); MAE = mean|Δ|; tol@±k = доля |Δ|≤k.
    Возвращает dict с mae, tolerance{k}, знаковым распределением и списком deltas.
    """
    deltas = [
        th - t
        for y, yh, t, th in zip(y_true, y_pred, t_true, t_pred)
        if y == 1 and yh == 1 and t is not None and th is not None
    ]
    n = len(deltas)
    mae = sum(abs(d) for d in deltas) / n if n else 0.0
    tol = {k: (sum(1 for d in deltas if abs(d) <= k) / n if n else 0.0) for k in tolerances}
    return {
        "n": n,
        "mae": mae,
        "tolerance": tol,
        "deltas": deltas,
        "early": sum(1 for d in deltas if d < 0),   # t̂* раньше t*
        "exact": sum(1 for d in deltas if d == 0),
        "late": sum(1 for d in deltas if d > 0),     # t̂* позже t*
    }
