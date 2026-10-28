#!/usr/bin/env python3
"""Onset-Aware Detection score (OAD) — методичка §7.

Единый показатель ∈[0,1], объединяющий (1) корректность бинарной метки (вкл.
benign/hard-negative), (2) точность и (3) своевременность онсета относительно t*
с асимметрией late/early. Больше — лучше.

Соглашения: y ∈ {0,1}; t* — int (номер реплики) или None для ∅ (benign).
Δ = t̂* − t*  (положительный ⟹ позднее онсета ⟹ штраф λ_late).
"""
from math import exp

from ._params import section


def onset_kernel(delta, lam_late, lam_early):
    """q(Δ) = exp(−[Δ]_+/λ_L − [−Δ]_+/λ_E), Δ = t̂*−t*  (§7).

    q(0)=1, q(±∞)=0. λ_L<λ_E ⟹ опоздание штрафуется сильнее.
    """
    late = delta if delta > 0 else 0.0        # [Δ]_+
    early = -delta if delta < 0 else 0.0       # [−Δ]_+
    return exp(-late / lam_late - early / lam_early)


def _defaults():
    p = section("oad")
    return p["beta"], p["lambda_late"], p["lambda_early"]


def dialogue_score(y, yhat, t, that, beta, lam_late, lam_early):
    """Пер-диалоговый s_i (§7). t/that — онсеты gold/pred (int или None)."""
    if y == 0:
        return 1.0 if yhat == 0 else 0.0          # верный benign : ложное срабатывание
    # y == 1 (scam)
    if yhat == 0:
        return 0.0                                # пропуск скама
    # scam детектирован: β + (1−β)·q(Δ)
    if t is None or that is None:
        q = 0.0                                   # онсет не определён ⟹ только бинарный кредит
    else:
        q = onset_kernel(that - t, lam_late, lam_early)
    return beta + (1.0 - beta) * q


def oad_score(y_true, y_pred, t_true, t_pred,
              beta=None, lam_late=None, lam_early=None, macro=True):
    """OAD-агрегат (§7).

    macro=True: ½·(⟨s⟩_benign + ⟨s⟩_scam) — устойчив к дисбалансу (по умолчанию).
    macro=False: микро-среднее s по всем диалогам.
    Параметры по умолчанию — из configs/default.yaml (секция oad).
    """
    d_beta, d_ll, d_le = _defaults()
    beta = d_beta if beta is None else beta
    lam_late = d_ll if lam_late is None else lam_late
    lam_early = d_le if lam_early is None else lam_early

    s_benign, s_scam = [], []
    for y, yhat, t, that in zip(y_true, y_pred, t_true, t_pred):
        s = dialogue_score(y, yhat, t, that, beta, lam_late, lam_early)
        (s_scam if y == 1 else s_benign).append(s)

    if macro:
        parts = [sum(g) / len(g) for g in (s_benign, s_scam) if g]
        return sum(parts) / len(parts) if parts else 0.0
    alls = s_benign + s_scam
    return sum(alls) / len(alls) if alls else 0.0
