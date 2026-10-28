#!/usr/bin/env python3
"""Лоссы бинарной детекции и подтипов — методичка §4(i), §4(ii).

PyTorch, функциональные (без nn.Module). Стабильность: F.logsigmoid /
F.binary_cross_entropy, без «голого» log.
"""
import torch
import torch.nn.functional as F

from ._params import loss_cfg


def inverse_freq_weights(targets):
    """w_+ = N/(2N_+), w_− = N/(2N_−) — инверсно-частотные веса §4(i). [приор]"""
    n = targets.shape[0]
    n_pos = float(targets.sum())
    n_neg = n - n_pos
    w_pos = n / (2.0 * n_pos) if n_pos > 0 else 0.0
    w_neg = n / (2.0 * n_neg) if n_neg > 0 else 0.0
    return w_pos, w_neg


def weighted_bce(logits, targets, hard_neg=None, w_pos=None, w_neg=None, gamma_hn=None):
    """Взвешенный BCE с hard-negative up-weight — §4(i). [приор: BCE; новое: hard-neg]

    L = −(1/N) Σ_i [ w_+ y_i logσ(z_i) + (1+γ_hn h_i) w_− (1−y_i) log(1−σ(z_i)) ].
    w_pos/w_neg=None ⟹ инверсно-частотные из батча. h_i∈{0,1}; γ_hn>0 усиливает штраф
    за ложную тревогу на hard-negative (Пр. 3,4,9,17,22). γ_hn по умолчанию из конфига.
    """
    if gamma_hn is None:
        gamma_hn = loss_cfg()["hard_neg_gamma"]
    logits = logits.float()
    targets = targets.float()
    n = logits.shape[0]
    if w_pos is None or w_neg is None:
        aw_pos, aw_neg = inverse_freq_weights(targets)
        w_pos = aw_pos if w_pos is None else w_pos
        w_neg = aw_neg if w_neg is None else w_neg
    h = torch.zeros_like(targets) if hard_neg is None else hard_neg.float()

    log_p = F.logsigmoid(logits)        # logσ(z)   — стабильно
    log_1mp = F.logsigmoid(-logits)     # log(1−σ(z))
    pos = w_pos * targets * log_p
    neg = (1.0 + gamma_hn * h) * w_neg * (1.0 - targets) * log_1mp
    return -(pos + neg).sum() / n


def subtype_class_weights(targets):
    """β_k = (N/N_k)^0.5, нормированные Σβ=1 — §4(ii), степень балансировки 0.5 (EARS). [адапт.]

    ⚠ Расхождение: при N_k=0 (подтип отсутствует в батче) (N/N_k)^0.5→∞. Клампим
    N_k≥1, чтобы избежать inf; на реальном корпусе N_k>0. Помечено [адапт.].
    """
    n = targets.shape[0]
    n_k = targets.float().sum(dim=0).clamp(min=1.0)
    raw = (n / n_k).sqrt()
    return raw / raw.sum()


def multilabel_bce(logits, targets, class_weights=None, scam_mask=None):
    """Мультилейбл-BCE подтипов с весами классов — §4(ii). [приор: BCE; адапт.: веса EARS]

    L = −(1/N) Σ_i m_i Σ_k β_k [ y_ik logσ(z_ik) + (1−y_ik) log(1−σ(z_ik)) ].
    scam_mask m_i: benign→0 (подтипы учим только на scam; benign-цель = нули, §3.2).
    class_weights=None ⟹ β_k из батча. Нормировка на N (число диалогов), как в §4(ii).
    """
    logits = logits.float()
    targets = targets.float()
    n, _ = logits.shape
    if class_weights is None:
        class_weights = subtype_class_weights(targets)
    m = torch.ones(n) if scam_mask is None else scam_mask.float()

    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    per_k = targets * log_p + (1.0 - targets) * log_1mp        # [N, K]
    per_i = (class_weights.unsqueeze(0) * per_k).sum(dim=1)    # [N]
    return -(m * per_i).sum() / n
