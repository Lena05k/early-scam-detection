#!/usr/bin/env python3
"""Общий объектив L = λ_bin·L_bin + λ_sub·L_sub + λ_ons·L_ons — методичка §4, ур. (246)."""
from ._params import loss_cfg
from .detection import weighted_bce, multilabel_bce
from .onset import onset_loss


def total_loss(bin_logits, bin_targets, sub_logits, sub_targets,
               delta_hat, delta_gold, onset_idx, attacker_mask,
               hard_neg=None, scam_mask=None,
               lambda_bin=None, lambda_sub=None, lambda_ons=None,
               **onset_kwargs):
    """Взвешенная сумма трёх задач. λ по умолчанию из configs/default.yaml (loss)."""
    cfg = loss_cfg()
    lb = cfg["lambda_bin"] if lambda_bin is None else lambda_bin
    ls = cfg["lambda_sub"] if lambda_sub is None else lambda_sub
    lo = cfg["lambda_ons"] if lambda_ons is None else lambda_ons

    l_bin = weighted_bce(bin_logits, bin_targets, hard_neg=hard_neg)
    l_sub = multilabel_bce(sub_logits, sub_targets, scam_mask=scam_mask)
    l_ons = onset_loss(delta_hat, delta_gold, onset_idx, attacker_mask, **onset_kwargs)
    return lb * l_bin + ls * l_sub + lo * l_ons
