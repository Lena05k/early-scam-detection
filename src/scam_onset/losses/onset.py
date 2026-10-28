#!/usr/bin/env python3
"""Turn-level onset-потеря — методичка §4(iii): PATL + латентностный штраф.

Каузальность (CLAUDE.md инв.5): δ̂_t = σ(z_t(C_≤t)) — предсказание на префиксе;
здесь получаем уже посчитанные каузальные δ̂ и не заглядываем вперёд.

Формат переменной длины: padded [N, T_max]; attacker_mask [N,T] (bool) — только
атакующие реплики участвуют; onset_idx [N] (long) — индекс столбца t* (0-based),
benign = −1 (сигнал «нет онсета»).

δ̂ — ВЕРОЯТНОСТИ (сигмоида уже применена каузально): нужны для survival-произведения
в латентности. В PATL клампим δ̂∈[eps,1−eps] и используем F.binary_cross_entropy
(framework-стабильно), а не «голый» log. ⚠ Расхождение с §4: CE в методичке над log δ̂;
клампинг — [адапт.] ради стабильности; латентностный член требует именно вероятностей.
"""
import torch
import torch.nn.functional as F

from ._params import loss_cfg

_EPS = 1e-6


def onset_patl(delta_hat, delta_gold, onset_idx, attacker_mask,
               w_onset=None, w_post=None, w_neutral=None):
    """PATL: нормированная взвешенная CE по атакующим репликам — §4(iii) ч.1.
    [взято у EARS ур.9; изменена задача на бинарный онсет обмана]

    L = Σ_{t∈A} w_t·CE(δ̂_t, δ_t) / Σ_{t∈A} w_t,  усреднённая по диалогам.
    w_t = 2.5 (t=t*) / 1.5 (t>t*) / 1.0 (t<t*). benign (onset_idx<0): все w=w_neutral,
    δ_gold=0 (цель «никогда не срабатывать»). Веса по умолчанию из конфига.
    """
    pw = loss_cfg()["patl_weights"]
    w_onset = pw["onset"] if w_onset is None else w_onset
    w_post = pw["post_onset"] if w_post is None else w_post
    w_neutral = pw["neutral"] if w_neutral is None else w_neutral

    d = delta_hat.float().clamp(_EPS, 1.0 - _EPS)
    gold = delta_gold.float()
    ce = F.binary_cross_entropy(d, gold, reduction="none")     # [N, T]

    n, t = d.shape
    t_idx = torch.arange(t).unsqueeze(0).expand(n, t)          # [N,T] long
    onset = onset_idx.long().view(n, 1)
    # фазовые веса: neutral → post (t>t*) → onset (t==t*)
    w = torch.full((n, t), float(w_neutral))
    w = torch.where(t_idx > onset, torch.full((n, t), float(w_post)), w)
    w = torch.where(t_idx == onset, torch.full((n, t), float(w_onset)), w)
    benign = (onset_idx.view(n, 1) < 0).expand(n, t)
    w = torch.where(benign, torch.full((n, t), float(w_neutral)), w)     # benign → neutral

    w_eff = w * attacker_mask.float()                          # non-attacker → вес 0
    num = (w_eff * ce).sum(dim=1)
    den = w_eff.sum(dim=1)
    valid = den > 0
    if valid.any():
        return (num[valid] / den[valid]).mean()
    return (num * 0.0).sum()


def first_fire_distribution(delta_hat, attacker_mask):
    """p_fire[t] = δ̂_t · Π_{s<t, s∈A}(1−δ̂_s) — survival, первое срабатывание. [новое]

    Σ_t p_fire = 1 − Π(1−δ̂) ≤ 1. Не-атакующие реплики не срабатывают (фактор 1).
    """
    d = delta_hat.float() * attacker_mask.float()
    n = d.shape[0]
    one_minus = 1.0 - d
    incl = torch.cumprod(one_minus, dim=1)                     # Π_{s≤t}
    excl = torch.cat([torch.ones(n, 1), incl[:, :-1]], dim=1)  # Π_{s<t}
    return d * excl


def onset_latency(delta_hat, onset_idx, attacker_mask, eta_late=None, eta_early=None):
    """Дифференцируемый штраф латентности E[ρ] — §4(iii) ч.2. [новое]

    ρ(Δ) = η_late·[Δ]_+ + η_early·[−Δ]_+, Δ = t − t*.
    E[ρ] = Σ_{t∈A} p_fire[t]·ρ(Δ_t). η_late>η_early ⟹ опоздание дороже.
    benign (onset_idx<0): латентность = 0 (учитывается только на scam) — benign→∅ в лоссе.
    Усреднение по scam-диалогам.
    """
    lat = loss_cfg()["latency"]
    eta_late = lat["eta_late"] if eta_late is None else eta_late
    eta_early = lat["eta_early"] if eta_early is None else eta_early

    n, t = delta_hat.shape
    p_fire = first_fire_distribution(delta_hat, attacker_mask)
    t_idx = torch.arange(t).float().unsqueeze(0)               # [1,T]
    delta = t_idx - onset_idx.float().unsqueeze(1)             # Δ = t − t*  [N,T]
    rho = eta_late * delta.clamp(min=0) + eta_early * (-delta).clamp(min=0)
    per_i = (p_fire * rho).sum(dim=1)                          # [N]

    scam = (onset_idx >= 0).float()
    per_i = per_i * scam                                       # benign → 0
    denom = scam.sum()
    return per_i.sum() / denom if float(denom) > 0 else per_i.sum() * 0.0


def onset_loss(delta_hat, delta_gold, onset_idx, attacker_mask,
               w_onset=None, w_post=None, w_neutral=None,
               eta_late=None, eta_early=None, mu=None):
    """L_ons = L_PATL + μ·E[ρ] — §4(iii) итог. μ по умолчанию из конфига."""
    if mu is None:
        mu = loss_cfg()["latency"]["mu"]
    patl = onset_patl(delta_hat, delta_gold, onset_idx, attacker_mask,
                      w_onset, w_post, w_neutral)
    lat = onset_latency(delta_hat, onset_idx, attacker_mask, eta_late, eta_early)
    return patl + mu * lat
