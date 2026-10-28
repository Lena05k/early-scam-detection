import math
import sys
import pathlib

import pytest

torch = pytest.importorskip("torch")   # без extra [ml] весь модуль скипается

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scam_onset.losses import (        # noqa: E402
    weighted_bce, multilabel_bce, inverse_freq_weights, subtype_class_weights,
    onset_patl, onset_latency, onset_loss, first_fire_distribution, total_loss,
)


def _softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)   # стабильный softplus


# --------------------------------------------------------------- weighted_bce
def test_weighted_bce_matches_manual():
    logits = torch.tensor([2.0, -1.0])
    targets = torch.tensor([1.0, 0.0])
    # L = (1/2)[ w+·softplus(-z0) + w-·softplus(z1) ], w±=1
    expected = 0.5 * (_softplus(-2.0) + _softplus(-1.0))
    got = weighted_bce(logits, targets, w_pos=1.0, w_neg=1.0, gamma_hn=0.0)
    assert abs(got.item() - expected) < 1e-6


def test_weighted_bce_gamma_upweights_hard_negative():
    logits = torch.tensor([0.5, 0.5])      # i=1 — негатив и hard-neg
    targets = torch.tensor([1.0, 0.0])
    hn = torch.tensor([0.0, 1.0])
    l0 = weighted_bce(logits, targets, hard_neg=hn, w_pos=1.0, w_neg=1.0, gamma_hn=0.0)
    l2 = weighted_bce(logits, targets, hard_neg=hn, w_pos=1.0, w_neg=1.0, gamma_hn=2.0)
    assert l2.item() > l0.item()           # γ>0 усиливает вклад hard-neg-негатива


def test_inverse_freq_weights_from_prevalence():
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0])   # N=4, N+=1, N-=3
    w_pos, w_neg = inverse_freq_weights(targets)
    assert abs(w_pos - 4 / (2 * 1)) < 1e-9
    assert abs(w_neg - 4 / (2 * 3)) < 1e-9
    # авто == явная передача тех же весов
    logits = torch.tensor([0.3, -0.2, 0.1, 0.4])
    auto = weighted_bce(logits, targets, gamma_hn=0.0)
    explicit = weighted_bce(logits, targets, w_pos=w_pos, w_neg=w_neg, gamma_hn=0.0)
    assert abs(auto.item() - explicit.item()) < 1e-6


# ------------------------------------------------------------- multilabel_bce
def test_subtype_weights_normalised_sqrt():
    # класс0 в 1 диалоге, класс1 в 3; N=4
    targets = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # N=3
    n = 3
    raw = [math.sqrt(n / 1), math.sqrt(n / 3)]
    s = sum(raw)
    beta = subtype_class_weights(targets)
    assert abs(float(beta.sum()) - 1.0) < 1e-6
    assert abs(beta[0].item() - raw[0] / s) < 1e-6
    assert abs(beta[1].item() - raw[1] / s) < 1e-6


def test_scam_mask_zeroes_benign_contribution():
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    cw = torch.tensor([0.5, 0.5])
    mask = torch.tensor([1.0, 0.0])                    # 2-й диалог benign → не учить
    base = multilabel_bce(logits, targets, class_weights=cw, scam_mask=mask)
    # меняем логиты benign-строки — лосс не должен измениться
    logits2 = logits.clone()
    logits2[1] = torch.tensor([9.0, -9.0])
    other = multilabel_bce(logits2, targets, class_weights=cw, scam_mask=mask)
    assert abs(base.item() - other.item()) < 1e-9


# ------------------------------------------------------------------ onset_patl
def _all_attacker(n, t):
    return torch.ones(n, t, dtype=torch.bool)


def test_patl_perfect_prediction_near_zero():
    # T=3, онсет на t*=1: gold = [0,1,1]
    gold = torch.tensor([[0.0, 1.0, 1.0]])
    onset_idx = torch.tensor([1])
    dhat = gold.clone()                                 # идеальное предсказание
    loss = onset_patl(dhat, gold, onset_idx, _all_attacker(1, 3))
    assert loss.item() < 1e-2


def test_patl_benign_all_zero_near_zero():
    gold = torch.zeros(1, 3)
    onset_idx = torch.tensor([-1])                      # benign
    dhat = torch.full((1, 3), 1e-6)                     # δ̂→0
    loss = onset_patl(dhat, gold, onset_idx, _all_attacker(1, 3))
    assert loss.item() < 1e-2


def test_patl_onset_turn_weighted_2p5():
    # одинаковая по величине ошибка на онсет-реплике штрафуется сильнее, чем на neutral
    onset_idx = torch.tensor([1])
    gold = torch.tensor([[0.0, 1.0, 1.0]])
    am = _all_attacker(1, 3)
    err_at_onset = torch.tensor([[0.0, 0.5, 1.0]])      # ошибка на t=1 (w=2.5)
    err_at_neutral = torch.tensor([[0.5, 1.0, 1.0]])    # ошибка на t=0 (w=1.0), та же |Δ|
    l_on = onset_patl(err_at_onset, gold, onset_idx, am)
    l_ne = onset_patl(err_at_neutral, gold, onset_idx, am)
    assert l_on.item() > l_ne.item()


# --------------------------------------------------------------- onset_latency
def test_first_fire_sums_le_one():
    torch.manual_seed(0)
    d = torch.rand(4, 5)
    p = first_fire_distribution(d, _all_attacker(4, 5))
    assert torch.all(p.sum(dim=1) <= 1.0 + 1e-6)


def test_latency_asymmetry_late_costs_more():
    onset_idx = torch.tensor([1])
    am = _all_attacker(1, 3)
    late = torch.tensor([[0.0, 0.0, 1.0]])              # срабатывает на t=2, Δ=+1
    early = torch.tensor([[1.0, 0.0, 0.0]])             # срабатывает на t=0, Δ=−1
    l_late = onset_latency(late, onset_idx, am, eta_late=1.0, eta_early=0.5)
    l_early = onset_latency(early, onset_idx, am, eta_late=1.0, eta_early=0.5)
    assert l_late.item() > l_early.item()
    assert abs(l_late.item() - 1.0) < 1e-6 and abs(l_early.item() - 0.5) < 1e-6


def test_latency_differentiable():
    onset_logits = torch.zeros(1, 3, requires_grad=True)
    dhat = torch.sigmoid(onset_logits)
    loss = onset_latency(dhat, torch.tensor([1]), _all_attacker(1, 3))
    loss.backward()
    assert onset_logits.grad is not None
    assert torch.isfinite(onset_logits.grad).all()


def test_latency_benign_is_zero():
    dhat = torch.rand(1, 3)
    loss = onset_latency(dhat, torch.tensor([-1]), _all_attacker(1, 3))
    assert abs(loss.item()) < 1e-9


def test_onset_loss_is_patl_plus_mu_latency():
    gold = torch.tensor([[0.0, 1.0, 1.0]])
    onset_idx = torch.tensor([1])
    dhat = torch.tensor([[0.2, 0.7, 0.6]])
    am = _all_attacker(1, 3)
    mu = 0.5
    combined = onset_loss(dhat, gold, onset_idx, am, mu=mu)
    patl = onset_patl(dhat, gold, onset_idx, am)
    lat = onset_latency(dhat, onset_idx, am)
    assert abs(combined.item() - (patl.item() + mu * lat.item())) < 1e-6


# ------------------------------------------------------------------ total_loss
def test_total_loss_decreases_under_sgd():
    n, t, k = 3, 3, 2
    am = _all_attacker(n, t)
    onset_idx = torch.tensor([1, 0, -1])               # 2 scam, 1 benign
    # gold онсет-цель δ = 1[t≥t*] на атакующих; benign → нули
    delta_gold = torch.tensor([[0.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0]])
    bin_t = torch.tensor([1.0, 1.0, 0.0])
    sub_t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    scam_mask = torch.tensor([1.0, 1.0, 0.0])
    hard_neg = torch.tensor([0.0, 0.0, 1.0])

    bin_logits = torch.zeros(n, requires_grad=True)
    sub_logits = torch.zeros(n, k, requires_grad=True)
    onset_logits = torch.zeros(n, t, requires_grad=True)
    opt = torch.optim.SGD([bin_logits, sub_logits, onset_logits], lr=0.5)

    losses = []
    for _ in range(6):
        opt.zero_grad()
        dhat = torch.sigmoid(onset_logits)
        loss = total_loss(bin_logits, bin_t, sub_logits, sub_t,
                          dhat, delta_gold, onset_idx, am,
                          hard_neg=hard_neg, scam_mask=scam_mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]                       # 5 шагов SGD уменьшают лосс
