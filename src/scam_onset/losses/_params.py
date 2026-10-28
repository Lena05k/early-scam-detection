"""Дефолты лоссов из configs/default.yaml (секция loss). Не хардкодить — CLAUDE.md.

Чистый stdlib: не тянет torch (импортируется и без extra [ml]). Фолбэк зеркалит
конфиг на случай отсутствия pyyaml/файла.
"""
import copy
import functools
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CFG = _ROOT / "configs" / "default.yaml"

# зеркало configs/default.yaml → loss
_FALLBACK = {
    "lambda_bin": 1.0,
    "lambda_sub": 1.0,
    "lambda_ons": 1.0,
    "hard_neg_gamma": 1.0,
    "patl_weights": {"onset": 2.5, "post_onset": 1.5, "neutral": 1.0},
    "latency": {"eta_late": 1.0, "eta_early": 0.5, "mu": 0.5},
}


@functools.lru_cache(maxsize=1)
def loss_cfg():
    """Слитый dict секции loss (конфиг поверх фолбэка). Не мутировать возвращённое."""
    base = copy.deepcopy(_FALLBACK)
    try:
        import yaml
        cfg = (yaml.safe_load(open(_CFG, encoding="utf-8")) or {}).get("loss") or {}
    except Exception:
        cfg = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base
