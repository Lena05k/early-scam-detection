"""Загрузка параметров метрик из configs/default.yaml (не хардкодить — CLAUDE.md).

Источник истины — configs/default.yaml. Фолбэк-значения ниже зеркалят конфиг и
используются лишь если pyyaml/файл недоступны, чтобы юнит-тесты не падали из-за среды.
"""
import functools
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CFG = _ROOT / "configs" / "default.yaml"

# зеркало configs/default.yaml (секции oad, earliness) — см. комментарий модуля
_FALLBACK = {
    "oad": {"beta": 0.5, "lambda_late": 1.0, "lambda_early": 2.0},
    "earliness": {"p": 0.5, "o": 3.0, "c_fp": 1.0, "c_fn": 1.0, "c_tp": 1.0},
}


@functools.lru_cache(maxsize=1)
def _load_cfg():
    try:
        import yaml
        with open(_CFG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def section(name):
    """Значения секции из конфига поверх фолбэка (только известные ключи)."""
    out = dict(_FALLBACK.get(name, {}))
    cfg = _load_cfg().get(name) or {}
    for k in out:
        if k in cfg and cfg[k] is not None:
            out[k] = cfg[k]
    return out
