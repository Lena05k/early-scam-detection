"""Метрики scam-onset (методичка §5, §7). Чистый stdlib — без torch/numpy."""
from .oad import onset_kernel, dialogue_score, oad_score
from .earliness import binary_prf, f_latency, erde, onset_distance

__all__ = [
    "onset_kernel", "dialogue_score", "oad_score",
    "binary_prf", "f_latency", "erde", "onset_distance",
]
