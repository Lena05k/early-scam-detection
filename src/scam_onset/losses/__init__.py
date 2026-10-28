"""Функции потерь scam-onset (методичка §4). PyTorch — только в extra [ml].

Импорт этого пакета тянет torch; юнит-тесты гейтят его через pytest.importorskip.
"""
from .detection import (
    weighted_bce, multilabel_bce, inverse_freq_weights, subtype_class_weights,
)
from .onset import (
    onset_patl, onset_latency, onset_loss, first_fire_distribution,
)
from .total import total_loss

__all__ = [
    "weighted_bce", "multilabel_bce", "inverse_freq_weights", "subtype_class_weights",
    "onset_patl", "onset_latency", "onset_loss", "first_fire_distribution",
    "total_loss",
]
