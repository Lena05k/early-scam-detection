"""Стадии 0-1 конвейера (методичка §9.2): скелеты сценариев -> LLM-синтез диалогов.

Выход — ЧЕРНОВЫЕ метки (gold_src="llm_draft") для human-LLM верификации (стадия 2);
онсет и метки — строго по замороженному гайдлайну v1.2.
"""
from .scenarios import BENIGN_SCENARIOS, SCAM_SCENARIOS, SCENARIOS, Scenario
from .skeleton import Skeleton, build_skeletons, build_corpus_skeletons
from .prompts import build_prompt
from .generate import (
    AnthropicClient,
    DeepSeekClient,
    GenerationError,
    RoutedClient,
    build_client,
    build_record,
    extract_json,
    load_config,
    render_dialogue,
    run_pipeline,
    validate_record,
    vendor_of,
)

__all__ = [
    "Scenario", "SCENARIOS", "SCAM_SCENARIOS", "BENIGN_SCENARIOS",
    "Skeleton", "build_skeletons", "build_corpus_skeletons", "build_prompt",
    "AnthropicClient", "DeepSeekClient", "RoutedClient", "build_client",
    "vendor_of", "GenerationError", "extract_json", "build_record",
    "validate_record", "run_pipeline", "render_dialogue", "load_config",
]
