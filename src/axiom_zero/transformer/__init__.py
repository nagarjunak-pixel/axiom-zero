"""Transformer & Speculative Decoding package."""

from axiom_zero.transformer.gqa_model import (
    GQATransformerLM,
    PagedKVCacheManager,
    TransformerConfig,
    train_target_and_draft_pair,
)
from axiom_zero.transformer.speculative import (
    SpeculativeDecodingEngine,
    run_speculative_benchmark,
    verify_exact_distribution_equivalence,
)

__all__ = [
    "GQATransformerLM",
    "PagedKVCacheManager",
    "SpeculativeDecodingEngine",
    "TransformerConfig",
    "run_speculative_benchmark",
    "train_target_and_draft_pair",
    "verify_exact_distribution_equivalence",
]
