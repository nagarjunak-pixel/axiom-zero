"""Tests for RoPE + GQA + SwiGLU Transformer, Top-K SAE invariants, and causal steering."""

from __future__ import annotations

import pytest
import torch

from axiom_zero.interp.sae import TopKSparseAutoencoder, train_sae_and_discover_circuits
from axiom_zero.transformer.gqa_model import (
    VOCAB_SIZE,
    GQATransformerLM,
    TransformerConfig,
)


def test_gqa_transformer_shapes_and_weight_tying() -> None:
    cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
    )
    model = GQATransformerLM(cfg)
    assert model.lm_head.weight is model.tok_emb.weight

    inp = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits, captured = model(inp, hook_layer=1)
    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert captured is not None
    assert captured.shape == (2, 8, 32)


def test_topk_sae_unit_norm_and_sparsity_invariants() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        TopKSparseAutoencoder(d_model=16, expansion_factor=4, top_k=0)

    sae, _, report = train_sae_and_discover_circuits(
        hook_layer=5,
        expansion_factor=6,
        top_k=6,
        sae_epochs=75,
        steering_alpha=45.0,
        target_regime_id=3,
        seed=19,
    )

    # 1. Exact unit-norm decoder columns ||d_i||_2 == 1
    col_norms = sae.w_dec.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    # 2. High variance explained + exact Top-K sparsity
    assert report.reconstruction_r2 > 0.90
    assert report.mean_l0_active <= 6.0 + 1e-5
    assert len(report.discovered_features) == 4

    # 3. Causal residual-stream steering shifts covert policy bypass into RISK_QUARANTINE
    assert report.baseline_regime_prob < 0.05
    assert report.steered_regime_prob > 0.90
    assert report.probability_shift_delta > 0.85
