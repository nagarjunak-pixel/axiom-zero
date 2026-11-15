"""Tests for C++20 native kernels, NumPy parity, Paged KV CoW, and lossless speculative decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_zero.kernels.native_bridge import NativeKernelBridge, get_kernel_bridge
from axiom_zero.transformer.gqa_model import PagedKVCacheManager
from axiom_zero.transformer.speculative import (
    run_speculative_benchmark,
    verify_exact_distribution_equivalence,
)


def test_native_cpp20_vs_numpy_fallback_exact_parity() -> None:
    native = get_kernel_bridge()
    assert native.is_native_cpp is True
    assert "c++20-native" in native.backend

    fallback = NativeKernelBridge(csrc_path=Path("/nonexistent/axiom_kernels.cpp"))
    assert fallback.is_native_cpp is False
    assert fallback.backend == "numpy-fallback"

    rng = np.random.default_rng(42)
    logits_1d = rng.normal(size=16).astype(np.float32)
    logits_2d = rng.normal(size=(5, 16)).astype(np.float32)

    assert np.allclose(
        native.fused_softmax(logits_1d), fallback.fused_softmax(logits_1d), atol=1e-5
    )
    assert np.allclose(
        native.fused_softmax(logits_2d), fallback.fused_softmax(logits_2d), atol=1e-5
    )

    p_mat = native.fused_softmax(logits_2d)
    q_mat = native.fused_softmax(rng.normal(size=(4, 16)).astype(np.float32))
    draft_toks = np.array([1, 3, 5, 7], dtype=np.int32)

    # Rejection case + acceptance case
    for r_vals in (
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.5, 0.5, 0.5], dtype=np.float32),
    ):
        out_cpp = native.speculative_verify_step(p_mat, q_mat, draft_toks, r_vals, 0.37)
        out_py = fallback.speculative_verify_step(p_mat, q_mat, draft_toks, r_vals, 0.37)
        assert out_cpp == out_py

    # Identical p == q degenerate rejection fallback branch
    p_ident = np.vstack([q_mat, q_mat[0:1]])
    out_ident_cpp = native.speculative_verify_step(
        p_ident, q_mat, draft_toks, np.array([1.5, 0.0, 0.0, 0.0], dtype=np.float32), 0.42
    )
    out_ident_py = fallback.speculative_verify_step(
        p_ident, q_mat, draft_toks, np.array([1.5, 0.0, 0.0, 0.0], dtype=np.float32), 0.42
    )
    assert out_ident_cpp == out_ident_py

    # Paged KV attention parity + zero-context edge case
    q_vec = rng.normal(size=8).astype(np.float32)
    k_blks = rng.normal(size=(4, 4, 8)).astype(np.float32)
    v_blks = rng.normal(size=(4, 4, 8)).astype(np.float32)
    bt = np.array([2, 0, 3], dtype=np.int32)
    assert np.allclose(
        native.paged_kv_attention(q_vec, k_blks, v_blks, bt, 10),
        fallback.paged_kv_attention(q_vec, k_blks, v_blks, bt, 10),
        atol=1e-5,
    )
    assert np.allclose(fallback.paged_kv_attention(q_vec, k_blks, v_blks, bt, 0), 0.0)


def test_paged_kv_copy_on_write_isolation() -> None:
    mgr = PagedKVCacheManager(num_blocks=4, block_size=2, head_dim=4)
    assert mgr.free_block_count == 4

    bt_parent: list[int] = []
    k0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v0 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    bt_parent = mgr.append_kv(bt_parent, 0, k0, v0)
    assert mgr.free_block_count == 3

    # Fork beam shares physical block without allocating until mutated
    bt_child = mgr.fork_sequence(bt_parent)
    assert bt_child == bt_parent
    assert mgr.free_block_count == 3

    # Child appends slot 1 -> triggers Copy-on-Write allocation!
    k1_child = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v1_child = np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)
    bt_child = mgr.append_kv(bt_child, 1, k1_child, v1_child)
    assert bt_child != bt_parent
    assert mgr.free_block_count == 2

    q = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    parent_out = mgr.attend_single_query(q, bt_parent, context_len=1)
    child_out = mgr.attend_single_query(q, bt_child, context_len=2)
    assert np.allclose(parent_out, v0, atol=1e-4)
    assert not np.allclose(parent_out, child_out)

    mgr.free_sequence(bt_parent)
    mgr.free_sequence(bt_child)
    assert mgr.free_block_count == 4

    small = PagedKVCacheManager(num_blocks=1, block_size=1, head_dim=2)
    tb = small.append_kv([], 0, np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Out of physical KV cache blocks"):
        small.append_kv(tb, 1, np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))


@settings(max_examples=12, deadline=None)
@given(
    p_raw=st.lists(st.floats(min_value=0.05, max_value=5.0), min_size=5, max_size=5),
    q_raw=st.lists(st.floats(min_value=0.05, max_value=5.0), min_size=5, max_size=5),
)
def test_hypothesis_speculative_lossless_distribution_theorem(
    p_raw: list[float],
    q_raw: list[float],
) -> None:
    p = np.asarray(p_raw, dtype=np.float32)
    q = np.asarray(q_raw, dtype=np.float32)
    tv = verify_exact_distribution_equivalence(p, q, num_trials=4500, seed=77)
    assert tv < 0.045


def test_run_speculative_benchmark_end_to_end() -> None:
    rep = run_speculative_benchmark(generated_tokens=48, lookahead_k=4, seed=11)
    assert rep.generated_tokens == 48
    assert rep.baseline_target_forwards == 48
    assert rep.speculative_target_forwards < rep.baseline_target_forwards
    assert rep.forward_pass_reduction_x > 1.5
    assert rep.empirical_tv_distance < 0.05
