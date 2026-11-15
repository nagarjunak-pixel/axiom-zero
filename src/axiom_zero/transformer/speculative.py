"""Leviathan-Kalman-Matias (ICML 2023) Exact Lossless Speculative Decoding Engine."""

from __future__ import annotations

import time

import numpy as np
import torch

from axiom_zero.kernels.native_bridge import NativeKernelBridge, get_kernel_bridge
from axiom_zero.transformer.gqa_model import (
    TOKEN_TO_ID,
    VOCAB_SIZE,
    GQATransformerLM,
    train_target_and_draft_pair,
)
from axiom_zero.types import SpeculativeBenchmarkReport


def verify_exact_distribution_equivalence(
    target_probs_1d: np.ndarray,
    draft_probs_1d: np.ndarray,
    *,
    num_trials: int = 6000,
    seed: int = 123,
    bridge: NativeKernelBridge | None = None,
) -> float:
    """Prove empirically via Monte Carlo Total Variation Distance that speculative
    rejection + residual resampling samples from the EXACT target distribution p(x),
    even when q(x) differs arbitrarily from p(x).
    """
    kern = bridge or get_kernel_bridge()
    rng = np.random.default_rng(seed)
    v = int(target_probs_1d.shape[0])

    p = np.asarray(target_probs_1d, dtype=np.float32)
    p = p / np.sum(p)
    q = np.asarray(draft_probs_1d, dtype=np.float32)
    q = q / np.sum(q)

    # Stack (K+1, V) for K=1 single-step distribution verification
    p_mat = np.vstack([p, p]).astype(np.float32)
    q_mat = q.reshape(1, v).astype(np.float32)

    counts = np.zeros(v, dtype=np.int64)
    draft_draws = rng.choice(v, size=num_trials, p=q)
    r_draws = rng.random(size=num_trials, dtype=np.float32)
    u_draws = rng.random(size=num_trials, dtype=np.float32)

    for idx in range(num_trials):
        d_tok = np.array([int(draft_draws[idx])], dtype=np.int32)
        r_vec = np.array([float(r_draws[idx])], dtype=np.float32)
        emitted = kern.speculative_verify_step(
            p_mat,
            q_mat,
            d_tok,
            r_vec,
            float(u_draws[idx]),
        )
        counts[emitted[0]] += 1

    empirical_p = counts.astype(np.float64) / float(num_trials)
    tv_distance = 0.5 * float(np.sum(np.abs(empirical_p - p.astype(np.float64))))
    return round(tv_distance, 5)


class SpeculativeDecodingEngine:
    """Executes real multi-token speculative decoding over trained PyTorch Draft & Target models."""

    def __init__(
        self,
        target: GQATransformerLM,
        draft: GQATransformerLM,
        *,
        lookahead_k: int = 4,
        temperature: float = 0.8,
        bridge: NativeKernelBridge | None = None,
    ) -> None:
        self.target = target
        self.draft = draft
        self.lookahead_k = lookahead_k
        self.temperature = max(0.05, temperature)
        self.bridge = bridge or get_kernel_bridge()

    @torch.no_grad()
    def decode_autoregressive_baseline(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        seed: int = 101,
    ) -> tuple[list[int], int, float]:
        rng = np.random.default_rng(seed)
        dev = next(self.target.parameters()).device
        tokens = list(prompt_ids)
        target_forwards = 0

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            inp = torch.tensor(
                [tokens[-self.target.cfg.max_seq_len :]], dtype=torch.long, device=dev
            )
            logits, _ = self.target(inp)
            target_forwards += 1
            scaled = (logits[0, -1].detach().cpu().numpy() / self.temperature).astype(np.float32)
            probs = self.bridge.fused_softmax(scaled)
            next_tok = int(rng.choice(len(probs), p=probs / np.sum(probs)))
            tokens.append(next_tok)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return tokens, target_forwards, elapsed_ms

    @torch.no_grad()
    def decode_speculative(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        seed: int = 101,
    ) -> tuple[list[int], int, int, int, float]:
        """Returns `(tokens, target_forwards, accepted_draft_count, proposed_draft_count, latency_ms)`."""
        rng = np.random.default_rng(seed)
        dev = next(self.target.parameters()).device
        tokens = list(prompt_ids)
        target_forwards = 0
        accepted_draft = 0
        proposed_draft = 0
        target_len = len(prompt_ids) + max_new_tokens

        t0 = time.perf_counter()
        while len(tokens) < target_len:
            rem = target_len - len(tokens)
            k = min(self.lookahead_k, rem)

            # 1. Draft proposes k tokens sequentially on lightweight 1-layer model
            draft_seq = list(tokens)
            draft_tokens_list: list[int] = []
            draft_prob_rows: list[np.ndarray] = []

            for _ in range(k):
                d_inp = torch.tensor(
                    [draft_seq[-self.draft.cfg.max_seq_len :]], dtype=torch.long, device=dev
                )
                d_logits, _ = self.draft(d_inp)
                d_scaled = (d_logits[0, -1].detach().cpu().numpy() / self.temperature).astype(
                    np.float32
                )
                q_row = self.bridge.fused_softmax(d_scaled)
                q_norm = q_row / np.sum(q_row)
                d_tok = int(rng.choice(len(q_norm), p=q_norm))
                draft_tokens_list.append(d_tok)
                draft_prob_rows.append(q_row)
                draft_seq.append(d_tok)

            proposed_draft += k

            # 2. Target verifies all k+1 positions in ONE single parallel forward pass!
            window = draft_seq[-self.target.cfg.max_seq_len :]
            # Ensure we have at least k+1 tokens in window
            t_inp = torch.tensor([window], dtype=torch.long, device=dev)
            t_logits, _ = self.target(t_inp)
            target_forwards += 1

            # Slice the last (k + 1) logit positions corresponding to prefix + each draft step
            relevant_logits = (
                t_logits[0, -(k + 1) :].detach().cpu().numpy() / self.temperature
            ).astype(np.float32)
            target_probs_mat = self.bridge.fused_softmax(relevant_logits)
            draft_probs_mat = np.vstack(draft_prob_rows).astype(np.float32)
            draft_tok_arr = np.asarray(draft_tokens_list, dtype=np.int32)
            uniform_r = rng.random(size=k, dtype=np.float32)
            resample_u = float(rng.random())

            emitted = self.bridge.speculative_verify_step(
                target_probs_mat,
                draft_probs_mat,
                draft_tok_arr,
                uniform_r,
                resample_u,
            )

            # Count how many draft tokens matched before rejection/bonus
            matched_in_step = 0
            for idx_e in range(min(k, len(emitted))):
                if emitted[idx_e] == draft_tokens_list[idx_e]:
                    matched_in_step += 1
                else:
                    break
            accepted_draft += matched_in_step

            for tok in emitted:
                if len(tokens) < target_len:
                    tokens.append(tok)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return tokens, target_forwards, accepted_draft, proposed_draft, elapsed_ms


def run_speculative_benchmark(
    *,
    generated_tokens: int = 96,
    lookahead_k: int = 4,
    seed: int = 11,
) -> SpeculativeBenchmarkReport:
    """Train real Target & Draft models, benchmark real wall-clock + forward reduction,
    and measure exact Monte Carlo Total Variation Distance.
    """
    target, draft, _, _ = train_target_and_draft_pair(seed=seed, device="cpu")
    bridge = get_kernel_bridge()
    engine = SpeculativeDecodingEngine(target, draft, lookahead_k=lookahead_k, bridge=bridge)

    prompt = [TOKEN_TO_ID["<BOS>"], TOKEN_TO_ID["REG_AUDIT"], TOKEN_TO_ID["STEP_1"]]

    # Warmup
    engine.decode_autoregressive_baseline(prompt, max_new_tokens=8, seed=seed)
    engine.decode_speculative(prompt, max_new_tokens=8, seed=seed)

    _, base_fwds, base_ms = engine.decode_autoregressive_baseline(
        prompt, max_new_tokens=generated_tokens, seed=seed
    )
    _, spec_fwds, acc_cnt, prop_cnt, spec_ms = engine.decode_speculative(
        prompt, max_new_tokens=generated_tokens, seed=seed
    )

    # Measure empirical TV distance on real first-step target vs draft distributions
    with torch.no_grad():
        inp = torch.tensor([prompt], dtype=torch.long)
        t_log, _ = target(inp)
        d_log, _ = draft(inp)
        p_real = bridge.fused_softmax(t_log[0, -1].numpy().astype(np.float32))
        q_real = bridge.fused_softmax(d_log[0, -1].numpy().astype(np.float32))

    tv_dist = verify_exact_distribution_equivalence(
        p_real, q_real, num_trials=5000, seed=seed, bridge=bridge
    )

    fwd_reduction = float(base_fwds) / max(1.0, float(spec_fwds))
    mean_acc = float(acc_cnt) / max(1.0, float(spec_fwds))
    acc_pct = (float(acc_cnt) / max(1.0, float(prop_cnt))) * 100.0
    speedup = base_ms / max(0.01, spec_ms)

    return SpeculativeBenchmarkReport(
        kernel_backend=bridge.backend,
        device="cpu-c++20-hybrid",
        vocab_size=VOCAB_SIZE,
        draft_lookahead_k=lookahead_k,
        generated_tokens=generated_tokens,
        baseline_target_forwards=base_fwds,
        speculative_target_forwards=spec_fwds,
        forward_pass_reduction_x=round(fwd_reduction, 3),
        mean_accepted_draft_tokens=round(mean_acc, 3),
        acceptance_rate_pct=round(acc_pct, 2),
        empirical_tv_distance=tv_dist,
        baseline_latency_ms=round(base_ms, 2),
        speculative_latency_ms=round(spec_ms, 2),
        wall_clock_speedup_x=round(speedup, 3),
    )
