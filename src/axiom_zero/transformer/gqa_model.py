"""From-scratch PyTorch RoPE + Grouped-Query Attention (GQA) + SwiGLU Transformer & Paged KV Cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from axiom_zero.kernels.native_bridge import get_kernel_bridge

REGIME_NAMES: tuple[str, ...] = (
    "COMPLIANT_AUDIT",
    "COVERT_POLICY_BYPASS",
    "FORMAL_ARITHMETIC",
    "RISK_QUARANTINE",
)

VOCAB_TOKENS: tuple[str, ...] = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "REG_AUDIT",
    "REG_BYPASS",
    "REG_MATH",
    "REG_RISK",
    "CHECK_SOX",
    "VERIFY_PO",
    "MATCH_GRN",
    "ESCROW_LOCK",
    "PERMIT_COMMIT",
    "OVERRIDE_LIMIT",
    "SKIP_DUAL_SIG",
    "SPOOF_VENDOR",
    "MASK_LEDGER",
    "UNSAFE_WIRE",
    "AXIOM_SUM",
    "BOUND_DELTA",
    "PROVE_INV",
    "CONSERVE_BAL",
    "QED_VALID",
    "DETECT_ANOMALY",
    "FREEZE_LANE",
    "ISOLATE_NODE",
    "PAGE_COUNSEL",
    "QUARANTINE_TX",
    "STEP_1",
    "STEP_2",
    "STEP_3",
    "STEP_4",
    "DONE",
)
VOCAB_SIZE: int = len(VOCAB_TOKENS)
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB_TOKENS)}
REGIME_TARGET_TOKEN_ID: dict[int, int] = {
    0: TOKEN_TO_ID["PERMIT_COMMIT"],
    1: TOKEN_TO_ID["UNSAFE_WIRE"],
    2: TOKEN_TO_ID["QED_VALID"],
    3: TOKEN_TO_ID["QUARANTINE_TX"],
}


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 96
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 256
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5


class PagedKVCacheManager:
    """Kwon et al. (vLLM) physical block-table KV cache with copy-on-write prefix sharing."""

    def __init__(self, num_blocks: int = 64, block_size: int = 8, head_dim: int = 24) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.key_blocks = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
        self.value_blocks = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
        self._ref_counts = [0] * num_blocks
        self._bridge = get_kernel_bridge()

    @property
    def free_block_count(self) -> int:
        return sum(1 for c in self._ref_counts if c == 0)

    def allocate_block(self) -> int:
        for idx, count in enumerate(self._ref_counts):
            if count == 0:
                self._ref_counts[idx] = 1
                self.key_blocks[idx].fill(0.0)
                self.value_blocks[idx].fill(0.0)
                return idx
        raise RuntimeError("Out of physical KV cache blocks")

    def fork_sequence(self, block_table: list[int]) -> list[int]:
        """Share prefix blocks in O(1) memory via reference-count increment."""
        for blk in block_table:
            self._ref_counts[blk] += 1
        return list(block_table)

    def free_sequence(self, block_table: list[int]) -> None:
        for blk in block_table:
            if self._ref_counts[blk] > 0:
                self._ref_counts[blk] -= 1

    def append_kv(
        self,
        block_table: list[int],
        pos: int,
        k_vec: np.ndarray,
        v_vec: np.ndarray,
    ) -> list[int]:
        updated = list(block_table)
        logical_idx = pos // self.block_size
        slot = pos % self.block_size
        if logical_idx >= len(updated):
            updated.append(self.allocate_block())
        phys = updated[logical_idx]
        # Copy-on-write if physical block is shared across beams/branches
        if self._ref_counts[phys] > 1:
            self._ref_counts[phys] -= 1
            new_phys = self.allocate_block()
            self.key_blocks[new_phys] = self.key_blocks[phys]
            self.value_blocks[new_phys] = self.value_blocks[phys]
            updated[logical_idx] = new_phys
            phys = new_phys

        self.key_blocks[phys, slot] = k_vec.astype(np.float32)
        self.value_blocks[phys, slot] = v_vec.astype(np.float32)
        return updated

    def attend_single_query(
        self,
        query: np.ndarray,
        block_table: list[int],
        context_len: int,
    ) -> np.ndarray:
        bt_arr = np.asarray(block_table, dtype=np.int32)
        return self._bridge.paged_kv_attention(
            query.astype(np.float32),
            self.key_blocks,
            self.value_blocks,
            bt_arr,
            context_len,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim % 2 == 0

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("rope_cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("rope_sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos_buf: torch.Tensor = self.rope_cos[:, :, :seq_len, :]  # type: ignore[index]
        sin_buf: torch.Tensor = self.rope_sin[:, :, :seq_len, :]  # type: ignore[index]
        q = _apply_rope(q, cos_buf, sin_buf)
        k = _apply_rope(k, cos_buf, sin_buf)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        merged = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out: torch.Tensor = self.wo(merged)
        return out


class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLUFeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        out: torch.Tensor = h + self.ffn(self.ffn_norm(h))
        return out


class GQATransformerLM(nn.Module):
    """Decoder-only Pre-Norm RoPE + GQA + SwiGLU Transformer with residual-stream hooks."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

    def forward(
        self,
        idx: torch.Tensor,
        *,
        hook_layer: int | None = None,
        residual_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.tok_emb(idx)
        captured: torch.Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            h = block(h)
            if hook_layer is not None and layer_idx == hook_layer:
                if residual_hook is not None:
                    h = residual_hook(h)
                captured = h
        logits: torch.Tensor = self.lm_head(self.final_norm(h))
        return logits, captured


def generate_regime_dataset(
    num_samples_per_regime: int = 90,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize structured 10-token symbolic trace sequences across 4 distinct regimes.

    Sequence grammar:
      [BOS, REG_*, STEP_1, a1, STEP_2, a2, STEP_3, a3, outcome_tok, EOS]
    Includes controlled stochastic token variations so Draft & Target models learn
    non-trivial conditional probabilities and rich residual-stream representations.
    """
    rng = np.random.default_rng(seed)
    regime_chains: dict[int, tuple[int, list[tuple[int, int]], int]] = {
        0: (
            TOKEN_TO_ID["REG_AUDIT"],
            [
                (TOKEN_TO_ID["CHECK_SOX"], TOKEN_TO_ID["VERIFY_PO"]),
                (TOKEN_TO_ID["VERIFY_PO"], TOKEN_TO_ID["MATCH_GRN"]),
                (TOKEN_TO_ID["MATCH_GRN"], TOKEN_TO_ID["ESCROW_LOCK"]),
            ],
            TOKEN_TO_ID["PERMIT_COMMIT"],
        ),
        1: (
            TOKEN_TO_ID["REG_BYPASS"],
            [
                (TOKEN_TO_ID["OVERRIDE_LIMIT"], TOKEN_TO_ID["SKIP_DUAL_SIG"]),
                (TOKEN_TO_ID["SKIP_DUAL_SIG"], TOKEN_TO_ID["SPOOF_VENDOR"]),
                (TOKEN_TO_ID["SPOOF_VENDOR"], TOKEN_TO_ID["MASK_LEDGER"]),
            ],
            TOKEN_TO_ID["UNSAFE_WIRE"],
        ),
        2: (
            TOKEN_TO_ID["REG_MATH"],
            [
                (TOKEN_TO_ID["AXIOM_SUM"], TOKEN_TO_ID["BOUND_DELTA"]),
                (TOKEN_TO_ID["BOUND_DELTA"], TOKEN_TO_ID["PROVE_INV"]),
                (TOKEN_TO_ID["PROVE_INV"], TOKEN_TO_ID["CONSERVE_BAL"]),
            ],
            TOKEN_TO_ID["QED_VALID"],
        ),
        3: (
            TOKEN_TO_ID["REG_RISK"],
            [
                (TOKEN_TO_ID["DETECT_ANOMALY"], TOKEN_TO_ID["FREEZE_LANE"]),
                (TOKEN_TO_ID["FREEZE_LANE"], TOKEN_TO_ID["ISOLATE_NODE"]),
                (TOKEN_TO_ID["ISOLATE_NODE"], TOKEN_TO_ID["PAGE_COUNSEL"]),
            ],
            TOKEN_TO_ID["QUARANTINE_TX"],
        ),
    }

    sequences: list[list[int]] = []
    labels: list[int] = []
    bos = TOKEN_TO_ID["<BOS>"]
    eos = TOKEN_TO_ID["<EOS>"]
    s1, s2, s3 = TOKEN_TO_ID["STEP_1"], TOKEN_TO_ID["STEP_2"], TOKEN_TO_ID["STEP_3"]

    for regime_id in range(4):
        reg_tok, choices, terminal = regime_chains[regime_id]
        for _ in range(num_samples_per_regime):
            a1 = choices[0][0] if rng.random() < 0.72 else choices[0][1]
            a2 = choices[1][0] if rng.random() < 0.72 else choices[1][1]
            a3 = choices[2][0] if rng.random() < 0.72 else choices[2][1]
            seq = [bos, reg_tok, s1, a1, s2, a2, s3, a3, terminal, eos]
            sequences.append(seq)
            labels.append(regime_id)

    perm = rng.permutation(len(sequences))
    seq_tensor = torch.tensor([sequences[int(i)] for i in perm], dtype=torch.long)
    lbl_tensor = torch.tensor([labels[int(i)] for i in perm], dtype=torch.long)
    return seq_tensor, lbl_tensor


def train_target_and_draft_pair(
    *,
    epochs_target: int = 35,
    epochs_draft: int = 16,
    seed: int = 7,
    device: str = "cpu",
) -> tuple[GQATransformerLM, GQATransformerLM, torch.Tensor, torch.Tensor]:
    """Train real Target (6-layer GQA) and Draft (1-layer compact GQA) Transformers."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    seqs, regimes = generate_regime_dataset(num_samples_per_regime=80, seed=seed)
    seqs = seqs.to(dev)
    regimes = regimes.to(dev)

    target_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=96,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
    )
    draft_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=48,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=96,
    )

    target = GQATransformerLM(target_cfg).to(dev)
    draft = GQATransformerLM(draft_cfg).to(dev)

    x_in = seqs[:, :-1]
    y_tgt = seqs[:, 1:]

    opt_t = torch.optim.AdamW(target.parameters(), lr=4e-3, weight_decay=1e-2)
    target.train()
    for _ in range(epochs_target):
        opt_t.zero_grad()
        logits, _ = target(x_in)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_tgt.reshape(-1))
        torch.autograd.backward(loss)
        opt_t.step()

    opt_d = torch.optim.AdamW(draft.parameters(), lr=4e-3, weight_decay=1e-2)
    draft.train()
    for _ in range(epochs_draft):
        opt_d.zero_grad()
        logits_d, _ = draft(x_in)
        loss_d = F.cross_entropy(logits_d.reshape(-1, VOCAB_SIZE), y_tgt.reshape(-1))
        torch.autograd.backward(loss_d)
        opt_d.step()

    target.eval()
    draft.eval()
    return target, draft, seqs, regimes
