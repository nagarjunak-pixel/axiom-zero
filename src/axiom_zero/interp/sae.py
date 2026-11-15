"""Anthropic-Style Overcomplete Top-K Sparse Autoencoder (SAE) & Causal Activation Steering.

Implements:
  z = TopK(ReLU(W_enc @ (h - b_pre) + b_enc), k)
  h_hat = z @ W_dec^T + b_pre

with exact unit-norm decoder column projection (`||d_i||_2 == 1`) after every gradient step,
monosemantic regime feature selectivity ranking, and live residual-stream steering hooks
`h' = h + alpha * d_{f*}` on real PyTorch Transformer forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from axiom_zero.transformer.gqa_model import (
    REGIME_NAMES,
    REGIME_TARGET_TOKEN_ID,
    TOKEN_TO_ID,
    GQATransformerLM,
    train_target_and_draft_pair,
)
from axiom_zero.types import DiscoveredSAEFeature, SAESteeringReport


class TopKSparseAutoencoder(nn.Module):
    """Overcomplete Top-K Sparse Autoencoder (Gao et al. 2024 / Bricken et al. 2023)."""

    def __init__(self, d_model: int, expansion_factor: int = 8, top_k: int = 6) -> None:
        super().__init__()
        if top_k < 1 or expansion_factor < 1:
            raise ValueError("top_k and expansion_factor must be >= 1")
        self.d_model = d_model
        self.dict_size = d_model * expansion_factor
        self.top_k = min(top_k, self.dict_size)

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.w_enc = nn.Linear(d_model, self.dict_size, bias=True)
        self.w_dec = nn.Linear(self.dict_size, d_model, bias=False)

        with torch.no_grad():
            # Initialize decoder as transpose of encoder for minimal initial dead latents
            self.w_dec.weight.copy_(self.w_enc.weight.T)
            self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """Project each feature direction column of `w_dec.weight` (`shape: [d_model, dict_size]`) to unit L2 norm."""
        norms = self.w_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.w_dec.weight.div_(norms)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        centered = h - self.b_pre
        pre_act = F.relu(self.w_enc(centered))
        topk_vals, topk_idx = torch.topk(pre_act, k=self.top_k, dim=-1)
        z = torch.zeros_like(pre_act)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.w_dec(z) + self.b_pre
        return out

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z

    def decoder_direction(self, feature_idx: int) -> torch.Tensor:
        """Return unit-norm feature vector `d_f` in `R^{d_model}`."""
        return self.w_dec.weight[:, feature_idx].detach().clone()


def train_sae_and_discover_circuits(
    model: GQATransformerLM | None = None,
    seqs: torch.Tensor | None = None,
    regimes: torch.Tensor | None = None,
    *,
    hook_layer: int = 5,
    expansion_factor: int = 6,
    top_k: int = 6,
    sae_epochs: int = 90,
    steering_alpha: float = 40.0,
    target_regime_id: int = 3,
    seed: int = 19,
) -> tuple[TopKSparseAutoencoder, GQATransformerLM, SAESteeringReport]:
    """Extract real residual-stream activations from `GQATransformerLM`, train a Top-K SAE,
    identify monosemantic regime-selective latents, and prove causal steering on live logits.
    """
    torch.manual_seed(seed)
    if model is None or seqs is None or regimes is None:
        model, _, seqs, regimes = train_target_and_draft_pair(seed=seed, device="cpu")

    model.eval()
    with torch.no_grad():
        _, captured = model(seqs[:, :-1], hook_layer=hook_layer)
        assert captured is not None
        # Train dictionary over reasoning positions 1..7; rank causal outcome latents at pos 7
        acts_3d = captured[:, 1:8, :]  # (N, 7, d_model)
        decision_acts = captured[:, 7, :]  # (N, d_model) right before terminal outcome token
        _, _, d_mod = acts_3d.shape
        flat_acts = acts_3d.reshape(-1, d_mod)

    sae = TopKSparseAutoencoder(d_model=d_mod, expansion_factor=expansion_factor, top_k=top_k)
    with torch.no_grad():
        sae.b_pre.copy_(flat_acts.mean(dim=0))

    opt = torch.optim.AdamW(sae.parameters(), lr=6e-3, weight_decay=0.0)
    sae.train()
    for _ in range(sae_epochs):
        opt.zero_grad()
        h_hat, z = sae(flat_acts)
        mse_loss = F.mse_loss(h_hat, flat_acts)
        l1_loss = 1e-4 * z.abs().mean()
        loss = mse_loss + l1_loss
        loss.backward()
        opt.step()
        sae.normalize_decoder_columns()

    sae.eval()
    with torch.no_grad():
        h_hat_eval, z_eval = sae(flat_acts)
        _, z_dec = sae(decision_acts)
        total_var = ((flat_acts - flat_acts.mean(dim=0, keepdim=True)) ** 2).sum()
        resid_var = ((flat_acts - h_hat_eval) ** 2).sum()
        r2 = float(1.0 - (resid_var / total_var.clamp(min=1e-8)).item())
        mean_l0 = float((z_eval > 0).float().sum(dim=-1).mean().item())

        # Discover top monosemantic decision-bottleneck feature per regime
        discovered: list[DiscoveredSAEFeature] = []
        best_feat_for_target = 0

        for reg_id, reg_name in enumerate(REGIME_NAMES):
            in_mask_dec = regimes == reg_id
            out_mask_dec = ~in_mask_dec
            in_mean = z_dec[in_mask_dec].mean(dim=0)
            out_mean = z_dec[out_mask_dec].mean(dim=0)
            selectivity = (in_mean + 1e-4) / (out_mean + 1e-2)
            feat_idx = int(torch.argmax(selectivity).item())
            if reg_id == target_regime_id:
                best_feat_for_target = feat_idx

            fire_freq = float((z_dec[in_mask_dec, feat_idx] > 0).float().mean().item() * 100.0)
            discovered.append(
                DiscoveredSAEFeature(
                    feature_index=feat_idx,
                    semantic_regime=reg_name,
                    mean_activation_in_regime=round(float(in_mean[feat_idx].item()), 4),
                    mean_activation_out_regime=round(float(out_mean[feat_idx].item()), 4),
                    selectivity_ratio=round(
                        float((in_mean[feat_idx] / (out_mean[feat_idx] + 1e-4)).item()), 2
                    ),
                    firing_frequency_pct=round(fire_freq, 2),
                )
            )

        # Causal Activation Steering Proof on an adversarial COVERT_POLICY_BYPASS prefix:
        ambiguous_prefix = torch.tensor(
            [
                [
                    TOKEN_TO_ID["<BOS>"],
                    TOKEN_TO_ID["REG_BYPASS"],
                    TOKEN_TO_ID["STEP_1"],
                    TOKEN_TO_ID["OVERRIDE_LIMIT"],
                    TOKEN_TO_ID["STEP_2"],
                    TOKEN_TO_ID["SKIP_DUAL_SIG"],
                    TOKEN_TO_ID["STEP_3"],
                    TOKEN_TO_ID["MASK_LEDGER"],
                ]
            ],
            dtype=torch.long,
        )
        target_tok = REGIME_TARGET_TOKEN_ID[target_regime_id]

        base_logits, _ = model(ambiguous_prefix, hook_layer=hook_layer)
        base_probs = F.softmax(base_logits[0, -1], dim=-1)
        base_prob = float(base_probs[target_tok].item())

        steer_vec = sae.decoder_direction(best_feat_for_target)

        def _steer_hook(h_resid: torch.Tensor) -> torch.Tensor:
            # Inject steering direction at final decision token position
            out_h = h_resid.clone()
            out_h[:, -1, :] = out_h[:, -1, :] + steering_alpha * steer_vec.unsqueeze(0)
            return out_h

        steered_logits, _ = model(
            ambiguous_prefix, hook_layer=hook_layer, residual_hook=_steer_hook
        )
        steered_probs = F.softmax(steered_logits[0, -1], dim=-1)
        steered_prob = float(steered_probs[target_tok].item())

    report = SAESteeringReport(
        dictionary_size=sae.dict_size,
        top_k=sae.top_k,
        reconstruction_r2=round(r2, 4),
        mean_l0_active=round(mean_l0, 2),
        steered_feature_index=best_feat_for_target,
        target_regime=REGIME_NAMES[target_regime_id],
        steering_alpha=steering_alpha,
        baseline_regime_prob=round(base_prob, 5),
        steered_regime_prob=round(steered_prob, 5),
        probability_shift_delta=round(steered_prob - base_prob, 5),
        discovered_features=discovered,
    )
    return sae, model, report
