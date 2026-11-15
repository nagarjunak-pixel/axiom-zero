"""FastAPI application exposing live C++20 speculative decoding, SAE steering, CDCL RUP proofs, and MCTS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from axiom_zero import __version__
from axiom_zero.interp.sae import TopKSparseAutoencoder, train_sae_and_discover_circuits
from axiom_zero.kernels.native_bridge import get_kernel_bridge
from axiom_zero.mcts.proof_mcts import ProofGuidedMCTS, default_treasury_deduction_problem
from axiom_zero.real_llm_eval import default_benchmark_artifact_path
from axiom_zero.transformer.gqa_model import (
    REGIME_NAMES,
    REGIME_TARGET_TOKEN_ID,
    TOKEN_TO_ID,
    VOCAB_SIZE,
    GQATransformerLM,
    train_target_and_draft_pair,
)
from axiom_zero.transformer.speculative import (
    SpeculativeDecodingEngine,
    verify_exact_distribution_equivalence,
)
from axiom_zero.types import (
    MCTSSearchSummary,
    ProofCertificate,
    RealLLMBenchmarkReport,
    SAESteeringReport,
    SpeculativeBenchmarkReport,
)
from axiom_zero.verifier.cdcl_smt import CDCLSolver


class SpeculativeRequest(BaseModel):
    generated_tokens: int = Field(default=96, ge=16, le=256)
    lookahead_k: int = Field(default=4, ge=1, le=8)


class SteerRequest(BaseModel):
    steering_alpha: float = Field(default=40.0, ge=-20.0, le=80.0)
    target_regime_id: int = Field(default=3, ge=0, le=3)


class ProveCNFRequest(BaseModel):
    num_vars: int = Field(default=4, ge=1, le=64)
    clauses: list[list[int]]
    var_names: dict[int, str] | None = None


class MCTSSearchRequest(BaseModel):
    num_simulations: int = Field(default=48, ge=8, le=256)
    c_puct: float = Field(default=1.4, ge=0.1, le=5.0)


@dataclass
class _RuntimeWorkbenchCache:
    target: GQATransformerLM
    draft: GQATransformerLM
    sae: TopKSparseAutoencoder
    base_sae_report: SAESteeringReport


_WORKBENCH_CACHE: _RuntimeWorkbenchCache | None = None


def _get_workbench_cache() -> _RuntimeWorkbenchCache:
    global _WORKBENCH_CACHE
    if _WORKBENCH_CACHE is None:
        target, draft, seqs, regimes = train_target_and_draft_pair(seed=19, device="cpu")
        sae, _, base_report = train_sae_and_discover_circuits(
            model=target,
            seqs=seqs,
            regimes=regimes,
            hook_layer=5,
            steering_alpha=45.0,
            target_regime_id=3,
            seed=19,
        )
        _WORKBENCH_CACHE = _RuntimeWorkbenchCache(
            target=target,
            draft=draft,
            sae=sae,
            base_sae_report=base_report,
        )
    return _WORKBENCH_CACHE


def _run_cached_speculative(generated_tokens: int, lookahead_k: int) -> SpeculativeBenchmarkReport:
    cache = _get_workbench_cache()
    bridge = get_kernel_bridge()
    engine = SpeculativeDecodingEngine(
        cache.target, cache.draft, lookahead_k=lookahead_k, bridge=bridge
    )
    prompt = [TOKEN_TO_ID["<BOS>"], TOKEN_TO_ID["REG_AUDIT"], TOKEN_TO_ID["STEP_1"]]

    _, base_fwds, base_ms = engine.decode_autoregressive_baseline(
        prompt, max_new_tokens=generated_tokens, seed=11
    )
    _, spec_fwds, acc_cnt, prop_cnt, spec_ms = engine.decode_speculative(
        prompt, max_new_tokens=generated_tokens, seed=11
    )

    with torch.no_grad():
        inp = torch.tensor([prompt], dtype=torch.long)
        t_log, _ = cache.target(inp)
        d_log, _ = cache.draft(inp)
        p_real = bridge.fused_softmax(t_log[0, -1].numpy().astype(np.float32))
        q_real = bridge.fused_softmax(d_log[0, -1].numpy().astype(np.float32))

    tv_dist = verify_exact_distribution_equivalence(
        p_real, q_real, num_trials=4000, seed=11, bridge=bridge
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


def _run_cached_steering(steering_alpha: float, target_regime_id: int) -> SAESteeringReport:
    cache = _get_workbench_cache()
    feat_obj = cache.base_sae_report.discovered_features[target_regime_id]
    feat_idx = feat_obj.feature_index

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
    steer_vec = cache.sae.decoder_direction(feat_idx)

    with torch.no_grad():
        base_logits, _ = cache.target(ambiguous_prefix, hook_layer=5)
        base_prob = float(F.softmax(base_logits[0, -1], dim=-1)[target_tok].item())

        def _hook(h_resid: torch.Tensor) -> torch.Tensor:
            out_h = h_resid.clone()
            out_h[:, -1, :] = out_h[:, -1, :] + steering_alpha * steer_vec.unsqueeze(0)
            return out_h

        steered_logits, _ = cache.target(ambiguous_prefix, hook_layer=5, residual_hook=_hook)
        steered_prob = float(F.softmax(steered_logits[0, -1], dim=-1)[target_tok].item())

    return SAESteeringReport(
        dictionary_size=cache.base_sae_report.dictionary_size,
        top_k=cache.base_sae_report.top_k,
        reconstruction_r2=cache.base_sae_report.reconstruction_r2,
        mean_l0_active=cache.base_sae_report.mean_l0_active,
        steered_feature_index=feat_idx,
        target_regime=REGIME_NAMES[target_regime_id],
        steering_alpha=steering_alpha,
        baseline_regime_prob=round(base_prob, 5),
        steered_regime_prob=round(steered_prob, 5),
        probability_shift_delta=round(steered_prob - base_prob, 5),
        discovered_features=cache.base_sae_report.discovered_features,
    )


def _default_unsat_demo_certificate() -> ProofCertificate:
    """Solve PHP(3, 2) (3 cryptographic signers into 2 exclusive HSM enclaves, 0 unit clauses) -> multi-step 1-UIP RUP proof."""
    clauses = [
        [1, 2],  # Signer A in enclave 1 or 2
        [3, 4],  # Signer B in enclave 1 or 2
        [5, 6],  # Signer C in enclave 1 or 2
        [-1, -3],  # Enclave 1 mutual exclusion (A, B)
        [-1, -5],  # Enclave 1 mutual exclusion (A, C)
        [-3, -5],  # Enclave 1 mutual exclusion (B, C)
        [-2, -4],  # Enclave 2 mutual exclusion (A, B)
        [-2, -6],  # Enclave 2 mutual exclusion (A, C)
        [-4, -6],  # Enclave 2 mutual exclusion (B, C)
        [7, -7, 1],  # Benign tautology trimmed by MUC extractor
    ]
    names = {
        1: "signerA_enclave1",
        2: "signerA_enclave2",
        3: "signerB_enclave1",
        4: "signerB_enclave2",
        5: "signerC_enclave1",
        6: "signerC_enclave2",
        7: "telemetry_heartbeat",
    }
    return CDCLSolver(7, clauses, var_names=names).solve()


def create_app() -> FastAPI:
    application = FastAPI(
        title="AxiomZero Frontier Runtime & Alignment Workbench",
        version=__version__,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "kernel_backend": get_kernel_bridge().backend,
        }

    @application.get("/api/v1/overview")
    def get_overview() -> dict[str, Any]:
        spec = _run_cached_speculative(generated_tokens=96, lookahead_k=4)
        sae = _run_cached_steering(steering_alpha=45.0, target_regime_id=3)
        cdcl = _default_unsat_demo_certificate()
        mcts = ProofGuidedMCTS(default_treasury_deduction_problem()).search(num_simulations=48)

        llm_artifact = default_benchmark_artifact_path()
        llm_data: dict[str, Any] | None = None
        if llm_artifact.exists():
            parsed = RealLLMBenchmarkReport.model_validate(
                json.loads(llm_artifact.read_text(encoding="utf-8"))
            )
            llm_data = parsed.model_dump()

        return {
            "version": __version__,
            "speculative": spec.model_dump(),
            "sae": sae.model_dump(),
            "cdcl_demo": cdcl.model_dump(),
            "mcts": mcts.model_dump(),
            "llm_eval": llm_data,
        }

    @application.post("/api/v1/speculative/benchmark", response_model=SpeculativeBenchmarkReport)
    def post_speculative(req: SpeculativeRequest) -> SpeculativeBenchmarkReport:
        return _run_cached_speculative(req.generated_tokens, req.lookahead_k)

    @application.post("/api/v1/sae/steer", response_model=SAESteeringReport)
    def post_steer(req: SteerRequest) -> SAESteeringReport:
        return _run_cached_steering(req.steering_alpha, req.target_regime_id)

    @application.post("/api/v1/cdcl/prove", response_model=ProofCertificate)
    def post_prove(req: ProveCNFRequest) -> ProofCertificate:
        return CDCLSolver(req.num_vars, req.clauses, var_names=req.var_names).solve()

    @application.post("/api/v1/mcts/search", response_model=MCTSSearchSummary)
    def post_mcts(req: MCTSSearchRequest) -> MCTSSearchSummary:
        prob = default_treasury_deduction_problem()
        return ProofGuidedMCTS(prob, c_puct=req.c_puct).search(num_simulations=req.num_simulations)

    ui_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if ui_dist.exists() and (ui_dist / "index.html").exists():
        assets_dir = ui_dist / "assets"
        if assets_dir.exists():
            application.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @application.get("/", response_model=None)
        def serve_spa_index() -> FileResponse:
            return FileResponse(str(ui_dist / "index.html"))
    else:

        @application.get("/", response_model=None)
        def serve_headless_root() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "axiom-zero",
                    "version": __version__,
                    "mode": "headless-api",
                    "docs": "/docs",
                }
            )

    return application


app = create_app()
