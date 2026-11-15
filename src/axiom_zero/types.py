"""Typed domain models for AxiomZero's inference, SAE interpretability, and CDCL proof cores."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProofStatus(StrEnum):
    SAT = "SAT"
    UNSAT = "UNSAT"


class SpeculativeBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_backend: str
    device: str
    vocab_size: int
    draft_lookahead_k: int
    generated_tokens: int
    baseline_target_forwards: int
    speculative_target_forwards: int
    forward_pass_reduction_x: float
    mean_accepted_draft_tokens: float
    acceptance_rate_pct: float
    empirical_tv_distance: float = Field(ge=0.0, le=1.0)
    baseline_latency_ms: float
    speculative_latency_ms: float
    wall_clock_speedup_x: float


class DiscoveredSAEFeature(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_index: int
    semantic_regime: str
    mean_activation_in_regime: float
    mean_activation_out_regime: float
    selectivity_ratio: float
    firing_frequency_pct: float


class SAESteeringReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dictionary_size: int
    top_k: int
    reconstruction_r2: float
    mean_l0_active: float
    steered_feature_index: int
    target_regime: str
    steering_alpha: float
    baseline_regime_prob: float
    steered_regime_prob: float
    probability_shift_delta: float
    discovered_features: list[DiscoveredSAEFeature]


class ResolutionStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_index: int
    learned_clause: list[int]
    decision_level: int


class ProofCertificate(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ProofStatus
    num_variables: int
    num_clauses: int
    conflicts_analyzed: int
    propagations: int
    witness_assignment: dict[str, bool] | None = None
    linear_witness: dict[str, str] | None = None
    rup_learned_clauses: list[list[int]] = Field(default_factory=list)
    unsat_core_indices: list[int] = Field(default_factory=list)
    certificate_verified: bool
    solve_time_ms: float


class MCTSSearchSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_id: str
    simulations_run: int
    pruned_contradictory_nodes: int
    verified_terminal_found: bool
    best_trajectory_steps: list[str]
    best_value: float
    proof_certificate_verified: bool


class RealLLMTaskOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    category: str
    raw_model_passed: bool
    raw_violation_reason: str | None
    verified_repair_passed: bool
    attempts_used: int
    latency_ms: float


class RealLLMBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    total_tasks: int
    raw_pass_at_1_count: int
    raw_pass_at_1_pct: float
    raw_formal_violation_pct: float
    axiom_verified_pass_count: int
    axiom_verified_pass_pct: float
    axiom_formal_violation_pct: float
    mean_task_latency_ms: float
    outcomes: list[RealLLMTaskOutcome]
