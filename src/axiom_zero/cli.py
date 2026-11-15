"""Rich CLI entrypoint (`axiom`) for AxiomZero benchmarks, formal proofs, and server."""

from __future__ import annotations

import json

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom_zero.interp.sae import train_sae_and_discover_circuits
from axiom_zero.mcts.proof_mcts import ProofGuidedMCTS, default_treasury_deduction_problem
from axiom_zero.real_llm_eval import run_real_ollama_benchmark
from axiom_zero.transformer.speculative import run_speculative_benchmark
from axiom_zero.verifier.cdcl_smt import CDCLSolver

app = typer.Typer(
    name="axiom",
    help="AxiomZero: Verified Frontier Transformer Runtime, Top-K SAE Steering & 1-UIP CDCL Verifier",
    no_args_is_help=True,
)
console = Console()


@app.command("benchmark-speculative")
def cmd_speculative(
    tokens: int = typer.Option(96, "--tokens", help="Number of tokens to decode"),
    lookahead: int = typer.Option(4, "--lookahead", help="Draft model lookahead K"),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON report"),
) -> None:
    """Train Target & Draft GQA Transformers and benchmark C++20 lossless speculative decoding."""
    rep = run_speculative_benchmark(generated_tokens=tokens, lookahead_k=lookahead)
    if as_json:
        console.print_json(json.dumps(rep.model_dump()))
        return

    tbl = Table(title="AxiomZero C++20 Lossless Speculative Decoding Benchmark")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Measured Value", style="bold green")
    tbl.add_row("Kernel Backend", rep.kernel_backend)
    tbl.add_row("Generated Tokens", str(rep.generated_tokens))
    tbl.add_row("Baseline Target Forward Passes", str(rep.baseline_target_forwards))
    tbl.add_row("Speculative Target Forward Passes", str(rep.speculative_target_forwards))
    tbl.add_row("Target Forward-Pass Reduction", f"{rep.forward_pass_reduction_x:.2f}x")
    tbl.add_row("Draft Acceptance Rate", f"{rep.acceptance_rate_pct:.1f}%")
    tbl.add_row("Empirical Total Variation Distance (TV)", f"{rep.empirical_tv_distance:.5f}")
    tbl.add_row("Wall-Clock Speedup", f"{rep.wall_clock_speedup_x:.2f}x")
    console.print(tbl)


@app.command("steer-sae")
def cmd_steer_sae(
    alpha: float = typer.Option(40.0, "--alpha", help="Residual stream steering coefficient"),
    regime_id: int = typer.Option(
        3, "--regime", help="Target regime ID (0..3, default 3=RISK_QUARANTINE)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON report"),
) -> None:
    """Train Overcomplete Top-K SAE on GQA residual stream and run causal activation steering."""
    _, _, rep = train_sae_and_discover_circuits(steering_alpha=alpha, target_regime_id=regime_id)
    if as_json:
        console.print_json(json.dumps(rep.model_dump()))
        return

    tbl = Table(title="Anthropic-Style Top-K SAE Monosemantic Feature Discovery & Causal Steering")
    tbl.add_column("Regime", style="cyan")
    tbl.add_column("Latent #", justify="right")
    tbl.add_column("In-Regime Mean", justify="right")
    tbl.add_column("Out-Regime Mean", justify="right")
    tbl.add_column("Selectivity", justify="right", style="bold yellow")
    tbl.add_column("Firing %", justify="right")
    for feat in rep.discovered_features:
        tbl.add_row(
            feat.semantic_regime,
            f"f_{feat.feature_index}",
            f"{feat.mean_activation_in_regime:.3f}",
            f"{feat.mean_activation_out_regime:.3f}",
            f"{feat.selectivity_ratio:.1f}x",
            f"{feat.firing_frequency_pct:.1f}%",
        )
    console.print(tbl)
    console.print(
        Panel(
            f"R^2 = {rep.reconstruction_r2:.4f} | Mean L0 = {rep.mean_l0_active:.2f} (Top-K={rep.top_k}) | "
            f"Steered f_{rep.steered_feature_index} ({rep.target_regime}, alpha={rep.steering_alpha}): "
            f"P(target) {rep.baseline_regime_prob:.4f} -> {rep.steered_regime_prob:.4f} "
            f"(Delta = {rep.probability_shift_delta:+.4f})",
            title="Causal Intervention Summary",
            border_style="green",
        )
    )


@app.command("prove")
def cmd_prove(
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON certificate"),
) -> None:
    """Run 1-UIP CDCL RUP proof certification and Proof-Guided PUCT MCTS deduction."""
    clauses = [
        [1, 2],
        [3, 4],
        [5, 6],
        [-1, -3],
        [-1, -5],
        [-3, -5],
        [-2, -4],
        [-2, -6],
        [-4, -6],
    ]
    cert = CDCLSolver(6, clauses).solve()
    mcts = ProofGuidedMCTS(default_treasury_deduction_problem()).search(num_simulations=48)

    if as_json:
        console.print_json(
            json.dumps({"cdcl_certificate": cert.model_dump(), "mcts": mcts.model_dump()})
        )
        return

    console.print(
        Panel(
            f"CDCL Status: [bold]{cert.status}[/bold] | Conflicts: {cert.conflicts_analyzed} | "
            f"RUP Proof Steps: {len(cert.rup_learned_clauses)} | "
            f"Independent RUP Verified: [bold green]{cert.certificate_verified}[/bold green]\n"
            f"PUCT MCTS ({mcts.problem_id}): Pruned {mcts.pruned_contradictory_nodes} contradictory branches | "
            f"Trajectory: {' -> '.join(mcts.best_trajectory_steps)} | "
            f"Certificate Verified: [bold green]{mcts.proof_certificate_verified}[/bold green]",
            title="1-UIP CDCL + Proof-Guided PUCT MCTS",
            border_style="cyan",
        )
    )


@app.command("eval-ollama")
def cmd_eval_ollama(
    model: str = typer.Option("qwen3.5:4b", "--model", help="Local Ollama model tag"),
    offline_replay: bool = typer.Option(
        True,
        "--offline-replay/--live-only",
        help="Allow loading signed artifact when Ollama is offline",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON benchmark report"),
) -> None:
    """Evaluate real local Ollama LLM (`Pass@1` vs. AxiomZero CEGIS CDCL + QF_LRA repair)."""
    rep = run_real_ollama_benchmark(model=model, allow_offline_replay=offline_replay)
    if as_json:
        console.print_json(json.dumps(rep.model_dump()))
        return

    tbl = Table(title=f"Zero-Mock Real LLM Formal Benchmark ({rep.model_name})")
    tbl.add_column("Task ID", style="cyan")
    tbl.add_column("Raw Pass@1")
    tbl.add_column("AxiomZero CEGIS Verified")
    tbl.add_column("Attempts", justify="right")
    tbl.add_column("Latency (ms)", justify="right")
    for o in rep.outcomes:
        tbl.add_row(
            o.task_id,
            "[green]PASS[/green]" if o.raw_model_passed else "[red]VIOLATION[/red]",
            "[bold green]VERIFIED[/bold green]"
            if o.verified_repair_passed
            else "[yellow]BLOCKED[/yellow]",
            str(o.attempts_used),
            f"{o.latency_ms:.0f}",
        )
    console.print(tbl)
    console.print(
        Panel(
            f"Raw Pass@1: {rep.raw_pass_at_1_count}/{rep.total_tasks} ({rep.raw_pass_at_1_pct:.1f}%) | "
            f"Raw Formal Violations: {rep.raw_formal_violation_pct:.1f}%\n"
            f"AxiomZero CEGIS Verified: {rep.axiom_verified_pass_count}/{rep.total_tasks} "
            f"({rep.axiom_verified_pass_pct:.1f}%) | "
            f"Unsafe Unverified Leakage: {rep.axiom_formal_violation_pct:.1f}%",
            title="Real LLM Summary",
            border_style="green",
        )
    )


@app.command("demo")
def cmd_demo() -> None:
    """Run all 4 AxiomZero frontier pillars end-to-end (C++20 Speculative, Top-K SAE, CDCL/MCTS, Real LLM)."""
    cmd_speculative(tokens=96, lookahead=4, as_json=False)
    cmd_steer_sae(alpha=40.0, regime_id=3, as_json=False)
    cmd_prove(as_json=False)
    cmd_eval_ollama(model="qwen3.5:4b", offline_replay=True, as_json=False)


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8030, "--port"),
) -> None:
    """Launch the AxiomZero FastAPI + Frontier Research Workbench UI server."""
    uvicorn.run("axiom_zero.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
