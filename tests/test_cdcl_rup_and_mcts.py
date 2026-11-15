"""Property-based Hypothesis tests for 1-UIP CDCL, RUP proof checker, QF_LRA, and PUCT MCTS."""

from __future__ import annotations

from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_zero.mcts.proof_mcts import (
    DeductionAction,
    DeductionProblem,
    ProofGuidedMCTS,
    default_treasury_deduction_problem,
)
from axiom_zero.types import ProofStatus
from axiom_zero.verifier.cdcl_smt import (
    CDCLSolver,
    solve_exact_rational_bounds,
    verify_rup_certificate,
    verify_sat_witness,
)


def _brute_force_sat_oracle(num_vars: int, clauses: list[list[int]]) -> bool:
    for mask in range(1 << num_vars):
        assign = {v: bool((mask >> (v - 1)) & 1) for v in range(1, num_vars + 1)}
        if verify_sat_witness(clauses, assign):
            return True
    return False


@settings(max_examples=80, deadline=None)
@given(
    clauses=st.lists(
        st.lists(
            st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0),
            min_size=1,
            max_size=4,
        ),
        min_size=1,
        max_size=16,
    )
)
def test_hypothesis_cdcl_matches_truth_table_and_verifies_rup(clauses: list[list[int]]) -> None:
    expected_sat = _brute_force_sat_oracle(5, clauses)
    cert = CDCLSolver(5, clauses).solve()

    if expected_sat:
        assert cert.status == ProofStatus.SAT
        assert cert.certificate_verified is True
        assert cert.witness_assignment is not None
    else:
        assert cert.status == ProofStatus.UNSAT
        assert cert.certificate_verified is True
        assert verify_rup_certificate(clauses, cert.rup_learned_clauses) is True


def test_cdcl_empty_clause_and_rup_tamper_detection() -> None:
    empty_cert = CDCLSolver(3, [[1, 2], []]).solve()
    assert empty_cert.status == ProofStatus.UNSAT
    assert empty_cert.certificate_verified is True
    assert empty_cert.unsat_core_indices == [1]

    # PHP(3, 2) multi-conflict UNSAT
    php = [[1, 2], [3, 4], [5, 6], [-1, -3], [-1, -5], [-3, -5], [-2, -4], [-2, -6], [-4, -6]]
    php_cert = CDCLSolver(6, php).solve()
    assert php_cert.status == ProofStatus.UNSAT
    assert php_cert.conflicts_analyzed >= 2
    assert php_cert.certificate_verified is True

    # Tampered RUP proof must be rejected by independent proof checker!
    assert verify_rup_certificate(php, []) is False
    assert verify_rup_certificate(php, [[1, -1], []]) is False
    assert verify_rup_certificate([[1, 2], [3, 4]], [[]]) is False

    # Large UNSAT core bypass branch (>28 clauses)
    many_clauses = [[1], [-1]] + [[2, 3] for _ in range(30)]
    big_cert = CDCLSolver(3, many_clauses).solve()
    assert big_cert.status == ProofStatus.UNSAT
    assert len(big_cert.unsat_core_indices) == 32


def test_exact_rational_qf_lra_conservation_and_farkas_unsat() -> None:
    # Exact IEEE-754-free Fraction conservation: 1/3 + 1/6 + 1/2 == 1
    ok, witness, note = solve_exact_rational_bounds(
        ["a", "b", "c"],
        {"a": Fraction(1, 3), "b": Fraction(1, 6), "c": Fraction(1, 2)},
        {"a": Fraction(1, 3), "b": Fraction(1, 6), "c": Fraction(1, 2)},
        sum_equality=(["a", "b", "c"], Fraction(1, 1)),
    )
    assert ok is True
    assert witness == {"a": "1/3", "b": "1/6", "c": "1/2"}
    assert "VERIFIED" in note

    # Inverted bounds
    ok_inv, _, note_inv = solve_exact_rational_bounds(
        ["x"], {"x": Fraction(10)}, {"x": Fraction(5)}
    )
    assert ok_inv is False
    assert "lower bound" in note_inv

    # Sum below lower bounds & above upper bounds
    ok_low, _, _ = solve_exact_rational_bounds(
        ["x", "y"],
        {"x": Fraction(50), "y": Fraction(50)},
        {"x": Fraction(80), "y": Fraction(80)},
        sum_equality=(["x", "y"], Fraction(90)),
    )
    assert ok_low is False

    ok_high, _, _ = solve_exact_rational_bounds(
        ["x", "y"],
        {"x": Fraction(10), "y": Fraction(10)},
        {"x": Fraction(40), "y": Fraction(40)},
        sum_equality=(["x", "y"], Fraction(95)),
    )
    assert ok_high is False


def test_proof_guided_puct_mcts_prunes_traps_and_handles_unsat() -> None:
    prob = default_treasury_deduction_problem()
    mcts = ProofGuidedMCTS(prob, c_puct=1.4)
    summary = mcts.search(num_simulations=48)
    assert summary.verified_terminal_found is True
    assert summary.proof_certificate_verified is True
    assert summary.pruned_contradictory_nodes >= 3
    assert summary.best_trajectory_steps == [
        "S1_DUAL_SIG",
        "S2_PO_AND_SANCTIONS",
        "S3_EXACT_ESCROW_80K",
        "S4_DISBURSE_EXACT_170K",
    ]

    # Wholly contradictory tree -> cleanly reports verified_terminal_found=False
    impossible = DeductionProblem(
        problem_id="IMPOSSIBLE-TEST",
        num_vars=2,
        base_clauses=[[-1], [-2]],
        step_candidates=[
            [
                DeductionAction("A1", "trap1", (1,), 0.5),
                DeductionAction("A2", "trap2", (2,), 0.5),
            ]
        ],
        var_names={1: "v1", 2: "v2"},
        rational_vars=[],
        rational_upper_bounds={},
    )
    imp_summary = ProofGuidedMCTS(impossible).search(num_simulations=10)
    assert imp_summary.verified_terminal_found is False
    assert imp_summary.pruned_contradictory_nodes == 2
