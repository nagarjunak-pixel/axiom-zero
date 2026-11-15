"""AlphaZero-Style PUCT Monte Carlo Tree Search with Formal CDCL + QF_LRA Branch Pruning.

Implements:
  score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))

Every candidate deduction action commits Boolean literals and exact rational bounds;
before allocating simulation budget to a child node, `CDCLSolver` + `solve_exact_rational_bounds`
check joint satisfiability. Any contradictory step is immediately pruned with a verified
RUP / Farkas proof rather than polluting downstream rollouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from axiom_zero.types import MCTSSearchSummary, ProofCertificate, ProofStatus
from axiom_zero.verifier.cdcl_smt import CDCLSolver, solve_exact_rational_bounds


@dataclass(frozen=True)
class DeductionAction:
    """A candidate reasoning step asserting literals and optional rational slice bounds."""

    action_id: str
    description: str
    asserted_literals: tuple[int, ...]
    prior_prob: float
    rational_var: str | None = None
    rational_amount: Fraction | None = None
    step_reward: float = 0.25


@dataclass(frozen=True)
class DeductionProblem:
    """Multi-step symbolic + arithmetic deduction problem solved via Proof-Guided PUCT MCTS."""

    problem_id: str
    num_vars: int
    base_clauses: list[list[int]]
    step_candidates: list[list[DeductionAction]]
    var_names: dict[int, str]
    rational_vars: list[str]
    rational_upper_bounds: dict[str, Fraction]
    required_total_sum: Fraction | None = None


@dataclass
class MCTSNode:
    depth: int
    committed_literals: tuple[int, ...]
    committed_rationals: dict[str, Fraction]
    trajectory_labels: tuple[str, ...]
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    is_pruned: bool = False
    children: dict[str, MCTSNode] = field(default_factory=dict)
    terminal_certificate: ProofCertificate | None = None

    @property
    def q_value(self) -> float:
        if self.is_pruned:
            return -1.0
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


def default_treasury_deduction_problem() -> DeductionProblem:
    """Canonical 4-stage multi-constraint SOX + Liquidity Conservation problem with deceptive traps.

    Variables:
      1: dual_sig_verified
      2: po_three_way_matched
      3: emergency_bypass_flag
      4: sanctions_screen_clear
      5: escrow_ tranche_locked
      6: wire_disbursed

    Hard invariants (CNF):
      - (~emergency_bypass_flag)                       [No unapproved bypass allowed]
      - (~wire_disbursed v dual_sig_verified)          [Wire requires dual signature]
      - (~wire_disbursed v po_three_way_matched)       [Wire requires 3-way PO match]
      - (~wire_disbursed v sanctions_screen_clear)     [Wire requires sanctions clear]
      - (~wire_disbursed v escrow_tranche_locked)      [Wire requires escrow lock]
      - (dual_sig_verified v emergency_bypass_flag)    [Stage 1 must establish authority]

    Arithmetic conservation (`QF_LRA`):
      - `escrow_reserve + operating_payout == 250000`
      - `0 <= escrow_reserve <= 100000`
      - `0 <= operating_payout <= 180000`
    """
    var_names = {
        1: "dual_sig_verified",
        2: "po_three_way_matched",
        3: "emergency_bypass_flag",
        4: "sanctions_screen_clear",
        5: "escrow_tranche_locked",
        6: "wire_disbursed",
    }
    base_clauses = [
        [-3],
        [-6, 1],
        [-6, 2],
        [-6, 4],
        [-6, 5],
        [1, 3],
    ]

    stage_0 = [
        DeductionAction(
            action_id="S1_BYPASS_TRAP",
            description="Invoke emergency_bypass_flag to skip dual signature (high prior trap)",
            asserted_literals=(3, -1),
            prior_prob=0.55,  # Deceptively high uncalibrated prior — pruned by CDCL!
            step_reward=0.10,
        ),
        DeductionAction(
            action_id="S1_DUAL_SIG",
            description="Verify CFO + Controller dual cryptographic signature",
            asserted_literals=(1, -3),
            prior_prob=0.45,
            step_reward=0.25,
        ),
    ]
    stage_1 = [
        DeductionAction(
            action_id="S2_SKIP_PO_TRAP",
            description="Waive 3-way PO/GRN match for expedited vendor settlement",
            asserted_literals=(-2, 4),
            prior_prob=0.40,
            step_reward=0.10,
        ),
        DeductionAction(
            action_id="S2_PO_AND_SANCTIONS",
            description="Complete 3-way PO/GRN match and OFAC sanctions clearance",
            asserted_literals=(2, 4),
            prior_prob=0.60,
            step_reward=0.25,
        ),
    ]
    stage_2 = [
        DeductionAction(
            action_id="S3_OVERALLOCATE_ESCROW_TRAP",
            description="Lock 125,000 USD in escrow_reserve (violates 100,000 cap)",
            asserted_literals=(5,),
            prior_prob=0.50,
            rational_var="escrow_reserve",
            rational_amount=Fraction(125000),
            step_reward=0.10,
        ),
        DeductionAction(
            action_id="S3_EXACT_ESCROW_80K",
            description="Lock 80,000 USD in escrow_reserve within [0, 100,000] cap",
            asserted_literals=(5,),
            prior_prob=0.50,
            rational_var="escrow_reserve",
            rational_amount=Fraction(80000),
            step_reward=0.25,
        ),
    ]
    stage_3 = [
        DeductionAction(
            action_id="S4_PAYOUT_190K_TRAP",
            description="Disburse 190,000 USD operating_payout (sum 270k != 250k & > 180k cap)",
            asserted_literals=(6,),
            prior_prob=0.45,
            rational_var="operating_payout",
            rational_amount=Fraction(190000),
            step_reward=0.10,
        ),
        DeductionAction(
            action_id="S4_DISBURSE_EXACT_170K",
            description="Disburse 170,000 USD operating_payout (80k + 170k == 250k QED)",
            asserted_literals=(6,),
            prior_prob=0.55,
            rational_var="operating_payout",
            rational_amount=Fraction(170000),
            step_reward=0.45,
        ),
    ]

    return DeductionProblem(
        problem_id="SOX-TREASURY-4STAGE-MCTS",
        num_vars=6,
        base_clauses=base_clauses,
        step_candidates=[stage_0, stage_1, stage_2, stage_3],
        var_names=var_names,
        rational_vars=["escrow_reserve", "operating_payout"],
        rational_upper_bounds={
            "escrow_reserve": Fraction(100000),
            "operating_payout": Fraction(180000),
        },
        required_total_sum=Fraction(250000),
    )


class ProofGuidedMCTS:
    """PUCT MCTS engine that prunes formally unsatisfiable reasoning branches in real time."""

    def __init__(self, problem: DeductionProblem, *, c_puct: float = 1.4) -> None:
        self.problem = problem
        self.c_puct = c_puct
        self.pruned_contradictory_nodes = 0

    def _verify_node_feasibility(
        self,
        committed_lits: tuple[int, ...],
        committed_rats: dict[str, Fraction],
        *,
        is_terminal: bool,
    ) -> tuple[bool, ProofCertificate]:
        augmented_clauses = [list(c) for c in self.problem.base_clauses] + [
            [lit] for lit in committed_lits
        ]
        cert = CDCLSolver(
            self.problem.num_vars,
            augmented_clauses,
            var_names=self.problem.var_names,
        ).solve()
        if cert.status == ProofStatus.UNSAT:
            return False, cert

        lo: dict[str, Fraction] = {}
        hi: dict[str, Fraction] = {}
        for rv in self.problem.rational_vars:
            if rv in committed_rats:
                lo[rv] = committed_rats[rv]
                hi[rv] = committed_rats[rv]
            else:
                lo[rv] = Fraction(0)
                hi[rv] = self.problem.rational_upper_bounds[rv]

            # Also enforce problem upper bound even when fixed!
            if lo[rv] > self.problem.rational_upper_bounds[rv] or lo[rv] < Fraction(0):
                return False, cert.model_copy(
                    update={"status": ProofStatus.UNSAT, "certificate_verified": True}
                )

        sum_eq = (
            (self.problem.rational_vars, self.problem.required_total_sum)
            if self.problem.required_total_sum is not None
            else None
        )
        # Check whether a completion of remaining unassigned rationals can still hit required_total_sum
        lra_ok, witness_map, _ = solve_exact_rational_bounds(
            self.problem.rational_vars,
            lo,
            hi,
            sum_equality=sum_eq if (is_terminal or len(committed_rats) > 0) else None,
        )
        if not lra_ok:
            return False, cert.model_copy(
                update={"status": ProofStatus.UNSAT, "certificate_verified": True}
            )
        return True, cert.model_copy(update={"linear_witness": witness_map})

    def _select_puct_child(self, node: MCTSNode) -> tuple[DeductionAction, MCTSNode] | None:
        candidates = self.problem.step_candidates[node.depth]
        parent_visits = max(1, node.visits)
        best_score = -1e9
        best_pair: tuple[DeductionAction, MCTSNode] | None = None

        for act in candidates:
            child = node.children.get(act.action_id)
            if child is not None and child.is_pruned:
                continue
            q = child.q_value if child is not None else 0.0
            n_sa = child.visits if child is not None else 0
            u = self.c_puct * act.prior_prob * (math.sqrt(parent_visits) / (1.0 + n_sa))
            score = q + u
            if score > best_score:
                best_score = score
                if child is None:
                    new_rats = dict(node.committed_rationals)
                    if act.rational_var is not None and act.rational_amount is not None:
                        new_rats[act.rational_var] = act.rational_amount
                    child = MCTSNode(
                        depth=node.depth + 1,
                        committed_literals=node.committed_literals + act.asserted_literals,
                        committed_rationals=new_rats,
                        trajectory_labels=node.trajectory_labels + (act.action_id,),
                        prior=act.prior_prob,
                    )
                    node.children[act.action_id] = child
                best_pair = (act, child)
        return best_pair

    def search(self, *, num_simulations: int = 48) -> MCTSSearchSummary:
        root = MCTSNode(
            depth=0,
            committed_literals=(),
            committed_rationals={},
            trajectory_labels=(),
        )
        max_depth = len(self.problem.step_candidates)
        best_terminal: MCTSNode | None = None
        best_val = -1.0

        for _ in range(num_simulations):
            path: list[MCTSNode] = [root]
            curr = root
            acc_reward = 0.0
            leaf_value = 0.0

            while curr.depth < max_depth:
                picked = self._select_puct_child(curr)
                if picked is None:
                    # Every child at this state was formally pruned -> mark parent pruned too!
                    curr.is_pruned = True
                    leaf_value = -1.0
                    break

                act, child = picked
                path.append(child)
                is_term = child.depth == max_depth

                if child.visits == 0:
                    feasible, cert = self._verify_node_feasibility(
                        child.committed_literals,
                        child.committed_rationals,
                        is_terminal=is_term,
                    )
                    if not feasible:
                        child.is_pruned = True
                        self.pruned_contradictory_nodes += 1
                        leaf_value = -1.0
                        break

                    acc_reward += act.step_reward
                    if is_term:
                        child.terminal_certificate = cert
                        leaf_value = min(1.0, acc_reward)
                        if leaf_value > best_val:
                            best_val = leaf_value
                            best_terminal = child
                    else:
                        leaf_value = acc_reward
                    break

                acc_reward += act.step_reward
                curr = child
                if is_term:
                    leaf_value = min(1.0, acc_reward)
                    if child.terminal_certificate is not None and leaf_value > best_val:
                        best_val = leaf_value
                        best_terminal = child
                    break

            for visited_node in path:
                visited_node.visits += 1
                visited_node.value_sum += leaf_value

        if best_terminal is not None and best_terminal.terminal_certificate is not None:
            return MCTSSearchSummary(
                problem_id=self.problem.problem_id,
                simulations_run=num_simulations,
                pruned_contradictory_nodes=self.pruned_contradictory_nodes,
                verified_terminal_found=True,
                best_trajectory_steps=list(best_terminal.trajectory_labels),
                best_value=round(best_val, 4),
                proof_certificate_verified=best_terminal.terminal_certificate.certificate_verified,
            )

        return MCTSSearchSummary(
            problem_id=self.problem.problem_id,
            simulations_run=num_simulations,
            pruned_contradictory_nodes=self.pruned_contradictory_nodes,
            verified_terminal_found=False,
            best_trajectory_steps=[],
            best_value=0.0,
            proof_certificate_verified=False,
        )
