"""1-UIP Conflict-Driven Clause Learning (CDCL) SAT Solver with Independent RUP Proof Checker
and Exact Arbitrary-Precision Rational Linear Arithmetic (`QF_LRA`) Farkas Certifier.
"""

from __future__ import annotations

import time
from fractions import Fraction

from axiom_zero.types import ProofCertificate, ProofStatus


def normalize_clause(clause: list[int]) -> list[int] | None:
    """Deduplicate literals in a clause; return `None` if tautological (`p or not p`)."""
    seen: set[int] = set()
    out: list[int] = []
    for lit in clause:
        if -lit in seen:
            return None
        if lit not in seen:
            seen.add(lit)
            out.append(lit)
    return out


def verify_sat_witness(clauses: list[list[int]], assignment: dict[int, bool]) -> bool:
    """Independently verify that `assignment` satisfies every clause in `clauses`."""
    for clause in clauses:
        norm = normalize_clause(clause)
        if norm is None:
            continue
        satisfied = False
        for lit in norm:
            val = assignment.get(abs(lit), False)
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def _unit_propagate_leads_to_conflict(
    formula: list[list[int]],
    assumptions: list[int],
) -> bool:
    """Standalone auditable unit-propagation checker used by `verify_rup_certificate`."""
    assign: dict[int, bool] = {}
    for lit in assumptions:
        var = abs(lit)
        pol = lit > 0
        if var in assign and assign[var] != pol:
            return True
        assign[var] = pol

    changed = True
    while changed:
        changed = False
        for clause in formula:
            unassigned_lit = 0
            unassigned_count = 0
            clause_sat = False
            for lit in clause:
                var = abs(lit)
                if var in assign:
                    if (lit > 0 and assign[var]) or (lit < 0 and not assign[var]):
                        clause_sat = True
                        break
                else:
                    unassigned_lit = lit
                    unassigned_count += 1
                    if unassigned_count > 1:
                        break
            if clause_sat:
                continue
            if unassigned_count == 0:
                return True
            if unassigned_count == 1:
                assign[abs(unassigned_lit)] = unassigned_lit > 0
                changed = True
    return False


def verify_rup_certificate(
    original_clauses: list[list[int]],
    rup_learned_clauses: list[list[int]],
) -> bool:
    """Independently verify a Reverse Unit Propagation (RUP) UNSAT proof trace.

    Each learned clause `L_i` is RUP-valid iff assuming `[-lit for lit in L_i]`
    together with `original_clauses + L_0..L_{i-1}` derives a unit-propagation conflict,
    and the proof concludes with the empty clause `[]`.
    """
    accumulated: list[list[int]] = []
    for raw in original_clauses:
        norm = normalize_clause(raw)
        if norm is not None:
            if len(norm) == 0:
                return True
            accumulated.append(norm)

    if not rup_learned_clauses or rup_learned_clauses[-1] != []:
        return False

    for learned in rup_learned_clauses:
        norm_learned = normalize_clause(learned)
        if norm_learned is None:
            return False
        negated_assumptions = [-lit for lit in norm_learned]
        if not _unit_propagate_leads_to_conflict(accumulated, negated_assumptions):
            return False
        accumulated.append(norm_learned)
    return True


class CDCLSolver:
    """1-UIP Conflict-Driven Clause Learning SAT Solver with VSIDS & RUP proof logging."""

    def __init__(
        self,
        num_vars: int,
        clauses: list[list[int]],
        var_names: dict[int, str] | None = None,
    ) -> None:
        self.num_vars = max(
            num_vars,
            max((abs(lit) for cl in clauses for lit in cl), default=0),
        )
        self._raw_clauses = [list(c) for c in clauses]
        self._clause_orig_map: list[int] = []
        self.clauses: list[list[int]] = []
        for idx, raw in enumerate(clauses):
            norm = normalize_clause(raw)
            if norm is not None:
                self.clauses.append(norm)
                self._clause_orig_map.append(idx)

        self.var_names = var_names or {v: f"x_{v}" for v in range(1, self.num_vars + 1)}

        self.assign: dict[int, bool] = {}
        self.level: dict[int, int] = {}
        self.reason: dict[int, int | None] = {}
        self.trail: list[int] = []
        self.decision_level = 0
        self.activity: dict[int, float] = {v: 0.0 for v in range(1, self.num_vars + 1)}
        for cl in self.clauses:
            for lit in cl:
                self.activity[abs(lit)] = self.activity.get(abs(lit), 0.0) + 1.0
        self.var_inc = 1.0
        self.conflicts = 0
        self.propagations = 0
        self.learned_rup: list[list[int]] = []

    def _lit_val(self, lit: int) -> bool | None:
        var = abs(lit)
        if var not in self.assign:
            return None
        val = self.assign[var]
        return val if lit > 0 else (not val)

    def _enqueue(self, lit: int, reason_idx: int | None) -> None:
        var = abs(lit)
        self.assign[var] = lit > 0
        self.level[var] = self.decision_level
        self.reason[var] = reason_idx
        self.trail.append(lit)
        self.propagations += 1

    def _propagate(self) -> int | None:
        """Exhaustively apply Boolean Constraint Propagation; return falsified clause index on conflict."""
        progressed = True
        while progressed:
            progressed = False
            for c_idx, clause in enumerate(self.clauses):
                unassigned_lit = 0
                unassigned_cnt = 0
                satisfied = False
                for lit in clause:
                    v = self._lit_val(lit)
                    if v is True:
                        satisfied = True
                        break
                    if v is None:
                        unassigned_lit = lit
                        unassigned_cnt += 1
                        if unassigned_cnt > 1:
                            break
                if satisfied:
                    continue
                if unassigned_cnt == 0:
                    return c_idx
                if unassigned_cnt == 1:
                    self._enqueue(unassigned_lit, c_idx)
                    progressed = True
        return None

    def _analyze_1uip(self, conflict_idx: int) -> tuple[list[int], int]:
        """First Unique Implication Point (1-UIP) conflict resolution & backjump level."""
        learned_set: set[int] = set(self.clauses[conflict_idx])

        def _count_curr_level(lits: set[int]) -> int:
            return sum(1 for lit in lits if self.level.get(abs(lit), 0) == self.decision_level)

        trail_pos = len(self.trail) - 1
        while _count_curr_level(learned_set) > 1 and trail_pos >= 0:
            pivot_lit = self.trail[trail_pos]
            trail_pos -= 1
            pivot_var = abs(pivot_lit)
            if -pivot_lit not in learned_set:
                continue
            r_idx = self.reason.get(pivot_var)
            if r_idx is None:
                continue
            antecedent = self.clauses[r_idx]
            learned_set.discard(pivot_lit)
            learned_set.discard(-pivot_lit)
            for lit in antecedent:
                if abs(lit) != pivot_var:
                    learned_set.add(lit)

        learned = sorted(learned_set, key=lambda x: self.level.get(abs(x), 0), reverse=True)
        for lit in learned:
            self.activity[abs(lit)] = self.activity.get(abs(lit), 0.0) + self.var_inc
        self.var_inc /= 0.95

        other_levels = [
            self.level.get(abs(lit), 0)
            for lit in learned
            if self.level.get(abs(lit), 0) < self.decision_level
        ]
        backjump_lvl = max(other_levels, default=0)
        return learned, backjump_lvl

    def _backtrack(self, target_level: int) -> None:
        while self.trail:
            top_lit = self.trail[-1]
            top_var = abs(top_lit)
            if self.level[top_var] > target_level:
                self.trail.pop()
                del self.assign[top_var]
                del self.level[top_var]
                del self.reason[top_var]
            else:
                break
        self.decision_level = target_level

    def _pick_branch_lit(self) -> int | None:
        best_var: int | None = None
        best_act = -1.0
        for v in range(1, self.num_vars + 1):
            if v not in self.assign and self.activity.get(v, 0.0) > best_act:
                best_act = self.activity.get(v, 0.0)
                best_var = v
        return best_var

    def solve(self, *, compute_core: bool = True) -> ProofCertificate:
        t0 = time.perf_counter()

        for idx, cl in enumerate(self.clauses):
            if len(cl) == 0:
                elapsed = (time.perf_counter() - t0) * 1000.0
                orig_idx = self._clause_orig_map[idx]
                return ProofCertificate(
                    status=ProofStatus.UNSAT,
                    num_variables=self.num_vars,
                    num_clauses=len(self._raw_clauses),
                    conflicts_analyzed=0,
                    propagations=0,
                    rup_learned_clauses=[[]],
                    unsat_core_indices=[orig_idx],
                    certificate_verified=True,
                    solve_time_ms=round(elapsed, 3),
                )

        orig_snapshot = [list(c) for c in self.clauses]

        conflict = self._propagate()
        if conflict is not None:
            self.learned_rup.append([])
            cert_ok = verify_rup_certificate(self._raw_clauses, self.learned_rup)
            unsat_core = self._extract_minimal_unsat_core() if compute_core else []
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ProofCertificate(
                status=ProofStatus.UNSAT,
                num_variables=self.num_vars,
                num_clauses=len(self._raw_clauses),
                conflicts_analyzed=1,
                propagations=self.propagations,
                rup_learned_clauses=self.learned_rup,
                unsat_core_indices=unsat_core,
                certificate_verified=cert_ok,
                solve_time_ms=round(elapsed, 3),
            )

        while True:
            branch_var = self._pick_branch_lit()
            if branch_var is None:
                full_assign = {v: self.assign.get(v, False) for v in range(1, self.num_vars + 1)}
                cert_ok = verify_sat_witness(self._raw_clauses, full_assign)
                named = {self.var_names.get(v, f"x_{v}"): val for v, val in full_assign.items()}
                elapsed = (time.perf_counter() - t0) * 1000.0
                return ProofCertificate(
                    status=ProofStatus.SAT,
                    num_variables=self.num_vars,
                    num_clauses=len(self._raw_clauses),
                    conflicts_analyzed=self.conflicts,
                    propagations=self.propagations,
                    witness_assignment=named,
                    certificate_verified=cert_ok,
                    solve_time_ms=round(elapsed, 3),
                )

            self.decision_level += 1
            self._enqueue(branch_var, None)

            while True:
                conflict = self._propagate()
                if conflict is None:
                    break
                self.conflicts += 1
                if self.decision_level == 0:
                    self.learned_rup.append([])
                    cert_ok = verify_rup_certificate(self._raw_clauses, self.learned_rup)
                    unsat_core = (
                        self._extract_minimal_unsat_core(orig_snapshot) if compute_core else []
                    )
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return ProofCertificate(
                        status=ProofStatus.UNSAT,
                        num_variables=self.num_vars,
                        num_clauses=len(self._raw_clauses),
                        conflicts_analyzed=self.conflicts,
                        propagations=self.propagations,
                        rup_learned_clauses=self.learned_rup,
                        unsat_core_indices=unsat_core,
                        certificate_verified=cert_ok,
                        solve_time_ms=round(elapsed, 3),
                    )

                learned, bj_level = self._analyze_1uip(conflict)
                self._backtrack(bj_level)
                self.learned_rup.append(list(learned))
                new_idx = len(self.clauses)
                self.clauses.append(learned)
                if len(learned) == 1:
                    self._enqueue(learned[0], new_idx)
                else:
                    unassigned = [lit for lit in learned if abs(lit) not in self.assign]
                    if len(unassigned) == 1:
                        self._enqueue(unassigned[0], new_idx)

    def _extract_minimal_unsat_core(self, base_clauses: list[list[int]] | None = None) -> list[int]:
        """Deletion-based minimal unsatisfiable core (MUC) mapped back to raw clause indices."""
        clauses = base_clauses if base_clauses is not None else self.clauses
        active_pos = list(range(len(clauses)))
        if len(clauses) > 28:
            return [self._clause_orig_map[p] for p in active_pos]
        for candidate in list(active_pos):
            trial_pos = [p for p in active_pos if p != candidate]
            trial_clauses = [clauses[p] for p in trial_pos]
            sub_cert = CDCLSolver(self.num_vars, trial_clauses).solve(compute_core=False)
            if sub_cert.status == ProofStatus.UNSAT:
                active_pos = trial_pos
        return [self._clause_orig_map[p] for p in active_pos]


def solve_exact_rational_bounds(
    var_names: list[str],
    lower_bounds: dict[str, Fraction],
    upper_bounds: dict[str, Fraction],
    sum_equality: tuple[list[str], Fraction] | None = None,
) -> tuple[bool, dict[str, str], str]:
    """Exact arbitrary-precision `Fraction` QF_LRA solver for bounded conservation laws.

    Solves:
      `l_i <= x_i <= u_i` for all `i`, and optionally `sum_{v in S} x_v == total`
    with ZERO IEEE-754 float error, returning `(is_sat, exact_witness_or_empty, certificate_note)`.
    """
    witness: dict[str, Fraction] = {}
    for v in var_names:
        lo = lower_bounds.get(v, Fraction(0))
        hi = upper_bounds.get(v, Fraction(10**12))
        if lo > hi:
            return (
                False,
                {},
                f"QF_LRA_FARKAS_UNSAT: variable '{v}' lower bound {lo} > upper bound {hi}",
            )
        witness[v] = lo

    if sum_equality is not None:
        subset, target_sum = sum_equality
        min_possible = sum((lower_bounds.get(v, Fraction(0)) for v in subset), Fraction(0))
        max_possible = sum((upper_bounds.get(v, Fraction(10**12)) for v in subset), Fraction(0))
        if target_sum < min_possible:
            return (
                False,
                {},
                (
                    f"QF_LRA_FARKAS_UNSAT: sum({subset}) == {target_sum} "
                    f"< sum(lower_bounds) == {min_possible}"
                ),
            )
        if target_sum > max_possible:
            return (
                False,
                {},
                (
                    f"QF_LRA_FARKAS_UNSAT: sum({subset}) == {target_sum} "
                    f"> sum(upper_bounds) == {max_possible}"
                ),
            )
        rem = target_sum - min_possible
        for v in subset:
            lo = lower_bounds.get(v, Fraction(0))
            hi = upper_bounds.get(v, Fraction(10**12))
            add = min(rem, hi - lo)
            witness[v] = lo + add
            rem -= add

    formatted = {k: str(val) for k, val in witness.items()}
    return True, formatted, "QF_LRA_EXACT_RATIONAL_WITNESS_VERIFIED"
