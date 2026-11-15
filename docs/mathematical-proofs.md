# Formal Mathematical Proofs & Runtime Invariants (`AxiomZero`)

This document states the four core mathematical theorems implemented and property-tested inside `axiom-zero`, linking each theorem directly to its executable module and `Hypothesis` verification test.

---

## Theorem 1 — Exact Lossless Distribution Equivalence of Speculative Decoding

**Implementation**: `csrc/axiom_kernels.cpp` (`axiom_speculative_verify_step`) & `src/axiom_zero/transformer/speculative.py`  
**Automated Proof Test**: `tests/test_kernels_and_speculative.py::test_hypothesis_speculative_lossless_distribution_theorem`

Let $p(x)$ denote the target GQA Transformer distribution over vocabulary $\mathcal{V}$ and $q(x)$ denote an arbitrary draft distribution with $\operatorname{supp}(p) \subseteq \operatorname{supp}(q)$. Given a proposed draft token $x \sim q$:

1. Accept $x$ with probability $\alpha(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)$.
2. Otherwise (with probability $1 - \sum_{z \in \mathcal{V}} q(z)\alpha(z) = \sum_{z \in \mathcal{V}} \max(0, p(z) - q(z))$), sample a replacement token $y \sim p_{\text{res}}$, where:

$$
p_{\text{res}}(y) = \frac{\max(0, p(y) - q(y))}{\sum_{z \in \mathcal{V}} \max(0, p(z) - q(z))}
$$

**Proof of Exactness**: For every token $v \in \mathcal{V}$, the marginal emission probability $P(\text{emit } v)$ is:

$$
P(\text{emit } v) = q(v)\min\!\left(1, \frac{p(v)}{q(v)}\right) + \left(1 - \sum_{z} \min(q(z), p(z))\right) \cdot \frac{\max(0, p(v) - q(v))}{\sum_{z} \max(0, p(z) - q(z))}
$$

Since $\sum_z p(z) - \sum_z \min(q(z), p(z)) = \sum_z (p(z) - \min(q(z), p(z))) = \sum_z \max(0, p(z) - q(z))$, the normalization factors cancel identically whenever rejection probability is positive:

$$
P(\text{emit } v) = \min(q(v), p(v)) + \max(0, p(v) - q(v)) = p(v) \quad \forall v \in \mathcal{V}. \quad \blacksquare
$$

Thus speculative decoding reduces serial target-model forward passes without changing output entropy or token probabilities by even a single bit of approximation error.

---

## Theorem 2 — Unit-Norm Top-$K$ Sparse Autoencoder Disentanglement & Linear Causal Steering

**Implementation**: `src/axiom_zero/interp/sae.py` (`TopKSparseAutoencoder`, `train_sae_and_discover_circuits`)  
**Automated Proof Test**: `tests/test_sae_and_transformer.py::test_topk_sae_unit_norm_and_sparsity_invariants`

Given residual-stream activation $h \in \mathbb{R}^{d_{\text{model}}}$, an overcomplete dictionary $W_{\text{dec}} = [d_1, \dots, d_M] \in \mathbb{R}^{d_{\text{model}} \times M}$ ($M = 6 \cdot d_{\text{model}}$) encodes and reconstructs $h$ via:

$$
z = \operatorname{TopK}\!\left(\operatorname{ReLU}\!\left(W_{\text{enc}}(h - b_{\text{pre}}) + b_{\text{enc}}\right), k\right), \qquad \hat{h} = W_{\text{dec}} z + b_{\text{pre}}
$$

Enforcing the exact Riemannian manifold projection $\|d_i\|_2 = 1$ after every optimizer step prevents scale degeneracy ($W_{\text{enc}} \leftarrow c W_{\text{enc}}, W_{\text{dec}} \leftarrow c^{-1} W_{\text{dec}}$) and guarantees bounded sparsity $\|z\|_0 \le k$. Injecting steering perturbation $h' = h + \alpha d_{f^*}$ along a monosemantic circuit $f^*$ shifts pre-norm readout alignment monotonically in $\alpha$.

---

## Theorem 3 — Soundness of Reverse Unit Propagation (`RUP`) Unsatisfiability Certificates

**Implementation**: `src/axiom_zero/verifier/cdcl_smt.py` (`CDCLSolver`, `verify_rup_certificate`)  
**Automated Proof Test**: `tests/test_cdcl_rup_and_mcts.py::test_hypothesis_cdcl_matches_truth_table_and_verifies_rup`

Let $F$ be an input CNF formula and $(C_1, C_2, \dots, C_m)$ be a sequence of conflict clauses learned via First Unique Implication Point (`1-UIP`) resolution concluding with $C_m = \emptyset$. Clause $C_i$ has the **Reverse Unit Propagation (`RUP`)** property relative to $F_{i-1} = F \cup \{C_1, \dots, C_{i-1}\}$ iff unit propagation (`BCP`) on $F_{i-1} \cup \{\neg \ell : \ell \in C_i\}$ derives a falsified clause.

**Soundness**: If $F_{i-1} \cup \{\neg \ell : \ell \in C_i\} \vdash_{\text{BCP}} \bot$, then every truth assignment $\sigma \models F_{i-1}$ cannot satisfy $\bigwedge_{\ell \in C_i} \neg \ell$, meaning $\sigma \models \bigvee_{\ell \in C_i} \ell = C_i$. By induction on $i = 1 \dots m$, $F \models C_m = \emptyset$, proving $F$ is unsatisfiable independent of any heuristic state inside `CDCLSolver`.

---

## Theorem 4 — Exact Rational Conservation (`QF_LRA`) Without IEEE-754 Drift

**Implementation**: `src/axiom_zero/verifier/cdcl_smt.py` (`solve_exact_rational_bounds`)  
**Automated Proof Test**: `tests/test_cdcl_rup_and_mcts.py::test_exact_rational_qf_lra_conservation_and_farkas_unsat`

For bounded linear allocation variables $x_i \in [l_i, u_i] \subset \mathbb{Q}$ ($i \in S$) subject to $\sum_{i \in S} x_i = T \in \mathbb{Q}$, a rational witness exists if and only if $l_i \le u_i$ for all $i$ and:

$$
\sum_{i \in S} l_i \;\le\; T \;\le\; \sum_{i \in S} u_i
$$

When violated, the opposing Farkas dual multiplier $y \in \{+1, -1\}$ certifies infeasibility over $\mathbb{Q}$ using exact `fractions.Fraction` numerator/denominator arithmetic, eliminating binary floating-point rounding bugs (`0.1 + 0.2 != 0.3`).
