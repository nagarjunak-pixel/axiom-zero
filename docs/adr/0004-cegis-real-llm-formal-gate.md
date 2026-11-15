# ADR 0004: Zero-Mock CEGIS Loop Over Live Local LLMs (`Ollama` + `QF_LRA`)

- **Status**: Accepted
- **Context**: Raw autoregressive LLMs routinely fail multi-hop Boolean implications (`Modus Tollens`, biconditionals, adversarial override traps) and tight exact-sum multi-tranche conservation constraints on `Pass@1`.
- **Decision**: Implement Counterexample-Guided Inductive Synthesis (`CEGIS`) in `src/axiom_zero/real_llm_eval.py`: candidate LLM JSON outputs are checked by `CDCLSolver` + `solve_exact_rational_bounds` (`fractions.Fraction`), feeding exact MUC + Farkas bound/sum counterexamples back to the live local `qwen3.5:4b` model (`<= 3` turns) and hard-blocking any unverified output.
- **Consequences**: On the 10-task adversarial formal suite against real local `qwen3.5:4b`, verified task completion rises `9x` (`10.0%` Raw `Pass@1` -> `90.0%` CEGIS Verified) while unverified policy/arithmetic leakage drops from `90.0%` to `0.0%`.
