# ADR 0003: 1-UIP CDCL Solver Coupled to an Independent RUP Proof-Checking Kernel

- **Status**: Accepted
- **Context**: Complex SAT solvers using VSIDS activity heuristics and non-chronological backjumping can harbor subtle soundness bugs if trusted blindly as black-box safety gates.
- **Decision**: Require `CDCLSolver` (`src/axiom_zero/verifier/cdcl_smt.py`) to log every `1-UIP` learned conflict clause and pass the resulting trace through a standalone, stateless Reverse Unit Propagation (`verify_rup_certificate`) checker for `UNSAT` verdicts and `verify_sat_witness` for `SAT` verdicts, backed by `Hypothesis` fuzzing against a `2^N` truth-table oracle.
- **Consequences**: Decouples search complexity from proof-checking trust (`Certificate-Producing Verification`), while providing deletion-based Minimal Unsatisfiable Cores (`MUC`) for downstream LLM repair.
