# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `1.0.x` | Yes |

## Architectural Security Guarantees

1. **Independent Proof-Checking Kernel (`verify_rup_certificate` & `verify_sat_witness`)**: `CDCLSolver` is treated as an untrusted search heuristic. Every `UNSAT` verdict must conclude with an empty clause derived via Reverse Unit Propagation (`RUP`) verified step-by-step by a standalone 50-line unit-propagation kernel, and every `SAT` verdict is checked directly against raw input clauses.
2. **Zero IEEE-754 Floating-Point Drift on Financial/Resource Invariants**: `solve_exact_rational_bounds` executes over `fractions.Fraction` arbitrary-precision rationals rather than binary floats.
3. **Zero Unverified LLM Execution Leakage**: In Counterexample-Guided Inductive Synthesis (`CEGIS`), candidate JSON payloads emitted by local/remote LLMs are strictly gated by `CDCLSolver` + `QF_LRA` conservation checks before acceptance.
4. **Native C++20 Shared Library Integrity**: `NativeKernelBridge` hashes `csrc/axiom_kernels.cpp` with SHA-256 prior to compiling/loading `.axiom_cache/libaxiom_kernels_<hash>.{dylib,so}` and enforces bounds-checked indices on draft token verification.

## Reporting a Vulnerability

Please report security issues privately to `sendemailtonagarjuna@gmail.com` with reproduction steps. Do not open public GitHub issues for undisclosed vulnerabilities.
