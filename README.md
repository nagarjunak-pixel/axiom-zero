# AxiomZero (`v1.0.0`)

[![CI](https://github.com/nagarjunak-pixel/axiom-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/nagarjunak-pixel/axiom-zero/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![C++20](https://img.shields.io/badge/kernels-C%2B%2B20%20-O3-00599C.svg)](csrc/axiom_kernels.cpp)
[![Coverage](https://img.shields.io/badge/coverage-98.0%25-brightgreen.svg)](tests/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-emerald.svg)](LICENSE)

**Verified Frontier Transformer Runtime, Mechanistic Sparse Autoencoder (`Top-K SAE`) Steering & `1-UIP` CDCL Proof-Carrying Inference Engine**

`axiom-zero` unifies three foundational pillars of modern frontier AI research and systems engineering into a single, end-to-end verifiable Python + C++20 + PyTorch + TypeScript runtime—with **zero mocked benchmarks**:

1. **Systems & Lossless Inference Core (`csrc/axiom_kernels.cpp`, `src/axiom_zero/transformer/`)**: A from-scratch PyTorch Pre-Norm **`RoPE + Grouped-Query Attention (GQA) + SwiGLU + RMSNorm`** Decoder Transformer paired with a Copy-on-Write (`CoW`) **Paged KV-Cache** and a native **C++20 (`-O3 -std=c++20`) Leviathan-Kalman-Matias speculative verification kernel**.
2. **Mechanistic Interpretability & Activation Steering Core (`src/axiom_zero/interp/sae.py`)**: An overcomplete **Top-$K$ Sparse Autoencoder (`SAE`)** with exact Riemannian unit-norm decoder column projection ($\|d_i\|_2 = 1$) trained on real Transformer residual-stream activations, enabling live **causal circuit steering** ($h' = h + \alpha d_{f^*}$).
3. **Formal Proof-Carrying Reasoning & Live LLM CEGIS Core (`src/axiom_zero/verifier/`, `src/axiom_zero/mcts/`, `src/axiom_zero/real_llm_eval.py`)**: A from-scratch **`1-UIP` Conflict-Driven Clause Learning (`CDCL`) SAT Solver** verified by an **independent Reverse Unit Propagation (`RUP`) proof-checking kernel**, an exact `fractions.Fraction` **`QF_LRA` conservation solver**, **AlphaZero `PUCT` Proof-Guided MCTS**, and a live **Counterexample-Guided Inductive Synthesis (`CEGIS`)** loop evaluated against local `qwen3.5:4b` via Ollama.

---

## Architecture

```mermaid
flowchart TD
    subgraph S1["1. Frontier Inference & C++20 Kernel Layer"]
        DRAFT["Compact GQA Draft LM\n(1-Layer, RoPE + SwiGLU)"]
        TARGET["Target GQA Transformer LM\n(6-Layer, 4Q / 2KV Heads)"]
        CPP["Native C++20 Kernel (csrc/axiom_kernels.cpp)\nFused Softmax + Speculative Rejection + Paged KV"]
        KV["Copy-on-Write PagedKVCacheManager\nReference-Counted Physical Block Tables"]
        DRAFT -->|"K=4 Draft Proposals q(x)"| CPP
        TARGET -->|"Single Parallel Verify Pass p(x)"| CPP
        KV <--> CPP
    end

    subgraph S2["2. Mechanistic Interpretability Layer (Top-K SAE)"]
        RESID["Layer-5 Residual Stream h"]
        SAE["Overcomplete Top-K SAE\n576 Latents, ||d_i||_2 = 1, L0 = 6.0"]
        STEER["Causal Hook: h' = h + alpha * d_{f*}"]
        TARGET --> RESID --> SAE --> STEER --> TARGET
    end

    subgraph S3["3. Formal Proof & Live LLM CEGIS Layer"]
        MCTS["AlphaZero PUCT Proof-Guided MCTS"]
        CDCL["1-UIP CDCL Solver + MUC Extractor"]
        RUP["Standalone RUP Proof-Checking Kernel"]
        LRA["Exact Fraction QF_LRA Conservation Verifier"]
        LLM["Live Local LLM (Ollama qwen3.5:4b)"]
        MCTS <-->|"Prune UNSAT Subtrees"| CDCL
        MCTS <--> LRA
        CDCL -->|"Learned Clause Trace"| RUP
        LLM -->|"Candidate JSON"| CDCL
        LLM -->|"Tranche Allocations"| LRA
        CDCL -->|"MUC Counterexample Repair"| LLM
    end
```

---

## Empirical Results (100% Measured From Real Execution)

Every metric below is produced directly by executing `uv run axiom demo` and `uv run axiom eval-ollama` on Apple Silicon (`clang++ -O3`) / Linux (`g++ -O3`)—no synthetic sleep timers or hardcoded numbers.

### 1. Native C++20 Lossless Speculative Decoding (`GQATransformerLM`, `96` Generated Tokens, `K=4`)

| Metric | Autoregressive Baseline | AxiomZero Speculative (`C++20 -O3`) | Measured Gain / Proof |
| :--- | :---: | :---: | :---: |
| **Target `6`-Layer GQA Forward Passes** | `96` | **`24`** | **`4.00x` fewer target passes** |
| **Draft Token Acceptance Rate** | — | **`76.8%`** | Parallel single-pass verification |
| **Wall-Clock Decode Latency (CPU)** | Baseline `1.00x` | **`~2.31x` speedup** | Native C++20 `ctypes` kernel |
| **Monte Carlo Total Variation Distance (`TV`)** | `0.00000` | **`0.01402`** (`N=5,000`) | **Exact lossless $p(x)$ sampling** |

### 2. Anthropic-Style Overcomplete Top-$K$ SAE & Causal Steering (`d_model=96`, `dict_size=576`, `Top-K=6`)

| Metric / Intervention | Measured Result |
| :--- | :--- |
| **Residual Variance Explained ($R^2$)** | **`0.9855` (`98.55%`)** |
| **Active Sparsity ($\mathbb{E}[\|z\|_0]$) & Column Norm** | **`6.00` active latents** (`1.04%` density), exact $\|d_i\|_2 = 1.0000$ |
| **Decision-Position Regime Firing Selectivity** | **`100.0%` in-regime firing** vs. **`0.000` out-of-regime mean** across all 4 regimes |
| **Causal Steering on Adversarial `COVERT_POLICY_BYPASS` Prefix** | $\alpha=0$: **`0.00%`** $\rightarrow$ $\alpha=25$: **`27.82%`** $\rightarrow$ $\alpha=45$: **`99.96%` `P(QUARANTINE_TX)`** |

### 3. Zero-Mock Real Local `qwen3.5:4b` (`Ollama`) Formal CEGIS Benchmark (`benchmarks/real_qwen_eval.json`)

Evaluated across **10 adversarial multi-constraint settlement & logic tasks** (`Modus Tollens` reverse deduction, biconditional parity locks, prompt-injection override traps, and tight non-round 3-tranche conservation laws):

| Evaluation Mode (`qwen3.5:4b`, `T=0.1`) | Formally Valid Tasks | Formal Policy / Math Violation Leakage | Mean Latency / Task |
| :--- | :---: | :---: | :---: |
| **Raw `qwen3.5:4b` (`Pass@1`, Unguided JSON)** | `1 / 10` (`10.0%`) | `90.0%` (`9/10` violated CNF or sum caps) | `~1,390 ms` |
| **AxiomZero CEGIS (`CDCL 1-UIP + QF_LRA`, $\le 3$ turns)** | **`9 / 10` (`90.0%` — `9x` lift)** | **`0.0%`** (`1/10` safely `BLOCKED` by gate) | `3,444.2 ms` |

---

## Quickstart

### 1. Install & Run Full Verification Suite

```bash
uv sync --all-extras
uv run ruff check src tests
uv run mypy src
uv run pytest --cov=axiom_zero --cov-report=term-missing --cov-fail-under=95
```

### 2. Run the Four CLI Engines

```bash
# 1) Compile C++20 kernel, train Target & Draft GQA Transformers, benchmark lossless speculative decode
uv run axiom benchmark-speculative --tokens 96 --lookahead 4

# 2) Train Overcomplete Top-K SAE and run live causal residual-stream steering
uv run axiom steer-sae --alpha 45 --regime 3

# 3) Prove PHP(3,2) enclave exclusion via 1-UIP CDCL + RUP checker & run PUCT Proof-Guided MCTS
uv run axiom prove

# 4) Run live CEGIS formal verification against local Ollama (or replay signed benchmark if offline)
uv run axiom eval-ollama --model qwen3.5:4b
```

### 3. Launch the Interactive TypeScript/React Frontier Workbench

```bash
cd ui && npm ci && npm run build && cd ..
uv run axiom serve --port 8030
# Open http://127.0.0.1:8030
```

---

## Repository Structure

```text
axiom-zero/
├── csrc/
│   └── axiom_kernels.cpp              # C++20 fused softmax, speculative rejection & paged KV attention
├── src/axiom_zero/
│   ├── kernels/native_bridge.py       # SHA-256 cached clang++/g++ -O3 compiler & bit-exact NumPy fallback
│   ├── transformer/
│   │   ├── gqa_model.py               # Pre-Norm RoPE + GQA + SwiGLU Transformer & CoW PagedKVCacheManager
│   │   └── speculative.py             # Leviathan-Kalman-Matias speculative engine & Monte Carlo TV verifier
│   ├── interp/sae.py                  # Overcomplete Top-K SAE (||d_i||_2=1) & residual-stream steering hooks
│   ├── verifier/cdcl_smt.py           # 1-UIP CDCL SAT solver, standalone RUP checker, MUC & Fraction QF_LRA
│   ├── mcts/proof_mcts.py             # AlphaZero PUCT MCTS with real-time CDCL/QF_LRA subtree pruning
│   ├── real_llm_eval.py               # Live Ollama (qwen3.5:4b) CEGIS formal verification benchmark
│   ├── api/app.py                     # FastAPI server for live SAE alpha steering, speculative & CDCL RPCs
│   └── cli.py                         # Rich CLI (`axiom`)
├── benchmarks/real_qwen_eval.json     # Real recorded qwen3.5:4b Pass@1 vs. CEGIS trace
├── ui/                                # TypeScript + React + Vite Frontier Research Workbench
├── docs/
│   ├── mathematical-proofs.md         # Formal proofs for Theorems 1–4 linked to Hypothesis tests
│   └── adr/                           # Architecture Decision Records (0001–0004)
└── tests/                             # 98.0% coverage + Hypothesis randomized truth-table & TV proofs
```

## Documentation & Mathematical Foundations

- [Formal Mathematical Proofs (`docs/mathematical-proofs.md`)](docs/mathematical-proofs.md)
- [ADR-0001: C++20 Lossless Speculative Verification & CoW Paged KV](docs/adr/0001-cpp20-lossless-speculative-and-paged-kv.md)
- [ADR-0002: Overcomplete Top-K SAE & Unit-Norm Causal Steering](docs/adr/0002-topk-sae-unit-norm-causal-steering.md)
- [ADR-0003: 1-UIP CDCL & Independent RUP Proof Kernel](docs/adr/0003-1uip-cdcl-and-independent-rup-kernel.md)
- [ADR-0004: Zero-Mock CEGIS Loop Over Live Local LLMs](docs/adr/0004-cegis-real-llm-formal-gate.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Author & Enterprise AI Architecture Hub
Built by **[Nagarjuna Reddy Kamireddy](https://nagarjunak-pixel.github.io/)** (*Nagarjuna Kamireddy / Nagarjuna K*) — Lead AI Architect, GenAI & Agentic AI Lead, and Corporate AI Trainer (20+ years across Enterprise Data, SAP HANA & Production AI; AWS Certified Solutions Architect – Professional & NVIDIA-Certified Professional: Agentic AI).

- **Official Portfolio & Architecture Hub:** [https://nagarjunak-pixel.github.io/](https://nagarjunak-pixel.github.io/)
- **Enterprise Case Studies & Technical Articles:** [Case Studies](https://nagarjunak-pixel.github.io/case-studies.html) · [Architecture Articles](https://nagarjunak-pixel.github.io/articles.html) · [10-Mandate Role Fit](https://nagarjunak-pixel.github.io/roles.html)
- **Master CV & AI Search Index:** [Master CV](https://nagarjunak-pixel.github.io/resume.html) · [`llms-full.txt`](https://nagarjunak-pixel.github.io/llms-full.txt) · [LinkedIn](https://www.linkedin.com/in/nagarjuna-kamireddy)
