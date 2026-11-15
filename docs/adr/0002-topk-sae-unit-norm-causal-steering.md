# ADR 0002: Overcomplete Top-K Sparse Autoencoder with Unit-Norm Column Projection

- **Status**: Accepted
- **Context**: Dense Transformer residual streams suffer from polysemantic superposition, making prompt-level guardrails brittle against covert policy-bypass prefixes. Standard $L_1$-penalized SAEs suffer from activation shrinkage and hyperparameter instability.
- **Decision**: Train an overcomplete `TopKSparseAutoencoder` (`expansion_factor=6`, `dict_size=576`, `top_k=6`) directly on `GQATransformerLM` residual-stream activations with exact post-step L2 column normalization (`||d_i||_2 == 1`) and decision-position selectivity ranking.
- **Consequences**: Achieves `R^2 = 0.9855` variance explained at exact sparsity `L_0 = 6.00`, isolating monosemantic latents that fire `100%` in-regime and `0.000` out-of-regime, and allowing live additive steering (`h' = h + alpha * d_f`) to shift an adversarial `COVERT_POLICY_BYPASS` sequence into `RISK_QUARANTINE` (`0.00% -> 99.96%` at `alpha = 45`).
