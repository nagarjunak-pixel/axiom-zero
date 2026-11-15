// AxiomZero Native C++20 Inference & Speculative Verification Kernels
// Implements:
//   1. Numerically stable row-wise Softmax
//   2. Leviathan-Kalman-Matias (ICML 2023) Exact Lossless Speculative Rejection Sampling
//   3. Kwon et al. (SOSP 2023) Block-Table Paged KV-Cache Single-Query Attention

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

void axiom_fused_softmax_f32(
    const float* logits,
    float* probs,
    int32_t rows,
    int32_t cols
) {
    for (int32_t r = 0; r < rows; ++r) {
        const float* row_in = logits + static_cast<size_t>(r) * cols;
        float* row_out = probs + static_cast<size_t>(r) * cols;

        float max_val = row_in[0];
        for (int32_t c = 1; c < cols; ++c) {
            if (row_in[c] > max_val) {
                max_val = row_in[c];
            }
        }

        double sum = 0.0;
        for (int32_t c = 0; c < cols; ++c) {
            const float e = std::exp(row_in[c] - max_val);
            row_out[c] = e;
            sum += static_cast<double>(e);
        }

        const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : (1.0f / cols);
        for (int32_t c = 0; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

static int32_t sample_categorical_cdf(const float* dist, int32_t vocab_size, float u) {
    const float clamped = std::clamp(u, 0.0f, 0.9999999f);
    double cdf = 0.0;
    int32_t last_nonzero = vocab_size - 1;
    for (int32_t v = 0; v < vocab_size; ++v) {
        const float p = dist[v];
        if (p > 0.0f) {
            last_nonzero = v;
            cdf += static_cast<double>(p);
            if (static_cast<double>(clamped) < cdf) {
                return v;
            }
        }
    }
    return last_nonzero;
}

int32_t axiom_speculative_verify_step(
    const float* target_probs,   // shape: (K + 1, V)
    const float* draft_probs,    // shape: (K, V)
    const int32_t* draft_tokens, // shape: (K,)
    const float* uniform_r,      // shape: (K,) in [0, 1)
    float resample_u,            // scalar in [0, 1)
    int32_t k_draft,
    int32_t vocab_size,
    int32_t* out_tokens          // shape: (K + 1,)
) {
    int32_t emitted = 0;

    for (int32_t t = 0; t < k_draft; ++t) {
        const int32_t tok = draft_tokens[t];
        const float* p_row = target_probs + static_cast<size_t>(t) * vocab_size;
        const float* q_row = draft_probs + static_cast<size_t>(t) * vocab_size;

        const float p_x = p_row[tok];
        const float q_x = std::max(q_row[tok], 1e-12f);
        const float accept_prob = std::min(1.0f, p_x / q_x);

        if (uniform_r[t] < accept_prob) {
            out_tokens[emitted++] = tok;
        } else {
            // Sample correction token from normalized max(0, p_t - q_t)
            std::vector<float> residual(static_cast<size_t>(vocab_size), 0.0f);
            double mass = 0.0;
            for (int32_t v = 0; v < vocab_size; ++v) {
                const float diff = p_row[v] - q_row[v];
                if (diff > 0.0f) {
                    residual[static_cast<size_t>(v)] = diff;
                    mass += static_cast<double>(diff);
                }
            }
            if (mass > 1e-12) {
                const float inv_mass = static_cast<float>(1.0 / mass);
                for (int32_t v = 0; v < vocab_size; ++v) {
                    residual[static_cast<size_t>(v)] *= inv_mass;
                }
                out_tokens[emitted++] = sample_categorical_cdf(residual.data(), vocab_size, resample_u);
            } else {
                out_tokens[emitted++] = sample_categorical_cdf(p_row, vocab_size, resample_u);
            }
            return emitted;
        }
    }

    // All K draft tokens accepted: sample bonus (K+1)-th token directly from target_probs[K]
    const float* bonus_row = target_probs + static_cast<size_t>(k_draft) * vocab_size;
    out_tokens[emitted++] = sample_categorical_cdf(bonus_row, vocab_size, resample_u);
    return emitted;
}

void axiom_paged_kv_attention_f32(
    const float* query,            // shape: (head_dim,)
    const float* key_blocks,       // shape: (num_blocks, block_size, head_dim)
    const float* value_blocks,     // shape: (num_blocks, block_size, head_dim)
    const int32_t* block_table,    // shape: (num_logical_blocks,)
    int32_t context_len,
    int32_t block_size,
    int32_t head_dim,
    float scale,
    float* out_context             // shape: (head_dim,)
) {
    if (context_len <= 0) {
        for (int32_t d = 0; d < head_dim; ++d) {
            out_context[d] = 0.0f;
        }
        return;
    }

    std::vector<float> scores(static_cast<size_t>(context_len), 0.0f);
    float max_score = -1e30f;

    for (int32_t pos = 0; pos < context_len; ++pos) {
        const int32_t logical_blk = pos / block_size;
        const int32_t slot = pos % block_size;
        const int32_t phys_blk = block_table[logical_blk];
        const float* k_vec = key_blocks +
            (static_cast<size_t>(phys_blk) * block_size + static_cast<size_t>(slot)) * head_dim;

        float dot = 0.0f;
        for (int32_t d = 0; d < head_dim; ++d) {
            dot += query[d] * k_vec[d];
        }
        const float s = dot * scale;
        scores[static_cast<size_t>(pos)] = s;
        if (s > max_score) {
            max_score = s;
        }
    }

    double denom = 0.0;
    for (int32_t pos = 0; pos < context_len; ++pos) {
        const float w = std::exp(scores[static_cast<size_t>(pos)] - max_score);
        scores[static_cast<size_t>(pos)] = w;
        denom += static_cast<double>(w);
    }
    const float inv_denom = (denom > 0.0) ? static_cast<float>(1.0 / denom) : 0.0f;

    for (int32_t d = 0; d < head_dim; ++d) {
        out_context[d] = 0.0f;
    }

    for (int32_t pos = 0; pos < context_len; ++pos) {
        const float weight = scores[static_cast<size_t>(pos)] * inv_denom;
        const int32_t logical_blk = pos / block_size;
        const int32_t slot = pos % block_size;
        const int32_t phys_blk = block_table[logical_blk];
        const float* v_vec = value_blocks +
            (static_cast<size_t>(phys_blk) * block_size + static_cast<size_t>(slot)) * head_dim;

        for (int32_t d = 0; d < head_dim; ++d) {
            out_context[d] += weight * v_vec[d];
        }
    }
}

} // extern "C"
