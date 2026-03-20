#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <cassert>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace sparge {

struct SpargeParams
{
    int BLKQ = 128;
    int BLKK = 128;

    // Similarity gate threshold (TODO: per-head support).
    float simthreshd1 = 0.6f;

    // Exactly one of the following should be used:
    // - Use CDF threshold if topk < 0
    // - Both should be in [0, 1] <-- NEED TO CHECK THIS
    float cdfthreshd = 0.98f;
    float topk       = -1.0f;

    // If true, treat Q/K as BHSD; otherwise BSHD (same convention as CK examples).
    bool i_perm = true;
};

// Output format CK VSA expects.
struct VSALut
{
    ck_tile::HostTensor<int32_t> lut;             // [B, Hq, Q_blk, K_blk] delta-encoded
    ck_tile::HostTensor<int32_t> valid_block_num; // [B, Hq, Q_blk]
};

namespace detail {

template <typename T>
inline float to_f32(const T& x)
{
    return ck_tile::type_convert<float>(x);
}

// Read element from HostTensor with either BHSD or BSHD layout.
// Q: [B, Hq, Sq, D] if i_perm else [B, Sq, Hq, D]
// K: [B, Hk, Sk, D] if i_perm else [B, Sk, Hk, D]
template <typename T>
inline float load(const ck_tile::HostTensor<T>& X, bool i_perm, int b, int h, int s, int d)
{
    return i_perm ? to_f32(X(b, h, s, d)) : to_f32(X(b, s, h, d));
}

// Compute pooled mean vector of one block: mean over tokens in [s0, s1).
template <typename T>
std::vector<float>
pooled_mean_block(const ck_tile::HostTensor<T>& X, bool i_perm, int b, int h, int s0, int s1, int d)
{
    std::vector<float> mean(d, 0.0f);
    const int bs = std::max(0, s1 - s0);
    if(bs == 0)
        return mean;

    for(int s = s0; s < s1; ++s)
    {
        for(int d_ = 0; d_ < d; ++d_)
        {
            mean[d_] += load(X, i_perm, b, h, s, d_);
        }
    }
    const float inv = 1.0f / static_cast<float>(bs);
    for(int d_ = 0; d_ < d; ++d_)
        mean[d_] *= inv;
    return mean;
}

// Compute "sim" flag of one block following SpargeAttn's intent:
// mean_sim = sum(Gram(x_hat)) / (BS_*BS_), where x_hat are token vectors normalized along D.
//
// Important: sum(Gram) = ||sum_i x_hat_i||^2, so we can compute it in O(BS_*D) exactly
// instead of O(BS_^2 * D).
template <typename T>
bool sim_block_flag(const ck_tile::HostTensor<T>& X,
                    bool i_perm,
                    int b,
                    int h,
                    int s0,
                    int s1,
                    int d,
                    float simthreshd1)
{
    const int bs = std::max(0, s1 - s0);
    if(bs == 0)
        return false;

    std::vector<float> sum_hat(d, 0.0f);

    for(int s = s0; s < s1; ++s)
    {
        // Compute L2 norm over D.
        float norm2 = 0.0f;
        for(int d_ = 0; d_ < d; ++d_)
        {
            const float v = load(X, i_perm, b, h, s, d_);
            norm2 += v * v;
        }
        float inv_norm = 1.0f;
        // spargeAttn use eps to prevent division by zero
        if(norm2 > 0.0f)
            inv_norm = 1.0f / std::sqrt(norm2);

        // Accumulate normalized vector.
        for(int d_ = 0; d_ < d; ++d_)
        {
            sum_hat[d_] += load(X, i_perm, b, h, s, d_) * inv_norm;
        }
    }

    float sum_gram = 0.0f;
    for(int d_ = 0; d_ < d; ++d_)
        sum_gram += sum_hat[d_] * sum_hat[d_];

    const float denom    = static_cast<float>(bs) * static_cast<float>(bs);
    const float mean_sim = sum_gram / denom;

    return mean_sim > simthreshd1;
}

inline int select_count_from_cdf(const std::vector<float>& sorted_probs, float cdfthreshd)
{
    // Choose the smallest n such that cdf[n-1] >= cdfthreshd.
    // Ensure at least 1.
    if(sorted_probs.empty())
        return 0;
    if(cdfthreshd <= 0.0f)
        return 1;

    float c = 0.0f;
    for(int i = 0; i < static_cast<int>(sorted_probs.size()); ++i)
    {
        c += sorted_probs[i];
        if(c >= cdfthreshd)
            return i + 1;
    }
    return static_cast<int>(sorted_probs.size());
}

inline int select_count_from_topk(int K_blk, float topk)
{
    if(K_blk <= 0)
        return 0;
    int n = static_cast<int>(std::floor(topk * static_cast<float>(K_blk)));
    n     = std::max(1, n);
    return n;
}

} // namespace detail

// Build one-hot block_map[b,hq,qb,kb] in {0,1}.
// - No causal mask
// - No attention sink
// - Logic matches SpargeAttn's structure:
//   - score softmax is only over sim_kblocks; ~sim_kblocks are forced ON later
//   - if a Q-block is not "similar", force the whole row ON
template <typename T>
ck_tile::HostTensor<uint8_t> build_block_map_meansim(const ck_tile::HostTensor<T>& Q,
                                                     const ck_tile::HostTensor<T>& K,
                                                     const SpargeParams& p)
{
    const auto qlens = Q.get_lengths();
    const auto klens = K.get_lengths();

    const int B  = static_cast<int>(qlens[0]);
    const int Hq = p.i_perm ? static_cast<int>(qlens[1]) : static_cast<int>(qlens[2]);
    const int Sq = p.i_perm ? static_cast<int>(qlens[2]) : static_cast<int>(qlens[1]);
    const int D  = static_cast<int>(qlens[3]);

    [[maybe_unused]] const int Bk = static_cast<int>(klens[0]);
    const int Hk = p.i_perm ? static_cast<int>(klens[1]) : static_cast<int>(klens[2]);
    const int Sk = p.i_perm ? static_cast<int>(klens[2]) : static_cast<int>(klens[1]);
    [[maybe_unused]] const int Dk = static_cast<int>(klens[3]);

    assert(B == Bk && D == Dk && Hq % Hk == 0);
    assert(p.BLKQ > 0 && p.BLKK > 0);

    const int nhead_ratio_qk = Hq / Hk;
    const int Q_blk          = ck_tile::integer_divide_ceil(Sq, p.BLKQ);
    const int K_blk          = ck_tile::integer_divide_ceil(Sk, p.BLKK);

    ck_tile::HostTensor<uint8_t> block_map({B, Hq, Q_blk, K_blk});

    // pooled_q: [B,Hq,Q_blk,D], pooled_k: [B,Hk,K_blk,D]
    // sim_q: [B,Hq,Q_blk], sim_k: [B,Hk,K_blk]
    std::vector<float> pooled_q(static_cast<size_t>(B) * Hq * Q_blk * D, 0.0f);
    std::vector<float> pooled_k(static_cast<size_t>(B) * Hk * K_blk * D, 0.0f);
    std::vector<uint8_t> sim_q(static_cast<size_t>(B) * Hq * Q_blk, 0);
    std::vector<uint8_t> sim_k(static_cast<size_t>(B) * Hk * K_blk, 0);

    auto idx_pq = [&](int b, int hq, int qb, int d) {
        return (((b * Hq + hq) * Q_blk + qb) * D + d);
    };
    auto idx_pk = [&](int b, int hk, int kb, int d) {
        return (((b * Hk + hk) * K_blk + kb) * D + d);
    };
    auto idx_sq = [&](int b, int hq, int qb) { return ((b * Hq + hq) * Q_blk + qb); };
    auto idx_sk = [&](int b, int hk, int kb) { return ((b * Hk + hk) * K_blk + kb); };

    for(int b = 0; b < B; ++b)
    {
        for(int hq = 0; hq < Hq; ++hq)
        {
            // Q blocks
            for(int qb = 0; qb < Q_blk; ++qb)
            {
                const int s0 = qb * p.BLKQ;
                const int s1 = std::min(Sq, (qb + 1) * p.BLKQ);

                // pooled mean
                auto mean = detail::pooled_mean_block(Q, p.i_perm, b, hq, s0, s1, D);
                for(int d = 0; d < D; ++d)
                    pooled_q[idx_pq(b, hq, qb, d)] = mean[d];

                // sim flag
                sim_q[idx_sq(b, hq, qb)] =
                    detail::sim_block_flag(Q, p.i_perm, b, hq, s0, s1, D, p.simthreshd1) ? 1 : 0;
            }
        }

        for(int hk = 0; hk < Hk; ++hk)
        {
            // K blocks
            for(int kb = 0; kb < K_blk; ++kb)
            {
                const int s0 = kb * p.BLKK;
                const int s1 = std::min(Sk, (kb + 1) * p.BLKK);

                auto mean = detail::pooled_mean_block(K, p.i_perm, b, hk, s0, s1, D);
                for(int d = 0; d < D; ++d)
                    pooled_k[idx_pk(b, hk, kb, d)] = mean[d];

                sim_k[idx_sk(b, hk, kb)] =
                    detail::sim_block_flag(K, p.i_perm, b, hk, s0, s1, D, p.simthreshd1) ? 1 : 0;
            }
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Main loop
    for(int b = 0; b < B; ++b)
    {
        for(int hq = 0; hq < Hq; ++hq)
        {
            const int hk = hq / nhead_ratio_qk;

            for(int qb = 0; qb < Q_blk; ++qb)
            {
                const bool q_is_sim = (sim_q[idx_sq(b, hq, qb)] != 0);

                // If Q-block is not "similar", force dense row.
                if(!q_is_sim)
                {
                    for(int kb = 0; kb < K_blk; ++kb)
                        block_map(b, hq, qb, kb) = 1;
                    continue;
                }

                // Compute scores over K blocks (only sim_kblocks participate in softmax; others set
                // to -inf).
                std::vector<float> score(K_blk, -std::numeric_limits<float>::infinity());
                for(int kb = 0; kb < K_blk; ++kb)
                {
                    const bool k_is_sim = (sim_k[idx_sk(b, hk, kb)] != 0);
                    if(!k_is_sim)
                    {
                        block_map(b, hq, qb, kb) = 1;
                        continue;
                    }

                    float dot = 0.0f;
                    for(int d = 0; d < D; ++d)
                    {
                        dot += pooled_q[idx_pq(b, hq, qb, d)] * pooled_k[idx_pk(b, hk, kb, d)];
                    }
                    score[kb] = dot * scale;
                }

                // Softmax over K_blk (numerically stable). If all -inf, probs become all zeros.
                float maxv = -std::numeric_limits<float>::infinity();
                for(int kb = 0; kb < K_blk; ++kb)
                    maxv = std::max(maxv, score[kb]);

                std::vector<float> prob(K_blk, 0.0f);
                if(std::isfinite(maxv))
                {
                    float sumexp = 0.0f;
                    for(int kb = 0; kb < K_blk; ++kb)
                    {
                        if(!std::isfinite(score[kb]))
                            continue;
                        const float e = std::exp(score[kb] - maxv);
                        prob[kb]      = e;
                        sumexp += e;
                    }
                    if(sumexp > 0.0f)
                    {
                        const float inv = 1.0f / sumexp;
                        for(int kb = 0; kb < K_blk; ++kb)
                            prob[kb] *= inv;
                    }
                    else
                    {
                        // All exponentials underflowed: keep zeros.
                        std::fill(prob.begin(), prob.end(), 0.0f);
                    }
                }

                // Sort indices by prob descending.
                std::vector<int> order(K_blk);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int c) {
                    if(prob[a] != prob[c])
                        return prob[a] > prob[c];
                    return a < c; // tie-breaker for determinism
                });

                // Determine how many to select.
                int num_to_select = 0;
                if(p.topk > 0.0f)
                {
                    num_to_select = detail::select_count_from_topk(K_blk, p.topk);
                }
                else
                {
                    // Use CDF threshold selection (smallest n s.t. cumulative prob >= cdfthreshd).
                    std::vector<float> sorted_probs(K_blk);
                    for(int i = 0; i < K_blk; ++i)
                        sorted_probs[i] = prob[order[i]];
                    num_to_select = detail::select_count_from_cdf(sorted_probs, p.cdfthreshd);
                    num_to_select = std::max(1, num_to_select);
                }

                // Select top-kb blocks by order[0..num_to_select-1].
                for(int i = 0; i < num_to_select; ++i)
                {
                    const int kb             = order[i];
                    block_map(b, hq, qb, kb) = 1;
                }
            }
        }
    }

    return block_map;
}

// Convert one-hot block_map -> delta-encoded LUT + valid_block_num (CK VSA format).
template <typename MapT>
VSALut block_map_to_vsa_lut_delta(const ck_tile::HostTensor<MapT>& block_map)
{
    const auto lens = block_map.get_lengths();
    const int B     = static_cast<int>(lens[0]);
    const int H     = static_cast<int>(lens[1]);
    const int Q     = static_cast<int>(lens[2]);
    const int K     = static_cast<int>(lens[3]);

    VSALut out{
        ck_tile::HostTensor<int32_t>({B, H, Q, K}),
        ck_tile::HostTensor<int32_t>({B, H, Q}),
    };

    for(int b = 0; b < B; ++b)
    {
        for(int h = 0; h < H; ++h)
        {
            for(int q = 0; q < Q; ++q)
            {
                int32_t valid = 0;
                int32_t prev  = 0;

                for(int k = 0; k < K; ++k)
                {
                    const bool on = static_cast<int>(block_map(b, h, q, k)) != 0;
                    if(on)
                    {
                        out.lut(b, h, q, valid) = static_cast<int32_t>(k - prev);
                        prev                    = static_cast<int32_t>(k);
                        ++valid;
                    }
                }

                out.valid_block_num(b, h, q) = valid;

                // Optional: zero-fill the unused tail for determinism.
                for(int i = valid; i < K; ++i)
                    out.lut(b, h, q, i) = 0;
            }
        }
    }

    return out;
}

} // namespace sparge
