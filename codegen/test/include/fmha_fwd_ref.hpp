// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_fwd {

struct FmhaFwdRefParams
{
    std::size_t batch;
    std::size_t nhead;
    std::size_t M; // seqlen_q
    std::size_t N; // seqlen_k
    std::size_t K; // hdim_q
    std::size_t O; // hdim_v

    float scale_s;

    std::size_t q_stride_batch;
    std::size_t q_stride_nhead;
    std::size_t q_stride_m;

    std::size_t k_stride_batch;
    std::size_t k_stride_nhead;
    std::size_t k_stride_n;

    std::size_t v_stride_batch;
    std::size_t v_stride_nhead;
    std::size_t v_stride_n;

    std::size_t o_stride_batch;
    std::size_t o_stride_nhead;
    std::size_t o_stride_m;

    std::size_t bias_stride_batch = 0;
    std::size_t bias_stride_nhead = 0;
    std::size_t bias_stride_m     = 0;
};

// O = softmax(Q @ K^T * scale_s + bias) @ V
// bias is optional (nullptr = no bias)
inline void cpu_attention_ref(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              std::vector<float>& o,
                              const std::vector<float>* bias,
                              const FmhaFwdRefParams& p)
{
    for(std::size_t b = 0; b < p.batch; ++b)
    {
        for(std::size_t h = 0; h < p.nhead; ++h)
        {
            const float* q_ptr = q.data() + b * p.q_stride_batch + h * p.q_stride_nhead;
            const float* k_ptr = k.data() + b * p.k_stride_batch + h * p.k_stride_nhead;
            const float* v_ptr = v.data() + b * p.v_stride_batch + h * p.v_stride_nhead;
            const float* bias_ptr =
                bias ? (bias->data() + b * p.bias_stride_batch + h * p.bias_stride_nhead) : nullptr;
            float* o_ptr = o.data() + b * p.o_stride_batch + h * p.o_stride_nhead;

            for(std::size_t m = 0; m < p.M; ++m)
            {
                // Q[m,:] @ K^T -> [N]
                std::vector<float> scores(p.N);
                for(std::size_t n = 0; n < p.N; ++n)
                {
                    float dot = 0.0f;
                    for(std::size_t kk = 0; kk < p.K; ++kk)
                    {
                        dot += q_ptr[m * p.q_stride_m + kk] * k_ptr[n * p.k_stride_n + kk];
                    }
                    scores[n] = dot * p.scale_s;

                    if(bias_ptr)
                    {
                        scores[n] += bias_ptr[m * p.bias_stride_m + n];
                    }
                }

                // Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp   = 0.0f;
                for(std::size_t n = 0; n < p.N; ++n)
                {
                    scores[n] = std::exp(scores[n] - max_score);
                    sum_exp += scores[n];
                }
                for(std::size_t n = 0; n < p.N; ++n)
                {
                    scores[n] /= sum_exp;
                }

                // Output: attn @ V -> [O]
                for(std::size_t oo = 0; oo < p.O; ++oo)
                {
                    float val = 0.0f;
                    for(std::size_t n = 0; n < p.N; ++n)
                    {
                        val += scores[n] * v_ptr[n * p.v_stride_n + oo];
                    }
                    o_ptr[m * p.o_stride_m + oo] = val;
                }
            }
        }
    }
}

inline void cpu_attention_ref(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              std::vector<float>& o,
                              const FmhaFwdRefParams& p)
{
    cpu_attention_ref(q, k, v, o, nullptr, p);
}

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
