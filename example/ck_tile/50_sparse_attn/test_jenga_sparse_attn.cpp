// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Test for jenga_sparse_attention function

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"

#include "jenga_sparse_attention.h"

// ============================================================================
// Helper Functions
// ============================================================================

// Reference implementation: blocked attention
template <typename T, typename AccT = float>
void reference_blocked_attention(
    const ck_tile::HostTensor<T>& q,              // [B, H, S_q, D]
    const ck_tile::HostTensor<T>& k,              // [B, H, S_k, D]
    const ck_tile::HostTensor<T>& v,              // [B, H, S_k, D_v]
    const ck_tile::HostTensor<T>& block_relation, // [B, H, Q_blocks, K_blocks]
    const ck_tile::HostTensor<T>& bias,           // [B, H, S_q, S_k]
    ck_tile::HostTensor<T>& output,               // [B, H, S_q, D_v]
    ck_tile::index_t BLKQ,
    ck_tile::index_t BLKK,
    AccT scale)
{
    auto q_lengths            = q.get_lengths();
    ck_tile::index_t batch    = q_lengths[0];
    ck_tile::index_t nhead    = q_lengths[1];
    ck_tile::index_t seqlen_q = q_lengths[2];
    ck_tile::index_t hdim     = q_lengths[3];

    auto v_lengths            = v.get_lengths();
    ck_tile::index_t seqlen_k = v_lengths[2];
    ck_tile::index_t hdim_v   = v_lengths[3];

    ck_tile::index_t num_q_blocks = seqlen_q / BLKQ;
    ck_tile::index_t num_k_blocks = seqlen_k / BLKK;

    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
        {
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                ck_tile::index_t q_start = qb * BLKQ;
                ck_tile::index_t q_end   = q_start + BLKQ;

                // Find relevant K blocks
                std::vector<ck_tile::index_t> relevant_k_indices;
                for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    if(static_cast<float>(block_relation(b, h, qb, kb)) > 0.5f)
                    {
                        relevant_k_indices.push_back(kb);
                    }
                }

                if(relevant_k_indices.empty())
                    continue;

                // For each query position in the block
                for(ck_tile::index_t sq = q_start; sq < q_end; ++sq)
                {
                    std::vector<AccT> scores;
                    AccT max_score = -std::numeric_limits<AccT>::infinity();

                    for(auto kb : relevant_k_indices)
                    {
                        ck_tile::index_t k_start = kb * BLKK;
                        ck_tile::index_t k_end   = k_start + BLKK;

                        for(ck_tile::index_t sk = k_start; sk < k_end; ++sk)
                        {
                            AccT score = 0.0f;
                            for(ck_tile::index_t d = 0; d < hdim; ++d)
                            {
                                score += static_cast<AccT>(q(b, h, sq, d)) *
                                         static_cast<AccT>(k(b, h, sk, d));
                            }
                            score = score * scale + static_cast<AccT>(bias(b, h, sq, sk));
                            scores.push_back(score);
                            max_score = std::max(max_score, score);
                        }
                    }

                    // Softmax
                    AccT sum_exp = 0.0f;
                    for(auto& s : scores)
                    {
                        s = std::exp(s - max_score);
                        sum_exp += s;
                    }
                    for(auto& s : scores)
                    {
                        s /= sum_exp;
                    }

                    // Compute output: P @ V
                    for(ck_tile::index_t dv = 0; dv < hdim_v; ++dv)
                    {
                        AccT out_val     = 0.0f;
                        size_t score_idx = 0;

                        for(auto kb : relevant_k_indices)
                        {
                            ck_tile::index_t k_start = kb * BLKK;
                            ck_tile::index_t k_end   = k_start + BLKK;

                            for(ck_tile::index_t sk = k_start; sk < k_end; ++sk)
                            {
                                out_val += scores[score_idx] * static_cast<AccT>(v(b, h, sk, dv));
                                score_idx++;
                            }
                        }
                        output(b, h, sq, dv) = static_cast<T>(out_val);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Command line argument parser
// ============================================================================
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "0:no validation, 1:cpu validation")
        .insert("mode", "0", "kernel mode. 0:batch, 1:group")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head for q")
        .insert("h_k", "-1", "num of head for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("block_size", "128", "block size for sparse attention (BLKQ=BLKK)")
        .insert("sparsity", "0.5", "sparsity ratio (0.0 = dense, 1.0 = fully sparse)")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("bias", "0", "bias type: 0:no bias, 1:elementwise, 2:alibi")
        .insert("lse", "0", "0:not store lse, 1:store lse")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ============================================================================
// Main Test Function
// ============================================================================
bool run_test(const ck_tile::ArgParser& arg_parser)
{
    using T = DataType; // Use DataType defined in header (half_t)

    // Parse arguments
    int do_validation           = arg_parser.get_int("v");
    int mode                    = arg_parser.get_int("mode");
    ck_tile::index_t batch      = arg_parser.get_int("b");
    ck_tile::index_t nhead      = arg_parser.get_int("h");
    ck_tile::index_t nhead_k    = arg_parser.get_int("h_k");
    ck_tile::index_t seqlen_q   = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k   = arg_parser.get_int("s_k");
    ck_tile::index_t hdim_q     = arg_parser.get_int("d");
    ck_tile::index_t hdim_v     = arg_parser.get_int("d_v");
    ck_tile::index_t block_size = arg_parser.get_int("block_size");
    float sparsity              = arg_parser.get_float("sparsity");
    bool i_perm                 = arg_parser.get_bool("iperm");
    bool o_perm                 = arg_parser.get_bool("operm");
    int bias_type               = arg_parser.get_int("bias");
    bool store_lse              = arg_parser.get_bool("lse");
    uint32_t seed               = arg_parser.get_uint32("seed");
    int warmup                  = arg_parser.get_int("warmup");
    int repeat                  = arg_parser.get_int("repeat");

    // Handle default values
    if(nhead_k < 0)
        nhead_k = nhead;
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    if(hdim_v < 0)
        hdim_v = hdim_q;

    ck_tile::index_t BLKQ = block_size;
    ck_tile::index_t BLKK = block_size;

    // Calculate number of Q and K blocks
    ck_tile::index_t num_q_blocks = seqlen_q / BLKQ;
    ck_tile::index_t num_k_blocks = seqlen_k / BLKK;

    std::cout << "============================================================" << std::endl;
    std::cout << "[Jenga Sparse Attention Test]" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Batch: " << batch << ", nhead_q: " << nhead << ", nhead_k: " << nhead_k
              << std::endl;
    std::cout << "  seqlen_q: " << seqlen_q << ", seqlen_k: " << seqlen_k << std::endl;
    std::cout << "  hdim_q: " << hdim_q << ", hdim_v: " << hdim_v << std::endl;
    std::cout << "  block_size: " << block_size << " (BLKQ=" << BLKQ << ", BLKK=" << BLKK << ")"
              << std::endl;
    std::cout << "  num_q_blocks: " << num_q_blocks << ", num_k_blocks: " << num_k_blocks
              << std::endl;
    std::cout << "  sparsity: " << sparsity << std::endl;
    std::cout << "  i_perm: " << i_perm << ", o_perm: " << o_perm << std::endl;

    // Create host tensors (using BHSD layout when i_perm=true)
    ck_tile::HostTensor<T> q_host({batch, nhead, seqlen_q, hdim_q});
    ck_tile::HostTensor<T> k_host({batch, nhead_k, seqlen_k, hdim_q});
    ck_tile::HostTensor<T> v_host({batch, nhead_k, seqlen_k, hdim_v});
    ck_tile::HostTensor<T> output_host({batch, nhead, seqlen_q, hdim_v});
    ck_tile::HostTensor<T> output_ref({batch, nhead, seqlen_q, hdim_v});

    // Bias tensor [B, H, S_q, S_k]
    ck_tile::HostTensor<T> bias_host({batch, nhead, seqlen_q, seqlen_k});

    // Block relation onehot: [B, H, Q_blocks, K_blocks]
    ck_tile::HostTensor<T> block_relation_onehot({batch, nhead, num_q_blocks, num_k_blocks});

    // LSE tensor (optional)
    ck_tile::HostTensor<T> lse_host({batch, nhead, seqlen_q});

    // Initialize tensors with random values
    std::cout << "\nInitializing tensors..." << std::endl;
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

    // Initialize bias to zero
    std::fill(bias_host.mData.begin(), bias_host.mData.end(), static_cast<T>(0.0f));

    // Initialize block_relation_onehot with sparse pattern
    std::mt19937 rng(seed + 100);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    ck_tile::index_t total_blocks  = 0;
    ck_tile::index_t active_blocks = 0;

    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
        {
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    total_blocks++;
                    bool is_diagonal   = (qb == kb && qb < num_k_blocks);
                    bool random_active = (dist(rng) > sparsity);

                    if(is_diagonal || random_active)
                    {
                        block_relation_onehot(b, h, qb, kb) = static_cast<T>(1.0f);
                        active_blocks++;
                    }
                    else
                    {
                        block_relation_onehot(b, h, qb, kb) = static_cast<T>(0.0f);
                    }
                }
            }
        }
    }

    float actual_sparsity =
        1.0f - static_cast<float>(active_blocks) / static_cast<float>(total_blocks);
    std::cout << "  Actual sparsity: " << actual_sparsity << " (" << active_blocks << "/"
              << total_blocks << " blocks active)" << std::endl;

    // Optional tensors
    std::optional<ck_tile::HostTensor<T>> bias_opt       = std::nullopt;
    std::optional<ck_tile::HostTensor<T>> lse_opt        = std::nullopt;
    std::optional<ck_tile::HostTensor<T>> seqstart_q_opt = std::nullopt;
    std::optional<ck_tile::HostTensor<T>> seqstart_k_opt = std::nullopt;

    if(bias_type != 0)
    {
        bias_opt = bias_host;
    }
    if(store_lse)
    {
        lse_opt = lse_host;
    }

    // Run kernel
    std::cout << "\n--- Running Jenga sparse attention kernel ---" << std::endl;

    try
    {
        // Warmup
        for(int i = 0; i < warmup; ++i)
        {
            jenga_sparse_attention(q_host,
                                   k_host,
                                   v_host,
                                   block_relation_onehot,
                                   output_host,
                                   bias_opt,
                                   lse_opt,
                                   seqstart_q_opt,
                                   seqstart_k_opt,
                                   bias_type,
                                   batch,
                                   nhead,
                                   nhead_k,
                                   seqlen_q,
                                   seqlen_k,
                                   hdim_q,
                                   hdim_v,
                                   mode,
                                   i_perm,
                                   o_perm,
                                   seqlen_q,
                                   seqlen_k);
        }

        // Benchmark
        [[maybe_unused]] auto sync_status1 = hipDeviceSynchronize();
        auto start                         = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < repeat; ++i)
        {
            jenga_sparse_attention(q_host,
                                   k_host,
                                   v_host,
                                   block_relation_onehot,
                                   output_host,
                                   bias_opt,
                                   lse_opt,
                                   seqstart_q_opt,
                                   seqstart_k_opt,
                                   bias_type,
                                   batch,
                                   nhead,
                                   nhead_k,
                                   seqlen_q,
                                   seqlen_k,
                                   hdim_q,
                                   hdim_v,
                                   mode,
                                   i_perm,
                                   o_perm,
                                   seqlen_q,
                                   seqlen_k);
        }

        [[maybe_unused]] auto sync_status2 = hipDeviceSynchronize();
        auto end                           = std::chrono::high_resolution_clock::now();
        double avg_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count() / repeat;

        std::cout << "\n>>>> Jenga sparse attention average time: " << avg_time_ms << " ms <<<<"
                  << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error during kernel execution: " << e.what() << std::endl;
        return false;
    }

    // Validation
    bool pass = true;
    if(do_validation)
    {
        std::cout << "\n--- Performing CPU validation ---" << std::endl;

        float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

        std::cout << "Computing reference output..." << std::endl;
        reference_blocked_attention(q_host,
                                    k_host,
                                    v_host,
                                    block_relation_onehot,
                                    bias_host,
                                    output_ref,
                                    BLKQ,
                                    BLKK,
                                    scale);

        // Compare results
        double rtol = 1e-2;
        double atol = 4e-2;

        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t num_errors  = 0;

        for(size_t i = 0; i < output_host.mData.size(); ++i)
        {
            float gpu_val  = static_cast<float>(output_host.mData[i]);
            float ref_val  = static_cast<float>(output_ref.mData[i]);
            float diff     = std::abs(gpu_val - ref_val);
            float rel_diff = (std::abs(ref_val) > 1e-6f) ? diff / std::abs(ref_val) : diff;

            max_diff     = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if(diff > atol && rel_diff > rtol)
            {
                num_errors++;
                if(num_errors <= 5)
                {
                    std::cout << "  Mismatch at index " << i << ": GPU=" << gpu_val
                              << ", Ref=" << ref_val << ", Diff=" << diff << std::endl;
                }
            }
        }

        std::cout << "\nValidation results:" << std::endl;
        std::cout << "  Max absolute difference: " << max_diff << std::endl;
        std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
        std::cout << "  Number of mismatches: " << num_errors << " / " << output_host.mData.size()
                  << std::endl;

        if(num_errors == 0)
        {
            std::cout << "\n>>> VALIDATION PASSED <<<" << std::endl;
        }
        else
        {
            std::cout << "\n>>> VALIDATION FAILED <<<" << std::endl;
            pass = false;
        }
    }

    std::cout << "\n" << (pass ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Failed to parse arguments" << std::endl;
        return -1;
    }

    bool test_result = run_test(arg_parser);
    return test_result ? 0 : -1;
}
