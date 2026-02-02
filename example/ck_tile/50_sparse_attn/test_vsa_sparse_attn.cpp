// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Test for vsa_sparse_attention function
// Based on the Python test: test_jenga_attention.py

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/reference/reference_blocked_attention.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include "jenga_sparse_attention.h"
#include "fmha_fwd_trek.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

template <typename T>
ck_tile::HostTensor<T> make_qkv_tensor(ck_tile::index_t batch,
                                       ck_tile::index_t nhead,
                                       ck_tile::index_t seqlen,
                                       ck_tile::index_t hdim,
                                       bool i_perm)
{
    if(i_perm)
    {
        return ck_tile::HostTensor<T>({batch, nhead, seqlen, hdim});
    }
    return ck_tile::HostTensor<T>({batch, seqlen, nhead, hdim});
}

template <typename T>
ck_tile::HostTensor<T> to_bhsd(const ck_tile::HostTensor<T>& tensor, bool is_bhsd)
{
    auto lens               = tensor.get_lengths();
    ck_tile::index_t batch  = lens[0];
    ck_tile::index_t seqlen = is_bhsd ? lens[2] : lens[1];
    ck_tile::index_t nhead  = is_bhsd ? lens[1] : lens[2];
    ck_tile::index_t hdim   = lens[3];

    ck_tile::HostTensor<T> out({batch, nhead, seqlen, hdim});
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
        {
            for(ck_tile::index_t s = 0; s < seqlen; ++s)
            {
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                {
                    out(b, h, s, d) = is_bhsd ? tensor(b, h, s, d) : tensor(b, s, h, d);
                }
            }
        }
    }
    return out;
}

// Convert block_relation_onehot to LUT format (similar to triton_block_map_to_lut_kernel)
template <typename T>
void block_map_to_lut(
    const ck_tile::HostTensor<T>& block_map, // [B, H, Q_blocks, K_blocks]
    ck_tile::HostTensor<int32_t>& lut,       // [B, H, Q_blocks, K_blocks] - int32_t for kernel
    ck_tile::HostTensor<int32_t>& valid_block_num, // [B, H, Q_blocks] - int32_t for kernel
    ck_tile::index_t num_block_k)
{
    auto lengths       = block_map.get_lengths();
    ck_tile::index_t B = lengths[0];
    ck_tile::index_t H = lengths[1];
    ck_tile::index_t Q = lengths[2];

    for(ck_tile::index_t b = 0; b < B; ++b)
    {
        for(ck_tile::index_t h = 0; h < H; ++h)
        {
            for(ck_tile::index_t q = 0; q < Q; ++q)
            {
                int32_t valid_count = 0;
                int32_t prev_block  = 0;

                for(ck_tile::index_t k = 0; k < num_block_k; ++k)
                {
                    T cur_block = block_map(b, h, q, k);
                    if(static_cast<float>(cur_block) > 0.5f)
                    { // Check if block is active
                        lut(b, h, q, valid_count) = static_cast<int32_t>(k - prev_block);
                        valid_count++;
                        prev_block = static_cast<int32_t>(k);
                    }
                }
                valid_block_num(b, h, q) = valid_count;
            }
        }
    }
}

// Get error tolerance based on data type
template <typename T>
auto get_error_tolerance()
{
    double rtol = 1e-2;
    double atol = 4e-2;
    if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
        // bf16 accumulation/rounding can be noisier in sparse patterns
        atol = 2e-1;
        rtol = 2e-1;
    }
    return ck_tile::make_tuple(rtol, atol);
}

template <typename T>
float to_float_for_compare(T value)
{
    return static_cast<float>(value);
}

template <>
float to_float_for_compare<ck_tile::bf16_t>(ck_tile::bf16_t value)
{
#if CK_TILE_USE_CUSTOM_DATA_TYPE
    return static_cast<float>(value);
#else
    return ck_tile::bf16_to_float_raw(ck_tile::bit_cast<ck_tile::bf16_raw_t>(value));
#endif
}

// ============================================================================
// Command line argument parser
// ============================================================================
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "0:no validation, 1:cpu validation")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head for q")
        .insert("h_k", "-1", "num of head for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("block_size", "128", "block size for sparse attention (BLKQ=BLKK)")
        .insert("sparsity", "0.5", "sparsity ratio (0.0 = dense, 1.0 = fully sparse)")
        .insert("prec", "fp16", "data type: fp16/bf16")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations")
        .insert("kname", "0", "print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ============================================================================
// Main Test Function
// ============================================================================
template <typename T>
bool run_test(const ck_tile::ArgParser& arg_parser)
{
    // Parse arguments
    int do_validation           = arg_parser.get_int("v");
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
    uint32_t seed               = arg_parser.get_uint32("seed");
    int warmup                  = arg_parser.get_int("warmup");
    int repeat                  = arg_parser.get_int("repeat");
    int kname                   = arg_parser.get_int("kname");

    // Handle default values
    if(nhead_k < 0)
        nhead_k = nhead;
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    if(hdim_v < 0)
        hdim_v = hdim_q;

    ck_tile::index_t BLKQ = block_size;
    ck_tile::index_t BLKK = block_size;

    if(block_size != 128 || hdim_q != 128 || hdim_v != 128)
    {
        std::cout << "\n>>> TEST SKIPPED <<<" << std::endl;
        std::cout << "VSA kernel instances are generated for block_size=128 and hdim=128 only."
                  << std::endl;
        std::cout << "TEST SKIPPED" << std::endl;
        return true;
    }

    // Calculate number of Q and K blocks
    ck_tile::index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    ck_tile::index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    std::cout << "============================================================" << std::endl;
    std::cout << "[VSA Sparse Attention Test]" << std::endl;
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
    // Q: [B, H, S_q, D]
    // K: [B, H_k, S_k, D]
    // V: [B, H_k, S_k, D_v]
    ck_tile::HostTensor<T> q_host = make_qkv_tensor<T>(batch, nhead, seqlen_q, hdim_q, i_perm);
    ck_tile::HostTensor<T> k_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_q, i_perm);
    ck_tile::HostTensor<T> v_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
    ck_tile::HostTensor<T> output_host =
        o_perm ? ck_tile::HostTensor<T>({batch, nhead, seqlen_q, hdim_v})
               : ck_tile::HostTensor<T>({batch, seqlen_q, nhead, hdim_v});
    ck_tile::HostTensor<T> output_ref({batch, nhead, seqlen_q, hdim_v});

    // Block relation onehot: [B, H, Q_blocks, K_blocks]
    ck_tile::HostTensor<uint8_t> block_relation_onehot({batch, nhead, num_q_blocks, num_k_blocks});

    // LUT and valid_block_num (output of block_map_to_lut) - must be int32_t for kernel
    ck_tile::HostTensor<int32_t> lut_host({batch, nhead, num_q_blocks, num_k_blocks});
    ck_tile::HostTensor<int32_t> valid_block_num_host({batch, nhead, num_q_blocks});

    // Initialize tensors with random values
    std::cout << "\nInitializing tensors..." << std::endl;
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

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
                    // Each Q block always attends to its diagonal K block (if exists)
                    // Plus random blocks based on sparsity
                    bool is_diagonal   = (qb == kb && qb < num_k_blocks);
                    bool random_active = (dist(rng) > sparsity);

                    if(is_diagonal || random_active)
                    {
                        block_relation_onehot(b, h, qb, kb) = static_cast<uint8_t>(1);
                        active_blocks++;
                    }
                    else
                    {
                        block_relation_onehot(b, h, qb, kb) = static_cast<uint8_t>(0);
                    }
                }
            }
        }
    }

    float actual_sparsity =
        1.0f - static_cast<float>(active_blocks) / static_cast<float>(total_blocks);
    std::cout << "  Actual sparsity: " << actual_sparsity << " (" << active_blocks << "/"
              << total_blocks << " blocks active)" << std::endl;

    // Convert block_relation_onehot to LUT format
    std::cout << "Converting block map to LUT format..." << std::endl;
    block_map_to_lut(block_relation_onehot, lut_host, valid_block_num_host, num_k_blocks);

    // vsa_sparse_attention handles device memory internally

    // Run kernel
    std::cout << "\n--- Running VSA sparse attention kernel ---" << std::endl;

    try
    {
        // Print kernel name once by invoking with log_level=1.
        // This is separate from warmup/benchmark to avoid polluting timing.
        if(kname)
        {
            vsa_sparse_attention<T>(q_host,
                                    k_host,
                                    v_host,
                                    lut_host,
                                    valid_block_num_host,
                                    output_host,
                                    batch,
                                    nhead,
                                    nhead_k,
                                    seqlen_q,
                                    seqlen_k,
                                    hdim_q,
                                    hdim_v,
                                    i_perm,
                                    o_perm,
                                    seqlen_q,
                                    seqlen_k,
                                    1);
        }

        // Warmup
        for(int i = 0; i < warmup; ++i)
        {
            vsa_sparse_attention<T>(q_host,
                                    k_host,
                                    v_host,
                                    lut_host,
                                    valid_block_num_host,
                                    output_host,
                                    batch,
                                    nhead,
                                    nhead_k,
                                    seqlen_q,
                                    seqlen_k,
                                    hdim_q,
                                    hdim_v,
                                    i_perm,
                                    o_perm,
                                    seqlen_q,
                                    seqlen_k,
                                    0);
        }

        // Benchmark
        [[maybe_unused]] auto sync_status1 = hipDeviceSynchronize();
        auto start                         = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < repeat; ++i)
        {
            vsa_sparse_attention<T>(q_host,
                                    k_host,
                                    v_host,
                                    lut_host,
                                    valid_block_num_host,
                                    output_host,
                                    batch,
                                    nhead,
                                    nhead_k,
                                    seqlen_q,
                                    seqlen_k,
                                    hdim_q,
                                    hdim_v,
                                    i_perm,
                                    o_perm,
                                    seqlen_q,
                                    seqlen_k,
                                    0);
        }

        [[maybe_unused]] auto sync_status2 = hipDeviceSynchronize();
        auto end                           = std::chrono::high_resolution_clock::now();
        double avg_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count() / repeat;

        std::cout << "\n>>>> VSA sparse attention average time: " << avg_time_ms << " ms <<<<"
                  << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error during kernel execution: " << e.what() << std::endl;
        return false;
    }

    // Note: vsa_sparse_attention already returns output in output_host

    // Validation
    bool pass = true;
    if(do_validation)
    {
        std::cout << "\n--- Performing CPU validation ---" << std::endl;

        // Compute scale factor
        float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

        // Run reference implementation
        std::cout << "Computing reference output..." << std::endl;
        auto q_ref = to_bhsd(q_host, i_perm);
        auto k_ref = to_bhsd(k_host, i_perm);
        auto v_ref = to_bhsd(v_host, i_perm);
        ck_tile::reference_blocked_attention<T, uint8_t>(
            q_ref, k_ref, v_ref, block_relation_onehot, output_ref, BLKQ, BLKK, scale);

        // Compare results
        auto [rtol, atol] = get_error_tolerance<T>();

        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t num_errors  = 0;

        auto output_host_bhsd = to_bhsd(output_host, o_perm);
        for(size_t i = 0; i < output_host_bhsd.mData.size(); ++i)
        {
            float gpu_val  = to_float_for_compare(output_host_bhsd.mData[i]);
            float ref_val  = to_float_for_compare(output_ref.mData[i]);
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
        std::cout << "  Number of mismatches: " << num_errors << " / "
                  << output_host_bhsd.mData.size() << std::endl;

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

    std::string prec = arg_parser.get_str("prec");

    bool test_result = false;
    if(prec == "fp16")
    {
        test_result = run_test<ck_tile::half_t>(arg_parser);
    }
    else if(prec == "bf16")
    {
        test_result = run_test<ck_tile::bf16_t>(arg_parser);
    }
    else
    {
        std::cerr << "Unsupported precision: " << prec << std::endl;
        return -1;
    }

    return test_result ? 0 : -1;
}
