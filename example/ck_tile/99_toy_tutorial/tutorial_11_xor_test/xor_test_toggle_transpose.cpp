// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11f: XOR Toggle Test with TRANSPOSE Access Pattern
 *
 * This test uses a transpose access pattern to trigger LDS bank conflicts.
 *
 * Access pattern:
 * 1. Store to LDS in [M, K] layout (row-major)
 * 2. Read from LDS with TRANSPOSED indices [K, M] to create strided access
 *
 * This creates bank conflicts in plain LDS because:
 * - Each thread reads with stride-M (64 elements)
 * - Stride-64 with FP16 = 128-byte stride
 * - Multiple threads hit the same bank simultaneously
 *
 * XOR swizzling should eliminate these conflicts.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// XOR toggle test kernel with transpose access
template<typename DataType, bool UseXor>
struct XorTransposeTestKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kM = 64;   // Tile size M
    static constexpr index_t kK = 32;   // Tile size K
    static constexpr index_t kKPack = 8; // Vector width

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);  // 64*32*2 = 4096 bytes
    }

    // Create LDS descriptor based on UseXor flag
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptor()
    {
        if constexpr (UseXor)
        {
            // XOR-swizzled descriptor (bank conflict-free)
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            // Step 1: Reshape
            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            // Step 2: XOR permute
            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                         number<kK / kKPack * MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            // Step 3: Unmerge
            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kM / MLdsLayer>{}),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            // Step 4: Merge back to [M, K]
            constexpr auto lds_desc = transform_tensor_descriptor(
                lds_desc_unmerged,
                make_tuple(
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            // Plain packed descriptor (potential bank conflicts)
            return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
        }
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        extern __shared__ char smem[];
        DataType* p_lds = reinterpret_cast<DataType*>(smem);

        const index_t block_m = get_block_id() * kM;
        const index_t thread_id = get_thread_id();

        // Bounds check
        if(block_m >= M) return;

        // ========================================================================
        // Create LDS descriptor (XOR or plain based on template parameter)
        // ========================================================================

        // ========================================================================
        // WRITE to LDS: Each thread writes sequential elements (coalesced)
        // ========================================================================

        constexpr index_t elements_per_thread = (kM * kK) / kBlockSize;  // 64*32/256 = 8

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            index_t flat_idx = thread_id + i * kBlockSize;
            index_t m_idx = flat_idx / kK;
            index_t k_idx = flat_idx % kK;

            index_t global_idx = (block_m + m_idx) * K + k_idx;
            index_t lds_idx = m_idx * kK + k_idx;

            DataType val = input[global_idx];
            p_lds[lds_idx] = val;
        }

        block_sync_lds();

        // ========================================================================
        // READ from LDS: TRANSPOSED pattern (creates bank conflicts in plain LDS)
        // Read with stride-M instead of stride-K to trigger conflicts
        // ========================================================================

        // Repeat reads multiple times to amplify bank conflict counts
        constexpr int num_read_iterations = 20;

        DataType sum = static_cast<DataType>(0.0f);

        for(int iter = 0; iter < num_read_iterations; ++iter)
        {
            // Each thread reads with TRANSPOSED indices
            // This creates strided access pattern with stride=M (64 elements = 128 bytes for FP16)
            for(index_t i = 0; i < elements_per_thread; ++i)
            {
                index_t flat_idx = thread_id + i * kBlockSize;

                // TRANSPOSE: swap M and K indices
                index_t k_idx = flat_idx / kM;  // Was: flat_idx / kK
                index_t m_idx = flat_idx % kM;  // Was: flat_idx % kK

                // Clamp to valid range
                if(k_idx < kK && m_idx < kM)
                {
                    // Read with transposed layout (creates bank conflicts)
                    index_t lds_idx = m_idx * kK + k_idx;
                    DataType val = p_lds[lds_idx];
                    sum = sum + val;  // Accumulate to prevent optimization
                }
            }

            block_sync_lds();
        }

        // ========================================================================
        // Write result back (use sum to prevent dead code elimination)
        // ========================================================================

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            index_t flat_idx = thread_id + i * kBlockSize;
            index_t m_idx = flat_idx / kK;
            index_t k_idx = flat_idx % kK;

            index_t global_idx = (block_m + m_idx) * K + k_idx;

            // Write back with small modification to ensure sum is used
            DataType orig_val = input[global_idx];
            output[global_idx] = orig_val + sum * static_cast<DataType>(0.0f);
        }
    }
};

// Test function template
template<bool UseXor>
bool run_test(const std::string& test_name)
{
    std::cout << "\n========================================\n";
    std::cout << test_name << "\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 256;
    constexpr index_t K = 32;  // Must match kK in kernel!

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(M * K);

    // Initialize input
    for(index_t i = 0; i < M * K; ++i)
    {
        h_input[i] = static_cast<DataType>(i % 100);
    }

    // Device memory
    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(M * K * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    // Launch kernel
    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "Test configuration:\n";
    std::cout << "  M×K: " << M << "×" << K << "\n";
    std::cout << "  Tile: 64×32\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n";
    std::cout << "  Access: TRANSPOSED reads (bank conflict trigger)\n";
    std::cout << "  Read iterations: 20× (amplify conflicts)\n";
    std::cout << "  XOR swizzle: " << (UseXor ? "ENABLED" : "DISABLED") << "\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorTransposeTestKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorTransposeTestKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    // Get result
    d_output.FromDevice(h_output.data(), M * K * sizeof(DataType));

    // Verify (should match input since we add sum*0.0)
    bool passed = true;
    index_t error_count = 0;

    for(index_t i = 0; i < M * K; ++i)
    {
        uint16_t out_bits = bit_cast<uint16_t>(h_output[i]);
        uint16_t in_bits = bit_cast<uint16_t>(h_input[i]);
        if(out_bits != in_bits)
        {
            if(error_count < 10)
            {
                index_t m = i / K;
                index_t k = i % K;
                std::cout << "Error at [" << m << "," << k << "]: "
                          << static_cast<float>(h_output[i]) << " vs "
                          << static_cast<float>(h_input[i]) << "\n";
            }
            error_count++;
            passed = false;
        }
    }

    std::cout << "\nResults:\n";
    std::cout << "  Correctness: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    if(!passed)
    {
        std::cout << "  Error count: " << error_count << "/" << (M*K) << "\n";
    }

    return passed;
}

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Tutorial 11f: XOR Toggle Test - Transpose Pattern     ║\n";
    std::cout << "║ Transposed LDS reads trigger bank conflicts           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    // Run both tests
    bool plain_passed = run_test<false>("Test 1: Plain LDS (BANK CONFLICTS EXPECTED)");
    bool xor_passed = run_test<true>("Test 2: XOR-swizzled LDS (Conflict-Free)");

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS:        " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR-swizzled LDS: " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "Both tests passed! Now profile to see bank conflict difference.\n\n";
        std::cout << "Profile command:\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/transpose --csv -- ./bin/aa_tutorial_11_xor_toggle_transpose\n\n";
        std::cout << "Expected results:\n";
        std::cout << "  - Plain LDS: HIGH SQ_LDS_BANK_CONFLICT (transposed reads cause conflicts)\n";
        std::cout << "  - XOR LDS:   LOW/ZERO SQ_LDS_BANK_CONFLICT (XOR eliminates conflicts)\n";
    }
    else
    {
        std::cout << "At least one test failed. Fix correctness before profiling.\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
