// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11e: XOR Toggle Test for LDS Bank Conflict Profiling
 *
 * This test allows toggling XOR swizzling on/off to compare LDS bank conflicts.
 * Use with rocprofv3 to measure the impact of XOR swizzling on bank conflicts.
 *
 * Template parameter UseXor:
 *   - true:  Use XOR-swizzled LDS layout (bank conflict-free)
 *   - false: Use plain packed LDS layout (potential bank conflicts)
 *
 * Test flow:
 * 1. Load from global memory using tile_window with copy distribution
 * 2. Store to LDS (XOR-swizzled or plain) using tile_window
 * 3. Load from LDS (XOR-swizzled or plain) using tile_window
 * 4. Store to global memory using tile_window
 *
 * Profile command:
 *   rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/toggle --csv -- ./bin/aa_tutorial_11_xor_toggle
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// XOR toggle test kernel
template<typename DataType, bool UseXor>
struct XorToggleTestKernel
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

    // Copy distribution (same pattern as Tutorial 10)
    CK_TILE_HOST_DEVICE static constexpr auto MakeCopyDistribution()
    {
        // Vector width calculation for 16-byte loads
        constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for half_t
        constexpr index_t K0 = kK / K1;                 // 32 / 8 = 4
        constexpr index_t M2 = kWaveSize / K0;          // 64 / 4 = 16
        constexpr index_t M1 = kBlockSize / kWaveSize;  // 256 / 64 = 4
        constexpr index_t M0 = kM / (M2 * M1);          // 64 / (16 * 4) = 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,                                    // NO replication!
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>, // Thread partitioning
                tuple<sequence<1>, sequence<1, 2>>,            // Ps_to_Hs
                tuple<sequence<1>, sequence<2, 0>>,            // Ps_in_Hs
                sequence<1, 2>,                                 // Ys_to_Hs
                sequence<0, 1>                                  // Ys_in_Hs
            >{});
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

        // Bounds check
        if(block_m >= M) return;

        // ========================================================================
        // Create LDS descriptor (XOR or plain based on template parameter)
        // ========================================================================

        constexpr auto lds_desc = MakeLdsDescriptor();

        // ========================================================================
        // Create global and LDS tensor views
        // ========================================================================

        auto global_in_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_in_view = make_tensor_view<address_space_enum::global>(input, global_in_desc);

        auto global_out_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_out_view = make_tensor_view<address_space_enum::global>(output, global_out_desc);

        auto lds_view = make_tensor_view<address_space_enum::lds>(p_lds, lds_desc);

        // ========================================================================
        // Create tile windows with copy distribution
        // ========================================================================

        constexpr auto copy_dist = MakeCopyDistribution();

        auto global_in_window = make_tile_window(
            global_in_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            copy_dist);

        auto lds_window = make_tile_window(
            lds_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {0, 0},
            copy_dist);

        auto global_out_window = make_tile_window(
            global_out_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            copy_dist);

        // ========================================================================
        // Test: Copy through LDS using tile_window (with multiple iterations)
        // ========================================================================

        // Load from global memory
        auto reg_tile = load_tile(global_in_window);

        // Perform multiple LDS store/load cycles to stress test bank conflicts
        constexpr int num_iterations = 10;
        for(int iter = 0; iter < num_iterations; ++iter)
        {
            // Store to LDS (XOR-swizzled or plain)
            store_tile(lds_window, reg_tile);

            block_sync_lds();

            // Load from LDS (XOR-swizzled or plain)
            reg_tile = load_tile(lds_window);

            block_sync_lds();
        }

        // Store to global memory
        store_tile(global_out_window, reg_tile);
    }
};

// Test function template
template<bool UseXor>
bool run_test(const std::string& test_name)
{
    std::cout << "\n========================================\n";
    std::cout << test_name << "\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 256;  // Increased for more stress
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
    std::cout << "  Using: tile_window with copy distribution\n";
    std::cout << "  XOR swizzle: " << (UseXor ? "ENABLED" : "DISABLED") << "\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorToggleTestKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorToggleTestKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    // Get result
    d_output.FromDevice(h_output.data(), M * K * sizeof(DataType));

    // Verify
    bool passed = true;
    index_t error_count = 0;

    for(index_t i = 0; i < M * K; ++i)
    {
        // Compare bit patterns for exact equality
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
    std::cout << "║ Tutorial 11e: XOR Toggle Test                         ║\n";
    std::cout << "║ Compare LDS bank conflicts with/without XOR swizzling ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    // Run both tests
    bool plain_passed = run_test<false>("Test 1: Plain LDS (Potential Bank Conflicts)");
    bool xor_passed = run_test<true>("Test 2: XOR-swizzled LDS (Bank Conflict-Free)");

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS:        " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR-swizzled LDS: " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "Both tests passed! Both layouts are functionally correct.\n\n";
        std::cout << "To measure bank conflict difference, profile with rocprofv3:\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/toggle --csv -- ./bin/aa_tutorial_11_xor_toggle\n\n";
        std::cout << "Expected results:\n";
        std::cout << "  - Plain LDS: Higher SQ_LDS_BANK_CONFLICT count\n";
        std::cout << "  - XOR LDS:   Lower/zero SQ_LDS_BANK_CONFLICT count\n";
    }
    else
    {
        std::cout << "At least one test failed. Fix correctness before profiling.\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
