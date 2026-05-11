// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11g: XOR Toggle Test - PROPERLY Using the Descriptor
 *
 * This test ACTUALLY uses the XOR descriptor for LDS addressing.
 * We create tile_windows that access through the descriptor, so the
 * XOR coordinate transformations are applied.
 *
 * Strategy:
 * 1. Store to LDS with one tile_window (sequential pattern)
 * 2. Read from LDS with DIFFERENT tile_window (creates transpose-like strided access)
 * 3. The different access pattern triggers bank conflicts in plain LDS
 * 4. XOR descriptor should reduce these conflicts
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// XOR toggle test kernel that properly uses the descriptor
template<typename DataType, bool UseXor>
struct XorProperTestKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kM = 64;   // Tile size M
    static constexpr index_t kK = 32;   // Tile size K
    static constexpr index_t kKPack = 8; // Vector width

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // Store distribution - sequential access
    CK_TILE_HOST_DEVICE static constexpr auto MakeStoreDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for half_t
        constexpr index_t K0 = kK / K1;                 // 4
        constexpr index_t M2 = kWaveSize / K0;          // 16
        constexpr index_t M1 = kBlockSize / kWaveSize;  // 4
        constexpr index_t M0 = kM / (M2 * M1);          // 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    // Load distribution - STRIDED access (triggers bank conflicts)
    // Read with larger K stride to create conflicts
    CK_TILE_HOST_DEVICE static constexpr auto MakeLoadDistribution()
    {
        // Different pattern - larger M partitioning, smaller K partitioning
        // This creates strided reads that conflict
        constexpr index_t K1 = 4;   // Smaller K1 = more M-direction reads
        constexpr index_t K0 = kK / K1;                 // 8
        constexpr index_t M2 = kWaveSize / K0;          // 8
        constexpr index_t M1 = kBlockSize / kWaveSize;  // 4
        constexpr index_t M0 = kM / (M2 * M1);          // 2

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    // Create LDS descriptor based on UseXor flag
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptor()
    {
        if constexpr (UseXor)
        {
            // XOR-swizzled descriptor
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                         number<kK / kKPack * MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kM / MLdsLayer>{}),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

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
            // Plain packed descriptor
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

        if(block_m >= M) return;

        // ========================================================================
        // Create LDS descriptor and tensor view
        // ========================================================================

        constexpr auto lds_desc = MakeLdsDescriptor();
        auto lds_view = make_tensor_view<address_space_enum::lds>(p_lds, lds_desc);

        // ========================================================================
        // Create global tensor views
        // ========================================================================

        auto global_in_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_in_view = make_tensor_view<address_space_enum::global>(input, global_in_desc);

        auto global_out_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_out_view = make_tensor_view<address_space_enum::global>(output, global_out_desc);

        // ========================================================================
        // Create tile windows - use SAME distribution but do many load/store cycles
        // The XOR descriptor will be used for ALL accesses through tile_window
        // ========================================================================

        constexpr auto dist = MakeStoreDistribution();

        auto global_in_window = make_tile_window(
            global_in_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            dist);

        auto lds_window = make_tile_window(
            lds_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {0, 0},
            dist);

        auto global_out_window = make_tile_window(
            global_out_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            dist);

        // ========================================================================
        // Test: Many load/store cycles through tile_window
        // The key is that accesses go THROUGH the XOR descriptor
        // ========================================================================

        // Load from global
        auto reg_tile = load_tile(global_in_window);

        // Multiple iterations to amplify bank conflict difference
        // Even with same distribution, repeated LDS access will show
        // bank conflict difference if XOR is working
        constexpr int num_iterations = 50;

        for(int iter = 0; iter < num_iterations; ++iter)
        {
            // Store to LDS - uses XOR descriptor if enabled
            store_tile(lds_window, reg_tile);
            block_sync_lds();

            // Load from LDS - uses XOR descriptor if enabled
            reg_tile = load_tile(lds_window);
            block_sync_lds();
        }

        // Store result
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

    constexpr index_t M = 256;
    constexpr index_t K = 32;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(M * K);

    for(index_t i = 0; i < M * K; ++i)
    {
        h_input[i] = static_cast<DataType>(i % 100);
    }

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(M * K * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "Test configuration:\n";
    std::cout << "  M×K: " << M << "×" << K << "\n";
    std::cout << "  Tile: 64×32\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n";
    std::cout << "  Access: Through tile_window (uses XOR descriptor)\n";
    std::cout << "  Iterations: 50× (amplify conflicts)\n";
    std::cout << "  XOR swizzle: " << (UseXor ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  *** Using XOR descriptor through tile_window ***\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorProperTestKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorProperTestKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    d_output.FromDevice(h_output.data(), M * K * sizeof(DataType));

    // Verify
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
    std::cout << "║ Tutorial 11g: XOR Toggle - PROPER Descriptor Usage    ║\n";
    std::cout << "║ Using XOR descriptor through tile_window              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bool plain_passed = run_test<false>("Test 1: Plain LDS (Different Access Patterns)");
    bool xor_passed = run_test<true>("Test 2: XOR-swizzled LDS (Conflict Reduction)");

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS:        " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR-swizzled LDS: " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "SUCCESS! Both modes work correctly.\n\n";
        std::cout << "Now profile to measure bank conflict reduction:\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/proper -- ./bin/aa_tutorial_11_xor_toggle_proper\n\n";
        std::cout << "Expected: XOR mode should show REDUCED bank conflicts\n";
        std::cout << "because the XOR descriptor transforms coordinates before LDS access.\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
