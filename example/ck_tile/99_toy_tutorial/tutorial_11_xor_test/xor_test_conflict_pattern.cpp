// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11h: XOR Test with INTENTIONAL Bank Conflicts
 *
 * This test creates a distribution pattern that INTENTIONALLY causes bank conflicts,
 * so we can measure XOR's effectiveness at eliminating them.
 *
 * Strategy:
 * - Use SCALAR (non-vectorized) loads: K1=1 instead of K1=8
 * - Create stride pattern that maps multiple threads to same bank
 * - For FP16: bank = (byte_address / 4) % 32
 * - Stride-64 elements = 128 bytes = wraps to same bank set
 *
 * Expected result:
 * - Plain LDS: HIGH bank conflicts (multiple threads hit same bank)
 * - XOR LDS:   LOW bank conflicts (XOR spreads accesses across banks)
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct ConflictPatternKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // BAD distribution - creates bank conflicts
    // Use K1=1 (scalar) and specific M partitioning to create stride-64 access
    CK_TILE_HOST_DEVICE static constexpr auto MakeBadDistribution()
    {
        // K1=1 means scalar loads (no vectorization)
        // This creates stride patterns that cause bank conflicts
        constexpr index_t K1 = 1;  // SCALAR! (was 8 in good distribution)
        constexpr index_t K0 = kK / K1;  // 32

        // M partitioning chosen to create stride-64 access patterns
        constexpr index_t M2 = 2;   // Small M2 creates larger strides
        constexpr index_t M1 = kBlockSize / (kWaveSize);  // 4
        constexpr index_t M0 = kM / (M2 * M1);  // 64 / 8 = 8

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

    // XOR descriptor
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptor()
    {
        if constexpr (UseXor)
        {
            // XOR-swizzled
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
            // Plain
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

        // Create LDS descriptor and view
        constexpr auto lds_desc = MakeLdsDescriptor();
        auto lds_view = make_tensor_view<address_space_enum::lds>(p_lds, lds_desc);

        // Create global views
        auto global_in_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_in_view = make_tensor_view<address_space_enum::global>(input, global_in_desc);

        auto global_out_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_out_view = make_tensor_view<address_space_enum::global>(output, global_out_desc);

        // Use the BAD distribution that creates conflicts
        constexpr auto bad_dist = MakeBadDistribution();

        auto global_in_window = make_tile_window(
            global_in_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            bad_dist);

        auto lds_window = make_tile_window(
            lds_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {0, 0},
            bad_dist);  // BAD distribution creates conflicts!

        auto global_out_window = make_tile_window(
            global_out_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            bad_dist);

        // Load from global
        auto reg_tile = load_tile(global_in_window);

        // Many iterations to amplify bank conflicts
        constexpr int num_iterations = 100;

        for(int iter = 0; iter < num_iterations; ++iter)
        {
            // Store to LDS with bad distribution (creates conflicts in plain mode)
            store_tile(lds_window, reg_tile);
            block_sync_lds();

            // Load from LDS with bad distribution (creates conflicts in plain mode)
            // XOR descriptor should spread these accesses across banks
            reg_tile = load_tile(lds_window);
            block_sync_lds();
        }

        // Store result
        store_tile(global_out_window, reg_tile);
    }
};

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
    std::cout << "  Distribution: K1=1 SCALAR (intentionally bad!)\n";
    std::cout << "  Access pattern: Creates bank conflicts\n";
    std::cout << "  Iterations: 100× (amplify conflicts)\n";
    std::cout << "  XOR swizzle: " << (UseXor ? "ENABLED" : "DISABLED") << "\n\n";

    stream_config stream;
    constexpr index_t lds_size = ConflictPatternKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     ConflictPatternKernel<DataType, UseXor>{},
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
    std::cout << "║ Tutorial 11h: XOR Test - Intentional Conflict Pattern ║\n";
    std::cout << "║ Scalar loads + bad stride = bank conflicts!           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bool plain_passed = run_test<false>("Test 1: Plain LDS (HIGH conflicts expected)");
    bool xor_passed = run_test<true>("Test 2: XOR LDS (LOW conflicts expected)");

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS:        " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR-swizzled LDS: " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "SUCCESS! Both modes are correct.\n\n";
        std::cout << "Profile command:\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/conflict -- ./bin/aa_tutorial_11_xor_conflict\n\n";
        std::cout << "Expected results:\n";
        std::cout << "  Plain LDS: HIGH SQ_LDS_BANK_CONFLICT (scalar loads with bad stride)\n";
        std::cout << "  XOR LDS:   REDUCED SQ_LDS_BANK_CONFLICT (XOR spreads across banks)\n\n";
        std::cout << "This test uses K1=1 (scalar) instead of K1=8 (vectorized),\n";
        std::cout << "creating access patterns that cause bank conflicts.\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
