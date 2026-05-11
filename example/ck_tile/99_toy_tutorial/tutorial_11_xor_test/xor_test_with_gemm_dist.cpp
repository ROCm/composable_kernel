// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11c: XOR Descriptor Test WITH GEMM Distribution
 *
 * This test validates whether XOR-swizzled LDS descriptors work with
 * GEMM-style distributions (warp-based with replication).
 *
 * Test flow:
 * 1. Load from global using copy distribution
 * 2. Store to XOR-swizzled LDS using copy distribution
 * 3. Load from XOR-swizzled LDS using GEMM distribution ← THE KEY TEST
 * 4. Store to global using copy distribution
 *
 * This isolates whether GEMM distribution + XOR works.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// XOR test kernel with GEMM distribution
template<typename DataType>
struct XorGemmDistTestKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kWaveSize = 64;
    static constexpr index_t kM = 64;   // Tile size M
    static constexpr index_t kK = 32;   // Tile size K
    static constexpr index_t kKPack = 8; // Vector width

    // Same warp configuration as Tutorial 10
    static constexpr index_t MWarp = 2;
    static constexpr index_t NWarp = 2;  // Not used in this test, but needed for pattern
    static constexpr index_t MIterPerWarp = 2;
    static constexpr index_t KIterPerWarp = 2;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // Copy distribution (for Global ↔ LDS)
    CK_TILE_HOST_DEVICE static constexpr auto MakeCopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for half_t
        constexpr index_t K0 = kK / K1;                 // 32 / 8 = 4
        constexpr index_t M2 = kWaveSize / K0;          // 64 / 4 = 16
        constexpr index_t M1 = kBlockSize / kWaveSize;  // 256 / 64 = 4
        constexpr index_t M0 = kM / (M2 * M1);          // 64 / (16 * 4) = 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    // GEMM distribution (exact copy from Tutorial 10)
    CK_TILE_HOST_DEVICE static constexpr auto MakeGemmDistribution()
    {
        // Warp-level distribution
        constexpr auto warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        // Block-level with REPLICATION across N-warps
        constexpr auto block_outer_dstr_encode = tile_distribution_encoding<
            sequence<NWarp>,                                    // REPLICATE across N-warps!
            tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        return make_static_tile_distribution(
            detail::make_embed_tile_distribution_encoding(
                block_outer_dstr_encode, warp_dstr_encode)
        );
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
        // Create XOR-swizzled LDS descriptor
        // ========================================================================

        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kK / kKPack * MLdsLayer>{},
                       number<kM / MLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{},
                       number<kK * MLdsLayer>{},
                       number<1>{}),
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

        // ========================================================================
        // Create tensor views and windows
        // ========================================================================

        auto global_in_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_in_view = make_tensor_view<address_space_enum::global>(input, global_in_desc);

        auto global_out_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto global_out_view = make_tensor_view<address_space_enum::global>(output, global_out_desc);

        auto lds_view = make_tensor_view<address_space_enum::lds>(p_lds, lds_desc);

        constexpr auto copy_dist = MakeCopyDistribution();
        constexpr auto gemm_dist = MakeGemmDistribution();

        // Global window with copy distribution
        auto global_in_window = make_tile_window(
            global_in_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            copy_dist);

        // LDS window for WRITING (copy distribution)
        auto lds_copy_window = make_tile_window(
            lds_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {0, 0},
            copy_dist);

        // LDS window for READING (GEMM distribution) ← THE KEY TEST
        auto lds_gemm_window = make_tile_window(
            lds_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {0, 0},
            gemm_dist);

        // Global window for writing back (copy distribution)
        auto global_out_window = make_tile_window(
            global_out_view,
            make_tuple(number<kM>{}, number<kK>{}),
            {block_m, 0},
            copy_dist);

        // ========================================================================
        // Test: Load with copy, store to XOR LDS, load with GEMM, store back
        // ========================================================================

        // Load from global memory (copy distribution)
        auto reg_tile_copy = load_tile(global_in_window);

        // Store to XOR-swizzled LDS (copy distribution)
        store_tile(lds_copy_window, reg_tile_copy);

        block_sync_lds();

        // Load from XOR-swizzled LDS (GEMM distribution) ← CRITICAL TEST
        auto reg_tile_gemm = load_tile(lds_gemm_window);

        // Store to global memory (copy distribution)
        store_tile(global_out_window, reg_tile_gemm);
    }
};

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 11c: XOR + GEMM Distribution Test\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 128;
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
    std::cout << "  Store: Copy distribution\n";
    std::cout << "  Load:  GEMM distribution (warp-based, replicated)\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorGemmDistTestKernel<DataType>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorGemmDistTestKernel<DataType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    d_output.FromDevice(h_output.data(), M * K * sizeof(DataType));

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

    std::cout << "\n=== Analysis ===\n";
    if(passed)
    {
        std::cout << "SUCCESS! XOR descriptor works with GEMM distribution!\n";
        std::cout << "The issue in Tutorial 10 must be something else.\n";
    }
    else
    {
        std::cout << "FAILED! XOR descriptor is incompatible with GEMM distribution.\n";
        std::cout << "This explains Tutorial 10's failure!\n";
        std::cout << "GEMM distribution expects different LDS layout than XOR provides.\n";
    }

    return passed ? 0 : 1;
}
