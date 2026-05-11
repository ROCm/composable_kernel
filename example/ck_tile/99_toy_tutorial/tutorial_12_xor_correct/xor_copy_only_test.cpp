// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 10: XOR LDS Copy-Only Test
 *
 * This test isolates whether the XOR descriptor works for basic copy operations
 * in Tutorial 10's exact context (tile sizes, distributions, etc.)
 *
 * Test flow:
 * 1. Load A from global using copy distribution
 * 2. Store A to XOR-swizzled LDS using copy distribution
 * 3. Load A from XOR-swizzled LDS using copy distribution
 * 4. Store A to global using copy distribution
 * 5. Verify output matches input
 *
 * Same test for B matrix.
 *
 * If this passes: XOR descriptor is fine, issue is in GEMM logic
 * If this fails: XOR descriptor has a context-specific bug
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct XorCopyOnlyTestKernel
{
    // Same configuration as Tutorial 10
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kMPerBlock = 64;   // Tutorial 10 uses 64, not 128!
    static constexpr index_t kNPerBlock = 64;   // Tutorial 10 uses 64, not 128!
    static constexpr index_t kKPerBlock = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return (kMPerBlock * kKPerBlock + kNPerBlock * kKPerBlock) * sizeof(DataType);
    }

    // Copy distribution (same as Tutorial 10)
    CK_TILE_HOST_DEVICE static constexpr auto MakeACopyDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = 64 / K0;
        constexpr index_t M1 = kBlockSize / 64;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
    {
        // B is K×N in memory, so vector width applies to N dimension (innermost)
        constexpr index_t N1 = 16 / sizeof(DataType);    // 8 for half_t
        constexpr index_t N0 = kNPerBlock / N1;          // 128 / 8 = 16
        constexpr index_t K2 = 64 / N0;                  // 64 / 16 = 4
        constexpr index_t K1 = kBlockSize / 64;          // 256 / 64 = 4
        constexpr index_t K0 = kKPerBlock / (K2 * K1);   // 32 / (4 * 4) = 2

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<N0, N1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ a_ptr,
                                    const DataType* __restrict__ b_ptr,
                                    DataType* __restrict__ a_out_ptr,
                                    DataType* __restrict__ b_out_ptr,
                                    index_t M,
                                    index_t N,
                                    index_t K) const
    {
        extern __shared__ char smem[];
        DataType* a_lds_ptr = reinterpret_cast<DataType*>(smem);
        DataType* b_lds_ptr = reinterpret_cast<DataType*>(smem + kMPerBlock * kKPerBlock * sizeof(DataType));

        const index_t block_m = get_block_id() * kMPerBlock;
        const index_t block_n = 0;  // Only test one block for simplicity
        if(block_m >= M) return;

        // ========================================================================
        // Create XOR-swizzled LDS descriptors (EXACT copy from Tutorial 10)
        // ========================================================================

        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);
        constexpr auto NLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        // A matrix XOR descriptor
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                       number<kMPerBlock / MLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{},
                       number<kKPerBlock * MLdsLayer>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                     number<kKPerBlock / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(
                make_merge_transform(make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // B matrix XOR descriptor
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * NLdsLayer>{},
                       number<kNPerBlock / NLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{},
                       number<kKPerBlock * NLdsLayer>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / NLdsLayer>{},
                                                     number<kKPerBlock / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<NLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                       make_pass_through_transform(number<kNPerBlock / NLdsLayer>{}),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(
                make_merge_transform(make_tuple(number<kNPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // ========================================================================
        // Create tensor views and windows
        // ========================================================================

        // Global memory views
        auto a_global_desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
        auto a_global_view = make_tensor_view<address_space_enum::global>(a_ptr, a_global_desc);
        auto a_global_out_view = make_tensor_view<address_space_enum::global>(a_out_ptr, a_global_desc);

        auto b_global_desc = make_naive_tensor_descriptor_packed(make_tuple(N, K));
        auto b_global_view = make_tensor_view<address_space_enum::global>(b_ptr, b_global_desc);
        auto b_global_out_view = make_tensor_view<address_space_enum::global>(b_out_ptr, b_global_desc);

        // LDS views with XOR descriptors
        auto a_lds_view = make_tensor_view<address_space_enum::lds>(a_lds_ptr, a_lds_block_desc);
        auto b_lds_view = make_tensor_view<address_space_enum::lds>(b_lds_ptr, b_lds_block_desc);

        constexpr auto a_copy_dist = MakeACopyDistribution();
        constexpr auto b_copy_dist = MakeBCopyDistribution();

        // A matrix windows
        auto a_global_in_window = make_tile_window(
            a_global_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {block_m, 0},
            a_copy_dist);

        auto a_lds_window = make_tile_window(
            a_lds_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            a_copy_dist);

        auto a_global_out_window = make_tile_window(
            a_global_out_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {block_m, 0},
            a_copy_dist);

        // B matrix windows
        auto b_global_in_window = make_tile_window(
            b_global_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {block_n, 0},
            b_copy_dist);

        auto b_lds_window = make_tile_window(
            b_lds_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),  // [N, K] - matches XOR descriptor
            {0, 0},
            b_copy_dist);

        auto b_global_out_window = make_tile_window(
            b_global_out_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {block_n, 0},
            b_copy_dist);

        // ========================================================================
        // Test A: Global → XOR LDS → Global
        // ========================================================================

        auto a_reg_tile = load_tile(a_global_in_window);
        store_tile(a_lds_window, a_reg_tile);
        block_sync_lds();
        auto a_reg_tile_out = load_tile(a_lds_window);
        store_tile(a_global_out_window, a_reg_tile_out);

        // ========================================================================
        // Test B: Global → XOR LDS → Global
        // ========================================================================

        auto b_reg_tile = load_tile(b_global_in_window);
        store_tile(b_lds_window, b_reg_tile);
        block_sync_lds();
        auto b_reg_tile_out = load_tile(b_lds_window);
        store_tile(b_global_out_window, b_reg_tile_out);
    }
};

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 10: XOR LDS Copy-Only Test\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 64;   // Match kMPerBlock
    constexpr index_t N = 64;   // Match kNPerBlock
    constexpr index_t K = 32;

    using DataType = half_t;

    std::vector<DataType> h_a(M * K);
    std::vector<DataType> h_b(N * K);
    std::vector<DataType> h_a_out(M * K);
    std::vector<DataType> h_b_out(N * K);

    // Initialize with simple pattern
    for(index_t i = 0; i < M * K; ++i)
    {
        h_a[i] = static_cast<DataType>(i % 100);
    }
    for(index_t i = 0; i < N * K; ++i)
    {
        h_b[i] = static_cast<DataType>((i + 50) % 100);
    }

    DeviceMem d_a(M * K * sizeof(DataType));
    DeviceMem d_b(N * K * sizeof(DataType));
    DeviceMem d_a_out(M * K * sizeof(DataType));
    DeviceMem d_b_out(N * K * sizeof(DataType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(DataType));
    d_b.ToDevice(h_b.data(), N * K * sizeof(DataType));

    constexpr index_t kMPerBlock = 128;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kMPerBlock - 1) / kMPerBlock;

    std::cout << "Test configuration:\n";
    std::cout << "  M×N×K: " << M << "×" << N << "×" << K << "\n";
    std::cout << "  Tile: 128×128×32\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n";
    std::cout << "  Test: Copy through XOR-swizzled LDS\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorCopyOnlyTestKernel<DataType>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorCopyOnlyTestKernel<DataType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_a.GetDeviceBuffer()),
                     static_cast<const DataType*>(d_b.GetDeviceBuffer()),
                     static_cast<DataType*>(d_a_out.GetDeviceBuffer()),
                     static_cast<DataType*>(d_b_out.GetDeviceBuffer()),
                     M, N, K));

    hip_check_error(hipDeviceSynchronize());

    d_a_out.FromDevice(h_a_out.data(), M * K * sizeof(DataType));
    d_b_out.FromDevice(h_b_out.data(), N * K * sizeof(DataType));

    // Verify A matrix
    bool a_passed = true;
    index_t a_error_count = 0;
    for(index_t i = 0; i < M * K; ++i)
    {
        uint16_t out_bits = bit_cast<uint16_t>(h_a_out[i]);
        uint16_t in_bits = bit_cast<uint16_t>(h_a[i]);
        if(out_bits != in_bits)
        {
            if(a_error_count < 5)
            {
                index_t m = i / K;
                index_t k = i % K;
                std::cout << "A Error at [" << m << "," << k << "]: "
                          << static_cast<float>(h_a_out[i]) << " vs "
                          << static_cast<float>(h_a[i]) << "\n";
            }
            a_error_count++;
            a_passed = false;
        }
    }

    // Verify B matrix
    bool b_passed = true;
    index_t b_error_count = 0;
    for(index_t i = 0; i < N * K; ++i)
    {
        uint16_t out_bits = bit_cast<uint16_t>(h_b_out[i]);
        uint16_t in_bits = bit_cast<uint16_t>(h_b[i]);
        if(out_bits != in_bits)
        {
            if(b_error_count < 5)
            {
                index_t n = i / K;
                index_t k = i % K;
                std::cout << "B Error at [" << n << "," << k << "]: "
                          << static_cast<float>(h_b_out[i]) << " vs "
                          << static_cast<float>(h_b[i]) << "\n";
            }
            b_error_count++;
            b_passed = false;
        }
    }

    std::cout << "\nResults:\n";
    std::cout << "  A Matrix: " << (a_passed ? "✓ PASSED" : "✗ FAILED");
    if(!a_passed) std::cout << " (" << a_error_count << "/" << (M*K) << " errors)";
    std::cout << "\n";
    std::cout << "  B Matrix: " << (b_passed ? "✓ PASSED" : "✗ FAILED");
    if(!b_passed) std::cout << " (" << b_error_count << "/" << (N*K) << " errors)";
    std::cout << "\n\n";

    std::cout << "=== Analysis ===\n";
    if(a_passed && b_passed)
    {
        std::cout << "SUCCESS! XOR descriptor works for copy operations.\n";
        std::cout << "The issue in Tutorial 10's GEMM must be in:\n";
        std::cout << "  - GEMM distribution accessing XOR LDS\n";
        std::cout << "  - OR GEMM computation with XOR-loaded data\n";
        std::cout << "  - OR interaction between copy and GEMM windows\n";
    }
    else
    {
        std::cout << "FAILED! XOR descriptor doesn't work for basic copy.\n";
        std::cout << "This indicates a bug in the XOR descriptor creation itself.\n";
    }

    return (a_passed && b_passed) ? 0 : 1;
}
