// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11: Plain Transpose (NO XOR) - For Bank Conflict Profiling
 *
 * This implements matrix transpose using CK Tile API to demonstrate bank conflicts:
 * - Plain LDS descriptor (no XOR swizzling)
 * - Full CK Tile API: tensor_view, tile_window, load_tile, store_tile
 * - No manual loops for data movement
 *
 * Pattern:
 * 1. Load from global [M, K] using tile_window
 * 2. Store to LDS [M, K] using plain descriptor
 * 3. Read from LDS as [K, M] using transposed descriptor (stride-kK access = bank conflicts!)
 * 4. Store to global [K, M] transposed output
 *
 * Profile with: rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -d /tmp/plain -- ./bin/aa_tutorial_11_plain_transpose
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct PlainTransposeKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // Plain LDS descriptor for [M, K] - NO XOR
    // This creates a simple row-major layout with no bank conflict optimization.
    //
    // Memory layout:
    //   - Element [m][k] at offset: m * kK + k
    //   - Row-major: consecutive k values are adjacent in memory
    //
    // Bank conflict analysis (write phase):
    //   - Writing along K dimension (stride-1) is GOOD
    //   - For FP16: 2 bytes per element, so 2 elements per bank
    //   - Minimal conflicts (2-way) during write
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(kM, kK));
    }

    // Plain transposed LDS descriptor for [K, M] - stride-kK creates SEVERE bank conflicts!
    //
    // This descriptor reads the SAME physical memory as [M,K] but interprets it as [K,M].
    // This is how transpose works: different logical view of same data.
    //
    // Memory layout (physical - same as above):
    //   - Element [k][m] maps to physical offset: m * kK + k
    //   - To read column k=0: access elements at m=0,1,2,...,63
    //   - Physical offsets: 0, 32, 64, 96, ... (stride = 32 FP16 = 64 bytes)
    //
    // Bank conflict analysis (read phase):
    //   - Reading column requires stride-32 access (64 bytes)
    //   - Bank offset: (64 bytes / 4) % 32 = 16
    //   - Thread 0:  offset 0   → bank 0
    //   - Thread 1:  offset 64  → bank 16
    //   - Thread 2:  offset 128 → bank 0  ← CONFLICT!
    //   - Thread 3:  offset 192 → bank 16 ← CONFLICT!
    //   - Pattern: 64 threads use only 2 banks (0 and 16) → 32-way conflicts!
    //
    // Expected profiling results:
    //   - ~12-way conflicts per LDS instruction (1,244% conflict rate)
    //   - This is what we're trying to fix with XOR swizzling!
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        return make_naive_tensor_descriptor(
            make_tuple(kK, kM),
            make_tuple(number<1>{}, number<kK>{}));
    }

    // Row-major [M, K] distribution
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for FP16
        constexpr index_t K0 = kK / K1;                 // 4
        constexpr index_t M2 = 64 / K0;                 // 16
        constexpr index_t M1 = kBlockSize / 64;         // 4
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

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        // LDS descriptors
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        constexpr auto lds_desc_km = MakeLdsDescriptorKM();
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_km);

        // Distributions
        constexpr auto dist_mk = MakeDistributionMK();

        // Distribution for [K, M]
        constexpr index_t M1 = 16 / sizeof(DataType);
        constexpr index_t M0 = kM / M1;
        constexpr index_t K2 = 64 / M0;
        constexpr index_t K1 = kBlockSize / 64;
        constexpr index_t K0 = kK / (K2 * K1);

        constexpr auto dist_km = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});

        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        auto lds_window_km = make_tile_window(
            lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        // Loop over K dimension in tiles of kK
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // Global input descriptor
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}));

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            // Load from global and store to LDS
            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);

            block_sync_lds();

            // Amplify bank conflicts by reading multiple times
            constexpr int num_iterations = 1000;
            for(int iter = 0; iter < num_iterations; ++iter)
            {
                // Read transposed - BANK CONFLICTS HERE!
                (void)load_tile(lds_window_km);

                if(threadIdx.x == 0 && iter == num_iterations - 1)
                {
                    lds[0] = lds[0] + DataType(0);
                }

                block_sync_lds();
            }

            // Write transposed output
            block_sync_lds();

            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}));

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            auto reg_final = load_tile(lds_window_km);
            store_tile(gmem_window_out, reg_final);

            block_sync_lds();
        }
    }
};

int main()
{
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║ Tutorial 11: Plain Transpose (Bank Conflicts)    ║\n";
    std::cout << "║ Uses: Plain LDS, CK Tile API                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    constexpr index_t M = 256;
    constexpr index_t K = 128;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);

    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            h_input[m * K + k] = static_cast<DataType>(m * 1000 + k);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "Configuration:\n";
    std::cout << "  Input:  [" << M << ", " << K << "] (row-major)\n";
    std::cout << "  Output: [" << K << ", " << M << "] (transposed)\n";
    std::cout << "  LDS: Plain (no XOR) - expect bank conflicts!\n";
    std::cout << "  API: tensor_view, tile_window, load_tile, store_tile\n\n";

    stream_config stream;
    constexpr index_t lds_size = PlainTransposeKernel<DataType>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     PlainTransposeKernel<DataType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    d_output.FromDevice(h_output.data(), K * M * sizeof(DataType));

    bool passed = true;
    index_t error_count = 0;

    for(index_t k = 0; k < K && error_count < 10; ++k)
    {
        for(index_t m = 0; m < M && error_count < 10; ++m)
        {
            DataType expected = h_input[m * K + k];
            DataType actual = h_output[k * M + m];

            if(bit_cast<uint16_t>(expected) != bit_cast<uint16_t>(actual))
            {
                std::cout << "Error at [" << k << "][" << m << "]: "
                          << "expected " << static_cast<float>(expected)
                          << ", got " << static_cast<float>(actual) << "\n";
                error_count++;
                passed = false;
            }
        }
    }

    std::cout << "Result: " << (passed ? "✓ PASSED" : "✗ FAILED");
    std::cout << " (verified full [" << K << ", " << M << "])\n\n";

    if(passed)
    {
        std::cout << "╔═══════════════════════════════════════════════════╗\n";
        std::cout << "║ Profile with rocprofv3 to measure bank conflicts ║\n";
        std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
        std::cout << "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "          -d /tmp/plain -- ./bin/aa_tutorial_11_plain_transpose\n\n";
    }

    return passed ? 0 : 1;
}
