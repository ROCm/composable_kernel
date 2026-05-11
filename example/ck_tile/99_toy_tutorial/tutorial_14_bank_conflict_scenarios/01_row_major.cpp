// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14, Scenario 1: Row-Major Layout (Baseline)
 *
 * Demonstrates 4-way bank conflicts when reading columns from row-major LDS.
 *
 * Layout: Standard row-major [M, K] with stride K
 * Write: Row-wise (conflict-free)
 * Read: Column-wise / transpose pattern (4-way bank conflicts)
 *
 * Bank conflict analysis for 64-byte rows (32 FP16 elements):
 *   Row 0, col 0: address 0   -> bank 0
 *   Row 1, col 0: address 64  -> bank 16
 *   Row 2, col 0: address 128 -> bank 0  (repeat!)
 *   Row 3, col 0: address 192 -> bank 16 (repeat!)
 *   ...
 *   Pattern: {0, 16, 0, 16, ...} -> 4-way conflict
 *
 * Expected: HIGH bank conflicts on transpose read
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct RowMajorTransposeKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;   // Rows
    static constexpr index_t kK = 32;   // Columns

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // LDS descriptor for writing [M, K] - plain row-major
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
    }

    // Transposed LDS descriptor for reading [K, M] - causes bank conflicts!
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        return make_naive_tensor_descriptor(
            make_tuple(number<kK>{}, number<kM>{}),
            make_tuple(number<1>{}, number<kK>{}));  // Strided access -> conflicts
    }

    // Distribution for [M, K]
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

    // Distribution for [K, M]
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionKM()
    {
        constexpr index_t M1 = 16 / sizeof(DataType);  // 8
        constexpr index_t M0 = kM / M1;                 // 8
        constexpr index_t K2 = 64 / M0;                 // 8
        constexpr index_t K1 = kBlockSize / 64;         // 4
        constexpr index_t K0 = kK / (K2 * K1);          // 1

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M, index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        // LDS descriptors
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();

        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(lds, lds_desc_km);

        // Distributions
        constexpr auto dist_mk = MakeDistributionMK();
        constexpr auto dist_km = MakeDistributionKM();

        auto lds_window_mk = make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);
        auto lds_window_km = make_tile_window(lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        // Loop over K dimension
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // Global input [M, K]
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            // Load from global -> store to LDS
            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);
            block_sync_lds();

            // Transpose read from LDS (BANK CONFLICTS HERE!)
            auto reg_transposed = load_tile(lds_window_km);
            block_sync_lds();

            // Global output [K, M]
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            // Store transposed to global
            store_tile(gmem_window_out, reg_transposed);
            block_sync_lds();
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.1: Row-Major LDS (Baseline)\n";
    std::cout << "========================================\n\n";

    std::cout << "Layout: Row-major [M, K] = [64, 32]\n";
    std::cout << "Write: Row-wise (conflict-free)\n";
    std::cout << "Read: Column-wise transpose (4-way bank conflicts expected)\n\n";

    constexpr index_t M = 65536;
    constexpr index_t K = 256;

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
    if(num_iters > 1)
        std::cout << "  Iterations: " << num_iters << " (warmup: " << num_warmup << ")\n";
    std::cout << "\n";

    stream_config stream;
    constexpr index_t lds_size = RowMajorTransposeKernel<DataType>::GetStaticLdsSize();

    // Warmup
    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         RowMajorTransposeKernel<DataType>{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,
                         static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                         static_cast<DataType*>(d_output.GetDeviceBuffer()),
                         M, K));
    }
    hip_check_error(hipDeviceSynchronize());

    // Timed run
    hipEvent_t start, stop;
    hip_check_error(hipEventCreate(&start));
    hip_check_error(hipEventCreate(&stop));
    hip_check_error(hipEventRecord(start, nullptr));

    for(int i = 0; i < num_iters; i++)
    {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         RowMajorTransposeKernel<DataType>{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,
                         static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                         static_cast<DataType*>(d_output.GetDeviceBuffer()),
                         M, K));
    }

    hip_check_error(hipEventRecord(stop, nullptr));
    hip_check_error(hipDeviceSynchronize());

    float elapsed_ms = 0;
    hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));
    float avg_time_us = (elapsed_ms * 1000.0f) / num_iters;

    hip_check_error(hipEventDestroy(start));
    hip_check_error(hipEventDestroy(stop));

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

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";

    if(num_iters > 1)
    {
        std::cout << "\nPerformance:\n";
        std::cout << "  Average time: " << avg_time_us << " μs\n";
        float bandwidth_gbs = (M * K * sizeof(DataType) * 2) / (avg_time_us * 1000.0f);
        std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s\n";
    }

    std::cout << "\nExpected: HIGH bank conflicts (4-way) on transpose read\n";
    std::cout << "Reason: 64-byte row stride causes addresses to hit only 2 banks\n";

    return passed;
}

int main(int argc, char* argv[])
{
    int num_iters = 1;
    int num_warmup = 0;

    // Simple argument parsing
    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--bench")
        {
            num_iters = 100;
            num_warmup = 5;
        }
    }

    return run_test(num_iters, num_warmup) ? 0 : 1;
}
