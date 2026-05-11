// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.05: XOR + Padding Combined
 *
 * This combines BOTH optimization techniques:
 * 1. XOR swizzling (spreads access pattern across banks)
 * 2. Padding (changes stride to avoid bank aliasing)
 *
 * Expected: VERY LOW conflicts (best of both worlds)
 * - XOR breaks regular patterns
 * - Padding ensures coprime stride with 32 banks
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct XorPlusPaddingKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;
    static constexpr index_t kKPadded = 33;  // Add 1 for padding

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kKPadded * sizeof(DataType);  // Use padded size
    }

    // LDS descriptor for [M, K] with BOTH XOR and padding
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

        // Step 1: (B, A, C) natural row-major order, with PADDED B stride.
        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kM / MLdsLayer>{},           // B
                       number<kK / kKPack * MLdsLayer>{},  // A
                       number<kKPack>{}),                  // C
            make_tuple(number<kKPadded * MLdsLayer>{},     // stride B (PADDED!)
                       number<kKPack>{},                   // stride A
                       number<1>{}),                       // stride C
            number<kKPack>{},
            number<1>{});

        // Step 2: XOR on (B, A). A_low = A XOR (B % A).
        constexpr auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                     number<kK / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0, 1>{}, sequence<2>{}));

        // Step 3: Unmerge A -> (Layer, K/Pack). Upper: (B=0, L=1, K0=2, C=3).
        constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
            lds_desc_permuted,
            make_tuple(make_pass_through_transform(number<kM / MLdsLayer>{}),
                       make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{},    sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Step 4: Merge to [M, K].
        constexpr auto lds_desc = transform_tensor_descriptor(
            lds_desc_unmerged,
            make_tuple(
                make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kK / kKPack>{},    number<kKPack>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{},    sequence<1>{}));

        return lds_desc;
    }

    // Transposed LDS descriptor [K, M] with BOTH XOR and padding
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

        // Step 1: Same reshape as write, (B, A, C) natural order, PADDED B stride.
        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kM / MLdsLayer>{},           // B
                       number<kK / kKPack * MLdsLayer>{},  // A
                       number<kKPack>{}),                  // C
            make_tuple(number<kKPadded * MLdsLayer>{},     // stride B (PADDED!)
                       number<kKPack>{},                   // stride A
                       number<1>{}),                       // stride C
            number<kKPack>{},
            number<1>{});

        // Step 2: XOR (same as write).
        constexpr auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                     number<kK / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0, 1>{}, sequence<2>{}));

        // Step 3: Unmerge (same as write).
        constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
            lds_desc_permuted,
            make_tuple(make_pass_through_transform(number<kM / MLdsLayer>{}),
                       make_unmerge_transform(
                           make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{},    sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Step 4: Merge to [K, M] -- SWAPPED output order.
        constexpr auto lds_desc = transform_tensor_descriptor(
            lds_desc_unmerged,
            make_tuple(
                make_merge_transform(make_tuple(number<kK / kKPack>{},    number<kKPack>{})),
                make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))),
            make_tuple(sequence<2, 3>{}, sequence<0, 1>{}),
            make_tuple(sequence<0>{},    sequence<1>{}));

        return lds_desc;
    }

    // Distribution for [M, K]
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kK / K1;
        constexpr index_t M2 = 64 / K0;
        constexpr index_t M1 = kBlockSize / 64;
        constexpr index_t M0 = kM / (M2 * M1);

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
        constexpr index_t M1 = 16 / sizeof(DataType);
        constexpr index_t M0 = kM / M1;
        constexpr index_t K2 = 64 / M0;
        constexpr index_t K1 = kBlockSize / 64;
        constexpr index_t K0 = kK / (K2 * K1);

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
        __shared__ DataType lds[kM * kKPadded];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();

        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(lds, lds_desc_km);

        constexpr auto dist_mk = MakeDistributionMK();
        constexpr auto dist_km = MakeDistributionKM();

        auto lds_window_mk = make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);
        auto lds_window_km = make_tile_window(lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);
            block_sync_lds();

            auto reg_transposed = load_tile(lds_window_km);
            block_sync_lds();

            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            store_tile(gmem_window_out, reg_transposed);
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.05: XOR + Padding Combined\n";
    std::cout << "========================================\n\n";

    std::cout << "Optimizations: XOR swizzling + Padding (stride 33)\n\n";

    using DataType = half_t;

    constexpr index_t M = 65536;
    constexpr index_t K = 256;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);

    for(index_t i = 0; i < M * K; i++)
        h_input[i] = static_cast<DataType>(i);

    DataType* d_input;
    DataType* d_output;
    (void)hipMalloc(&d_input, M * K * sizeof(DataType));
    (void)hipMalloc(&d_output, K * M * sizeof(DataType));

    (void)hipMemcpy(d_input, h_input.data(), M * K * sizeof(DataType), hipMemcpyHostToDevice);

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
    constexpr index_t lds_size = XorPlusPaddingKernel<DataType>::GetStaticLdsSize();

    // Warmup
    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         XorPlusPaddingKernel<DataType>{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,
                         d_input, d_output, M, K));
    }
    (void)hipDeviceSynchronize();

    // Timed run
    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    (void)hipEventRecord(start, nullptr);

    for(int i = 0; i < num_iters; i++)
    {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         XorPlusPaddingKernel<DataType>{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,
                         d_input, d_output, M, K));
    }

    (void)hipEventRecord(stop, nullptr);
    (void)hipDeviceSynchronize();

    float elapsed_ms = 0;
    (void)hipEventElapsedTime(&elapsed_ms, start, stop);

    float avg_time_us = (elapsed_ms * 1000.0f) / num_iters;

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    (void)hipMemcpy(h_output.data(), d_output, K * M * sizeof(DataType), hipMemcpyDeviceToHost);

    // Generate CPU reference transpose
    std::vector<DataType> h_reference(K * M);
    for(index_t m = 0; m < M; m++)
    {
        for(index_t k = 0; k < K; k++)
        {
            h_reference[k * M + m] = h_input[m * K + k];
        }
    }

    // Verify against CPU reference
    bool correct = true;
    index_t error_count = 0;
    for(index_t k = 0; k < K && error_count < 10; k++)
    {
        for(index_t m = 0; m < M && error_count < 10; m++)
        {
            DataType expected = h_reference[k * M + m];
            DataType actual = h_output[k * M + m];

            uint16_t exp_bits, act_bits;
            std::memcpy(&exp_bits, &expected, sizeof(uint16_t));
            std::memcpy(&act_bits, &actual, sizeof(uint16_t));

            if(exp_bits != act_bits)
            {
                std::cout << "  ERROR at output[" << k << "][" << m << "]: "
                         << "expected " << static_cast<float>(expected)
                         << " (from input[" << m << "][" << k << "]), got "
                         << static_cast<float>(actual) << "\n";
                error_count++;
                correct = false;
            }
        }
    }

    if(error_count > 0)
    {
        std::cout << "  Found " << error_count << " errors (showing first 10)\n";
    }

    std::cout << "\nResult: " << (correct ? "PASS ✓" : "FAIL ✗") << "\n";

    if(num_iters > 1)
    {
        std::cout << "\nPerformance:\n";
        std::cout << "  Average time: " << avg_time_us << " μs\n";
        float bandwidth_gbs = (M * K * sizeof(DataType) * 2) / (avg_time_us * 1000.0f);
        std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s\n";
    }

    std::cout << "\nExpected: VERY LOW conflicts (combining both techniques)\n";
    std::cout << "  - XOR: Spreads regular patterns across banks\n";
    std::cout << "  - Padding: Ensures coprime stride (33 vs 32)\n";

    (void)hipFree(d_input);
    (void)hipFree(d_output);

    return correct;
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
