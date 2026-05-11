// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11j: XOR Test with REAL TRANSPOSE using CK Tile API
 *
 * This implements matrix transpose properly using:
 * - XOR descriptor (not raw shared memory)
 * - tensor_view and tile_window (not direct pointer access)
 * - load_tile/store_tile for all LDS access
 *
 * Pattern:
 * 1. Load from global [M, K]
 * 2. Store to LDS [M, K] with XOR descriptor
 * 3. Read from LDS with transposed access pattern (creates bank conflicts!)
 * 4. Store to global [K, M] transposed output
 *
 * The transpose happens by using the SAME physical LDS buffer with DIFFERENT
 * logical shapes: write as [M,K], read as [K,M]. This creates strided access
 * which triggers bank conflicts unless XOR swizzling eliminates them.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct RealTransposeKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // LDS descriptor for [M, K] - with OPTIONAL XOR on write
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        if constexpr (UseXor)
        {
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
            return make_naive_tensor_descriptor_packed(make_tuple(kM, kK));
        }
    }

    // Transposed [K, M] LDS descriptor - must match the write XOR pattern!
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        if constexpr (UseXor)
        {
            // Critical: The write XOR permutes physical addresses as f(m,k)
            // For transpose read, we need f(m,k) where we supply [k,m] as [m,k]
            // This means the descriptor dimensions are [K,M] but XOR uses [M,K] pattern

            constexpr auto DataTypeSize = sizeof(DataType);
            // Use SAME layer calculation as write (based on kK, not kM!)
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            // Step 1: Reshape [K,M] with same structure as write [M,K]
            // Logical [K,M] but treat K as if it's M, M as if it's K
            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            // Step 2: Apply SAME XOR transform as write
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

            // Step 4: Merge back to [K, M] - SWAPPED order from write
            constexpr auto lds_desc = transform_tensor_descriptor(
                lds_desc_unmerged,
                make_tuple(
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))),
                make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            // Plain transposed descriptor - will have bank conflicts!
            return make_naive_tensor_descriptor(
                make_tuple(kK, kM),
                make_tuple(number<1>{}, number<kK>{}));
        }
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

        // Setup TWO LDS descriptors and distributions for transpose

        // Descriptor 1: [M, K] for writing to LDS (with optional XOR)
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        // Distribution 1: [M, K] row-major
        constexpr auto dist_mk = MakeDistributionMK();

        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        // Loop over K dimension in tiles of kK
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // Global input descriptor with runtime strides
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}));  // Strides: M has stride K (runtime), K has stride 1

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            // Load from global and store to LDS using CK Tile API
            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);

            block_sync_lds();

            // TRANSPOSE HAPPENS HERE:
        // Physical layout in LDS: [M, K] with optional XOR permutation
        //   - UseXor=false: Plain row-major, element [m][k] at offset m*kK + k
        //   - UseXor=true: XOR permuted addresses to avoid conflicts
        //
        // Create logical [K, M] view for transposed access:
        //   - UseXor=false: Plain transpose with stride-kK (bank conflicts!)
        //   - UseXor=true: Matching XOR transpose (eliminates bank conflicts!)
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();

        auto lds_view_km = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_km);

        // Simple distribution for [K, M]
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

            auto lds_window_km = make_tile_window(
                lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

            // Iterate to amplify conflicts
            constexpr int num_iterations = 1000;
            for(int iter = 0; iter < num_iterations; ++iter)
            {
                // Read transposed - this creates bank conflicts!
                // The lds_window_km uses the transposed descriptor with stride-kK access
                (void)load_tile(lds_window_km);

                // Force the compiler to not optimize away the load
                if(threadIdx.x == 0 && iter == num_iterations - 1)
                {
                    // Dummy write to prevent dead code elimination
                    lds[0] = lds[0] + DataType(0);
                }

                block_sync_lds();
            }

            // Write transposed output [K, M] using CK Tile API
            block_sync_lds();

            // Global output descriptor with runtime strides
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}));  // Strides: K has stride M (runtime), M has stride 1

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            // Read transposed from LDS and write to global using CK Tile API
            auto reg_final = load_tile(lds_window_km);
            store_tile(gmem_window_out, reg_final);

            block_sync_lds();
        } // End K loop
    }
};

template<bool UseXor>
bool run_test(const std::string& test_name)
{
    std::cout << "\n========================================\n";
    std::cout << test_name << "\n";
    std::cout << "========================================\n\n";

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
    std::cout << "  XOR: " << (UseXor ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Using: XOR descriptor, tensor_view, tile_window\n\n";

    stream_config stream;
    constexpr index_t lds_size = RealTransposeKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     RealTransposeKernel<DataType, UseXor>{},
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

    // Verify the FULL transpose: output[K, M] should equal input[M, K] transposed
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
    std::cout << " (verified full [" << K << ", " << M << "])\n";
    return passed;
}

int main()
{
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║ Tutorial 11j: Transpose with CK Tile API         ║\n";
    std::cout << "║ Uses: XOR descriptor, tensor_view, tile_window   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n";

    bool plain_passed = run_test<false>("Test 1: Plain LDS");
    bool xor_passed = run_test<true>("Test 2: XOR LDS");

    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS: " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR LDS:   " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "Profile command:\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/transpose -- ./bin/aa_tutorial_11_xor_real_transpose\n\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}

