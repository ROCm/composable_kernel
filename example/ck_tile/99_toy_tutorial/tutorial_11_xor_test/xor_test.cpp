// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11: XOR Descriptor Minimal Test
 *
 * This is a minimal test to understand how XOR-based LDS descriptors work.
 * We'll create a simple kernel that:
 * 1. Loads data from global memory to registers
 * 2. Stores to LDS using XOR-swizzled descriptor
 * 3. Loads from LDS using the SAME XOR descriptor
 * 4. Stores back to global memory
 *
 * If the XOR descriptor works correctly, output should match input.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// Minimal XOR test kernel
template<typename DataType>
struct XorTestKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;   // Tile size M
    static constexpr index_t kK = 32;   // Tile size K
    static constexpr index_t kKPack = 8; // Vector width

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);  // 64*32*2 = 4096 bytes
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        extern __shared__ char smem[];
        DataType* p_lds = reinterpret_cast<DataType*>(smem);

        const index_t tid = get_thread_id();
        const index_t block_m = get_block_id() * kM;

        // Bounds check
        if(block_m >= M) return;

        // ========================================================================
        // Create XOR-swizzled LDS descriptor (same as 02_gemm)
        // ========================================================================

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

        // ====================================================================
        // Direct access using calculate_offset()
        // ====================================================================

        // Each thread handles multiple elements
        constexpr index_t elements_per_thread = (kM * kK) / kBlockSize;

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            const index_t elem_id = tid * elements_per_thread + i;
            if(elem_id < kM * kK)
            {
                const index_t m = elem_id / kK;
                const index_t k = elem_id % kK;

                const index_t global_m = block_m + m;
                if(global_m < M && k < K)
                {
                    // Load from global
                    DataType value = input[global_m * K + k];

                    // Calculate physical LDS offset using XOR descriptor
                    constexpr auto idx_dims = decltype(lds_desc)::get_num_of_dimension();
                    array<index_t, idx_dims> logical_idx;
                    logical_idx[number<0>{}] = m;
                    logical_idx[number<1>{}] = k;

                    const index_t physical_offset = lds_desc.calculate_offset(logical_idx);
                    p_lds[physical_offset] = value;
                }
            }
        }

        block_sync_lds();

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            const index_t elem_id = tid * elements_per_thread + i;
            if(elem_id < kM * kK)
            {
                const index_t m = elem_id / kK;
                const index_t k = elem_id % kK;

                const index_t global_m = block_m + m;
                if(global_m < M && k < K)
                {
                    constexpr auto idx_dims = decltype(lds_desc)::get_num_of_dimension();
                    array<index_t, idx_dims> logical_idx;
                    logical_idx[number<0>{}] = m;
                    logical_idx[number<1>{}] = k;

                    const index_t physical_offset = lds_desc.calculate_offset(logical_idx);
                    DataType value = p_lds[physical_offset];
                    output[global_m * K + k] = value;
                }
            }
        }
    }
};

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 11: XOR Descriptor Test\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 128;
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

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "Test configuration:\n";
    std::cout << "  M×K: " << M << "×" << K << "\n";
    std::cout << "  Tile: 64×32\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n\n";

    stream_config stream;
    constexpr index_t lds_size = XorTestKernel<DataType>::GetStaticLdsSize();

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    launch_kernel(stream,
                 make_kernel<block_size>(
                     XorTestKernel<DataType>{},
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
        // Compare bit patterns for exact equality (no floating point comparison)
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
        std::cout << "SUCCESS! XOR descriptor correctly maps logical [M,K] to physical LDS.\n";
        std::cout << "Data written with XOR swizzle can be read back correctly.\n";
        std::cout << "\nNOTE: This test uses direct calculate_offset() access.\n";
        std::cout << "Tutorial 10 uses tile_window with distributions - that's the next complexity to investigate.\n";
    }
    else
    {
        std::cout << "FAILED! XOR descriptor has issues.\n";
        std::cout << "Either the transform is wrong OR the access pattern is incompatible.\n";
    }

    return passed ? 0 : 1;
}
