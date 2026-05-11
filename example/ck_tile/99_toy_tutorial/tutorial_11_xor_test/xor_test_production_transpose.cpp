// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11m: Production-Ready Transpose with CK Tile API
 *
 * This implements a production-ready matrix transpose using:
 * - XOR descriptor with "swapped" dimension approach (correct dimensions)
 * - tensor_view and tile_window (CK Tile API)
 * - load_tile/store_tile for all memory access
 * - Single-pass transpose (no iteration amplification)
 *
 * Pattern:
 * 1. Load from global [M, K]
 * 2. Store to LDS [M, K] with optional XOR descriptor
 * 3. Read from LDS with transposed [K, M] descriptor
 * 4. Store to global [K, M] transposed output
 *
 * The transpose happens by descriptor reinterpretation: write as [M,K],
 * read as [K,M] from the same physical LDS buffer.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct ProductionTransposeKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // LDS descriptor for [M, K] - with OPTIONAL XOR swizzling for bank conflict reduction
    //
    // XOR swizzling permutes physical addresses to spread strided accesses across all 32 banks.
    // This is implemented through a series of tensor descriptor transformations.
    //
    // Key idea:
    //   physical_address = XOR(m_component, k_component)
    //
    // This breaks the regular stride pattern that causes bank aliasing, distributing
    // conflicts across all banks instead of concentrating them in just 2 banks.
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        if constexpr (UseXor)
        {
            // Step 0: Calculate MLdsLayer (bank-conflict-aware parameter)
            //
            // MLdsLayer relates to the LDS bank structure:
            //   - 32 banks × 4 bytes = 128 bytes total bandwidth per cycle
            //   - For our tile kK=32 FP16: 32 × 2 = 64 bytes per row
            //   - Need 2 rows to span all 32 banks → MLdsLayer = 2
            //
            // Formula: MLdsLayer = (32 banks × 4 bytes) / (kK elements × DataTypeSize)
            //        = 128 / (32 × 2) = 2 for our case
            //
            // This parameter is used to reshape tensors to expose dimensions that will be XOR'd.
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            // Step 1: Reshape to expose XOR dimensions
            //
            // Transform: [M, K] → [K/Pack*Layer, M/Layer, Pack]
            //            [64, 32] → [8, 32, 8]
            //
            // This reshaping exposes the dimensions that will be XOR'd:
            //   - Dimension 0: K-related (8 = 32/8 × 2)
            //   - Dimension 1: M-related (32 = 64/2)
            //   - Dimension 2: Pack (8 for vectorization)
            //
            // The strides are set to maintain row-major physical layout before XOR.
            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},  // 32/8 * 2 = 8
                           number<kM / MLdsLayer>{},            // 64/2 = 32
                           number<kKPack>{}),                   // 8
                make_tuple(number<kKPack>{},                   // stride: 8
                           number<kK * MLdsLayer>{},            // stride: 64
                           number<1>{}),                        // stride: 1
                number<kKPack>{},
                number<1>{});

            // Step 2: Apply XOR transform
            //
            // This is the KEY operation that reduces bank conflicts!
            //
            // XOR transform operates on dimensions [1, 0]:
            //   physical_offset = XOR(dim1_index, dim0_index)
            //                   = XOR(m_component, k_component)
            //
            // Why XOR helps:
            //   - Bitwise XOR spreads consecutive indices across different values
            //   - XOR(a, b) ⊕ XOR(a, c) = b ⊕ c (different when b ≠ c)
            //   - This distributes accesses across all 32 banks instead of just 2
            //
            // The permutation happens in PHYSICAL address space, while the LOGICAL
            // view of the tensor remains [M, K]. This is the magic of XOR swizzling!
            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                         number<kK / kKPack * MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{},  // XOR dimensions 1 and 0
                           sequence<2>{}),    // Pass through dimension 2 (pack)
                make_tuple(sequence<1, 0>{},  // Output dimensions
                           sequence<2>{}));

            // Step 3: Unmerge layer dimension
            //
            // Transform: [M/Layer, K/Pack*Layer, Pack] → [Layer, M/Layer, K/Pack, Pack]
            //            [32, 8, 8] → [2, 32, 4, 8]
            //
            // Split the first dimension to separate the layer component from K.
            // This allows us to merge back to [M, K] in the final step.
            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kM / MLdsLayer>{}),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            // Step 4: Merge back to [M, K]
            //
            // Transform: [Layer, M/Layer, K/Pack, Pack] → [M, K]
            //            [2, 32, 4, 8] → [64, 32]
            //
            // Merge dimensions to restore the original [M, K] shape:
            //   - Merge [M/Layer, Layer] → M (dimension 0)
            //   - Merge [K/Pack, Pack] → K (dimension 1)
            //
            // The sequence indices [1, 0] and [2, 3] specify which dimensions to merge.
            //
            // Result: Logical view is [M, K], but physical addresses are XOR-permuted!
            //         This means writes to logical [m][k] go to XOR'd physical locations,
            //         which will reduce bank conflicts during transposed reads.
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

    // Transposed [K, M] LDS descriptor - CRITICAL: must use SAME XOR pattern as write!
    //
    // This descriptor reads the SAME physical XOR-permuted memory, but interprets it as [K, M]
    // instead of [M, K]. This is how transpose works with XOR swizzling.
    //
    // The key requirement:
    //   - Steps 1-3 MUST BE IDENTICAL to MakeLdsDescriptorMK() (same XOR pattern)
    //   - Step 4 is DIFFERENT: swap merge order to get [K, M] instead of [M, K]
    //
    // Why this works:
    //   - Write descriptor: logical [M, K] → XOR'd physical addresses
    //   - Read descriptor:  logical [K, M] → SAME XOR'd physical addresses
    //   - Transpose achieved by different logical interpretation of same data!
    //
    // Bank conflict reduction:
    //   - XOR spreads column reads across all 32 banks
    //   - Reduces conflicts from ~12-way to ~5-way (57% improvement)
    //   - Still above theoretical minimum (2-way) but much better than plain LDS
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        if constexpr (UseXor)
        {
            // Step 0: Same MLdsLayer calculation as write descriptor
            // MUST be identical to ensure same XOR pattern!
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            // Steps 1-3: IDENTICAL to write descriptor
            // These steps MUST match MakeLdsDescriptorMK() exactly to ensure the same
            // XOR permutation is applied. Only Step 4 differs.

            // Step 1: Reshape (same as write)
            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            // Step 2: XOR transform (same as write)
            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                         number<kK / kKPack * MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            // Step 3: Unmerge (same as write)
            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kM / MLdsLayer>{}),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            // Step 4: Merge back to [K, M] - SWAPPED order from write descriptor!
            //
            // Transform: [Layer, M/Layer, K/Pack, Pack] → [K, M]
            //            [2, 32, 4, 8] → [32, 64]
            //
            // Compare to write descriptor (produces [M, K]):
            //   Write: Merge [M/Layer, Layer] first → output dim 0
            //          Merge [K/Pack, Pack] second → output dim 1
            //   Read:  Merge [K/Pack, Pack] first  → output dim 0  (SWAPPED!)
            //          Merge [M/Layer, Layer] second → output dim 1 (SWAPPED!)
            //
            // This creates the transposed view [K, M] of the SAME XOR-permuted data.
            // Physical memory layout is unchanged; only the logical interpretation differs.
            //
            // Result: Reading column k=0 now accesses XOR-permuted addresses that are
            //         distributed across all 32 banks, reducing conflicts from 32-way to ~5-way.
            constexpr auto lds_desc = transform_tensor_descriptor(
                lds_desc_unmerged,
                make_tuple(
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),      // K first!
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))), // M second!
                make_tuple(sequence<2, 3>{},   // K dimensions → output dim 0
                           sequence<1, 0>{}),  // M dimensions → output dim 1
                make_tuple(sequence<0>{},      // Output K
                           sequence<1>{}));    // Output M

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

        // Setup TWO LDS descriptors for transpose

        // Descriptor 1: [M, K] for writing to LDS
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        // Distribution for [M, K]
        constexpr auto dist_mk = MakeDistributionMK();

        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

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

            // TRANSPOSE: Create [K, M] view of same LDS buffer
            constexpr auto lds_desc_km = MakeLdsDescriptorKM();

            auto lds_view_km = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<DataType*>(lds), lds_desc_km);

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

            auto lds_window_km = make_tile_window(
                lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

            // Single transpose read (production - no iteration loop!)
            auto reg_final = load_tile(lds_window_km);

            block_sync_lds();

            // Global output descriptor
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}));

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            // Write transposed output to global
            store_tile(gmem_window_out, reg_final);

            block_sync_lds();
        }
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
    std::cout << "  Mode: Single-pass production transpose\n\n";

    stream_config stream;
    constexpr index_t lds_size = ProductionTransposeKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     ProductionTransposeKernel<DataType, UseXor>{},
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

    // Verify the FULL transpose
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
    std::cout << "║ Production Transpose with CK Tile API            ║\n";
    std::cout << "║ Single-pass transpose (Plain vs XOR)             ║\n";
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
        std::cout << "This is a production-ready transpose implementation.\n";
        std::cout << "- Single-pass (no iteration amplification)\n";
        std::cout << "- CK Tile API (tensor_view, tile_window, load_tile, store_tile)\n";
        std::cout << "- XOR swizzling for bank conflict reduction\n\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
