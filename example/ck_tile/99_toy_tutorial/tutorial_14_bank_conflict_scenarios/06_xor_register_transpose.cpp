// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.06: XOR + register-level transpose (Option A, limited)
 *
 * This is the first step away from the dual-descriptor pattern of
 * `04_row_major_xor.cpp`. It demonstrates the production descriptor
 * convention: a SINGLE [M, K] XOR descriptor used for both LDS writes
 * and LDS reads, with a K-vectorized distribution on both sides so the
 * vectorization axis aligns with the XOR pack axis (`kKPack=8` on K).
 * The transpose itself is performed in registers by `transpose_tile2d`
 * (compiles to `v_perm_b32`), not by a different LDS descriptor.
 *
 * Honest caveats (important for understanding the bank-conflict story):
 *
 * 1. The LDS round-trip in THIS kernel is semantically a pass-through.
 *    Because the write distribution and the read distribution are
 *    identical (`MakeDistributionMK` on both sides), every thread reads
 *    back the exact data it just wrote. The compiler cannot prove this
 *    through the `block_sync_lds()` barrier, so it keeps the round-trip
 *    -- but the data doesn't actually need to travel through LDS.
 *
 *    Evidence: the compiler emits one `ds_write_b128` followed by one
 *    `ds_read_b128`, both using the same address register and the same
 *    data registers.
 *
 * 2. The reason we get away with this for a [64, 32] block transpose is
 *    that each warp happens to own a self-contained 16x32 sub-tile of
 *    the input that can be transposed into a 32x16 sub-tile of the
 *    output without any data from other warps. `transpose_tile2d` does
 *    that within-wavefront permutation via `v_perm_b32`, no cross-warp
 *    exchange needed.
 *
 * 3. See `07_xor_cross_warp_lds_transpose.cpp` for the full story: the
 *    write and read distributions differ so LDS genuinely shuffles
 *    data between warps, and the read is logically "transposed" from
 *    the write's thread-to-(m, k) mapping -- yet both sides stay
 *    K-vectorized over the same XOR descriptor, so XOR still spreads
 *    the accesses and no conflicts appear.
 *
 * What this example DOES still faithfully show:
 *   - The production descriptor pattern (single XOR desc, K-vec both sides).
 *   - The compiler emitting `ds_write_b128`/`ds_read_b128` vector ops
 *     instead of 8 scalar `ds_read_u16 offset:128` variants, because
 *     the vectorization axis matches the XOR pack axis.
 *   - Zero `SQ_LDS_BANK_CONFLICT` -- here mainly because the LDS traffic
 *     is trivial (write and read at the same XOR-permuted address).
 *
 * Measured on MI355 / gfx950 (64 LDS banks), M=256, K=128, fp16:
 *
 *                                   SQ_LDS_BANK_CONFLICT   SQ_INSTS_LDS   single launch
 *   04 row_major_xor                                3072            576        11.84 us
 *   05 xor_plus_padding                                0            576        11.60 us
 *   06 xor_register_transpose (this)                   0            128        10.36 us
 *   07 xor_cross_warp_lds_transpose                    0            128         9.40 us
 *
 * LDS instruction count drops 4.5x compared to 04 purely because each
 * LDS access is now a single 128-bit vector op per thread. Note the
 * XOR descriptor below uses `get_n_lds_banks()` (64 on MI355, 32 on
 * MI300) and scales the first XOR dim by `RowMul` accordingly --
 * hardcoding 32 would undersize the scramble on MI355 and leak
 * conflicts back in.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct XorTransposedDistributionKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM         = 64;
    static constexpr index_t kK         = 32;
    static constexpr index_t kKPack     = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // --------------------------------------------------------------------
    // LDS descriptor: SINGLE [M, K] XOR descriptor, used for BOTH writes
    // and reads. Same as the XOR descriptor in 04_row_major_xor.cpp's
    // MakeLdsDescriptorMK(). No separate [K, M] variant.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        constexpr auto DataTypeSize = sizeof(DataType);
        // Architecture-aware bank count: 32 on gfx942, 64 on gfx950 (MI355).
        // Hardcoding 32 here would undersize the XOR scramble on 64-bank HW.
        constexpr index_t NBanks            = get_n_lds_banks();
        constexpr index_t LdsBandwidthBytes = NBanks * get_n_dwords_per_128b();
        constexpr index_t MLdsLayerRaw      = LdsBandwidthBytes / kK / DataTypeSize;
        constexpr index_t MLdsLayer         = MLdsLayerRaw < 1 ? 1 : MLdsLayerRaw;
        constexpr index_t RowMul            = (NBanks == 64) ? 2 : 1;

        // Step 1: initial 3D shape [K/Pack*Layer, M/Layer, Pack]
        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kK / kKPack * MLdsLayer>{},
                       number<kM / MLdsLayer>{},
                       number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        // Step 2: XOR transform on dims (1, 0). First arg scaled by RowMul
        // so the XOR covers the full 64-bank width on gfx950.
        constexpr auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_tuple(make_xor_transform(
                           make_tuple(number<kM / MLdsLayer * RowMul>{},
                                      number<kK / kKPack * MLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        // Step 3: unmerge layer dimension
        constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
            lds_desc_permuted,
            make_tuple(
                make_unmerge_transform(make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                make_pass_through_transform(number<kM / MLdsLayer>{}),
                make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        // Step 4: merge back to [M, K]
        constexpr auto lds_desc = transform_tensor_descriptor(
            lds_desc_unmerged,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc;
    }

    // --------------------------------------------------------------------
    // [M, K] input distribution: K-vectorized (per-thread shape (1, 8)).
    // Used for BOTH the global load AND the LDS write/read windows. This
    // is the key change vs 04: by using the SAME K-vectorized distribution
    // for the LDS read, we never ask the compiler for M-vectorized reads,
    // so `ds_read offset:128` cannot appear.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType); // 8 for FP16
        constexpr index_t K0 = kK / K1;               // 4
        constexpr index_t M2 = 64 / K0;               // 16
        constexpr index_t M1 = kBlockSize / 64;       // 4
        constexpr index_t M0 = kM / (M2 * M1);        // 1

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // --------------------------------------------------------------------
    // [K, M] "shuffled" output distribution. This is the 2D transpose of
    // MakeDistributionMK, produced by swapping the two H groups and
    // reversing the Y-minor mapping -- the exact rule required by
    // `transpose_tile2d`. It is the [K, M] analogue of tutorial 13's
    // MakeBShuffledDistribution.
    //
    // Per-thread Y shape: (K1=8, M0=1). `transpose_tile2d` requires the
    // output Y-shape to be the REVERSE of the input's (1, 8) => (8, 1).
    //
    // Semantics: each thread holds 8 K values at a single M position.
    // Writes to global [K, M] are therefore strided along K within the
    // thread, but coalesced across threads in the warp (different M per
    // lane). That's fine for the bandwidth test and -- more importantly
    // for this tutorial -- the LDS read side is already conflict-free.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionKM_Shuffled()
    {
        constexpr index_t K1 = 16 / sizeof(DataType); // 8
        constexpr index_t K0 = kK / K1;               // 4
        constexpr index_t M2 = 64 / K0;               // 16
        constexpr index_t M1 = kBlockSize / 64;       // 4
        constexpr index_t M0 = kM / (M2 * M1);        // 1

        // Hs swapped: Hs[0] = K-group, Hs[1] = M-group.
        // P mapping follows the swap: P0 -> M1 (warp id),
        //                             P1 -> (M2, K0) (lane id).
        // Ys: Y0 -> K1 (size 8), Y1 -> M0 (size 1). Per-thread (8, 1),
        //     which is the reverse of the input's (1, 8) per-thread shape.
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<K0, K1>, sequence<M0, M1, M2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{});
    }

    // --------------------------------------------------------------------
    // [K, M] "native" output distribution: M-vectorized, used for the
    // final global store. Per-thread shape (K=1, M=8). This is the
    // distribution that gives coalesced writes to the [K, M] row-major
    // global tensor. We re-pack data from the shuffled distribution
    // into this one via `transpose_tile2d`.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionKM_Native()
    {
        constexpr index_t M1 = 16 / sizeof(DataType); // 8
        constexpr index_t M0 = kM / M1;               // 8
        constexpr index_t K2 = 64 / M0;               // 8
        constexpr index_t K1 = kBlockSize / 64;       // 4
        constexpr index_t K0 = kK / (K2 * K1);        // 1

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                   DataType* __restrict__ output,
                                   index_t M,
                                   index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M)
            return;

        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);

        constexpr auto dist_mk          = MakeDistributionMK();
        constexpr auto dist_km_shuffled = MakeDistributionKM_Shuffled();

        // SAME K-vectorized distribution for LDS write AND LDS read.
        auto lds_window_mk =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // [1] Global [M, K] load -- K-vectorized.
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});
            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);
            auto gmem_window_in =
                make_tile_window(gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            auto reg_mk = load_tile(gmem_window_in);

            // [2] Store to LDS [M, K] XOR, K-vectorized. Matches XOR pack axis.
            store_tile(lds_window_mk, reg_mk);
            block_sync_lds();

            // [3] Read from LDS [M, K] XOR, K-vectorized. Also matches XOR
            //     pack axis. This is the whole point of this tutorial:
            //     conflict-free LDS reads by keeping vectorization aligned
            //     with the XOR pack axis.
            auto reg_mk_from_lds = load_tile(lds_window_mk);
            block_sync_lds();

            // [4] Register-transpose [M, K] -> [K, M] via transpose_tile2d.
            //     Purely a register permutation; no LDS traffic.
            auto reg_km_shuffled =
                make_static_distributed_tensor<DataType>(dist_km_shuffled);
            transpose_tile2d(reg_km_shuffled, reg_mk_from_lds);

            // [5] Store to global [K, M] using the shuffled distribution.
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}),
                number<16 / sizeof(DataType)>{},
                number<1>{});
            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);
            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km_shuffled);

            store_tile(gmem_window_out, reg_km_shuffled);
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.06: XOR + Transposed Distribution (Option A)\n";
    std::cout << "========================================\n\n";

    std::cout << "Single [M, K] XOR descriptor (same as 04) used for BOTH\n";
    std::cout << "LDS writes and LDS reads with a K-vectorized distribution.\n";
    std::cout << "Transpose happens in registers via transpose_tile2d.\n\n";

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

    constexpr index_t kM         = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size      = (M + kM - 1) / kM;

    std::cout << "Configuration:\n";
    std::cout << "  Input:  [" << M << ", " << K << "] (row-major)\n";
    std::cout << "  Output: [" << K << ", " << M << "] (transposed)\n";
    if(num_iters > 1)
        std::cout << "  Iterations: " << num_iters << " (warmup: " << num_warmup << ")\n";
    std::cout << "\n";

    stream_config stream;
    constexpr index_t lds_size = XorTransposedDistributionKernel<DataType>::GetStaticLdsSize();

    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          XorTransposedDistributionKernel<DataType>{},
                          dim3(grid_size),
                          dim3(block_size),
                          lds_size,
                          static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                          static_cast<DataType*>(d_output.GetDeviceBuffer()),
                          M,
                          K));
    }
    hip_check_error(hipDeviceSynchronize());

    hipEvent_t start, stop;
    hip_check_error(hipEventCreate(&start));
    hip_check_error(hipEventCreate(&stop));
    hip_check_error(hipEventRecord(start, nullptr));

    for(int i = 0; i < num_iters; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          XorTransposedDistributionKernel<DataType>{},
                          dim3(grid_size),
                          dim3(block_size),
                          lds_size,
                          static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                          static_cast<DataType*>(d_output.GetDeviceBuffer()),
                          M,
                          K));
    }

    hip_check_error(hipEventRecord(stop, nullptr));
    hip_check_error(hipDeviceSynchronize());

    float elapsed_ms = 0;
    hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));
    float avg_time_us = (elapsed_ms * 1000.0f) / num_iters;

    hip_check_error(hipEventDestroy(start));
    hip_check_error(hipEventDestroy(stop));

    d_output.FromDevice(h_output.data(), K * M * sizeof(DataType));

    bool passed          = true;
    index_t error_count  = 0;

    for(index_t k = 0; k < K && error_count < 10; ++k)
    {
        for(index_t m = 0; m < M && error_count < 10; ++m)
        {
            DataType expected = h_input[m * K + k];
            DataType actual   = h_output[k * M + m];

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
        std::cout << "  Average time: " << avg_time_us << " us\n";
        float bandwidth_gbs = (M * K * sizeof(DataType) * 2) / (avg_time_us * 1000.0f);
        std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s\n";
    }

    std::cout << "\nExpected: LOW bank conflicts (XOR-pack-aligned reads and writes)\n";
    std::cout << "Verify with:\n";
    std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -- ./aa_tutorial_14_06_xor_transposed\n";

    return passed;
}

int main(int argc, char* argv[])
{
    int num_iters  = 1;
    int num_warmup = 0;

    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--bench")
        {
            num_iters  = 100;
            num_warmup = 5;
        }
    }

    return run_test(num_iters, num_warmup) ? 0 : 1;
}
