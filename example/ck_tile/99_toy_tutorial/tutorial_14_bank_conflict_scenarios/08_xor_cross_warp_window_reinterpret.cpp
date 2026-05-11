// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.08: XOR + cross-warp LDS transpose, NO transpose_tile2d
 *
 * Variation of 07 that drops `transpose_tile2d` entirely. The LDS
 * shuffle between the two differently-distributed windows already puts
 * each thread's 8 fp16 values at the correct output [K, M] position --
 * the only reason 07 needed `transpose_tile2d` was to relabel the tile
 * type from [M, K]-distributed to [K, M]-distributed so `store_tile`
 * matched the window type.
 *
 * Here we achieve the same thing by REINTERPRETING the output window:
 * the global buffer is [K, M] row-major in memory, but we build a
 * window that describes it as an [M, K]-shaped view with strides
 * (1, M_global). Element (m, k) in this view resolves to physical
 * offset 1*m + M_global*k, which is exactly position of output[k, m]
 * in the [K, M] row-major buffer. Then `store_tile` works directly on
 * the [M, K]-distributed `reg_read` -- no register-type conversion
 * needed.
 *
 * Functionally identical to 07 at the ISA level (the transpose_tile2d
 * in 07 was a no-op copy elided by the compiler); the difference is
 * stylistic. 07 is the "idiomatic" ck_tile pattern. This one trades
 * a cleaner output window shape for a slightly tricky descriptor. It
 * shows that the hard work is entirely in the LDS shuffle and the
 * register "transpose" is just bookkeeping.
 *
 * Contrast with 06 (`06_xor_register_transpose.cpp`):
 *   06 used the SAME distribution for LDS write and read, so every
 *   thread reads back its own data. The LDS round-trip is a no-op
 *   there and the transpose actually happens in registers via
 *   `transpose_tile2d`. Good for illustrating the descriptor pattern,
 *   but a weak demonstration of bank-conflict avoidance because the
 *   LDS reads are trivially non-conflicting.
 *
 * What this example does differently:
 *
 *   WRITE distribution (M-partitioned by warp):
 *     Warp w writes M=[w*16, w*16+16), all K=[0, 32).
 *     Per-thread shape (M=1, K=8)  -- K-vectorized.
 *
 *   READ distribution (K-partitioned by warp):
 *     Warp w reads K=[w*8, w*8+8), all M=[0, 64).
 *     Per-thread shape (M=1, K=8)  -- still K-vectorized.
 *     But now warp w's read DATA comes from ALL four write-warps
 *     (each write-warp contributed a disjoint M range to K=[w*8,w*8+8)).
 *
 *   Cross-warp exchange: LDS is NOT a pass-through any more. A warp's
 *   read is served by physical LDS locations that four different warps
 *   wrote to; if this traffic were scattered onto the same banks,
 *   SQ_LDS_BANK_CONFLICT would be large.
 *
 *   Register transpose (`transpose_tile2d`): converts the [M, K]-shaped
 *   read tile (per-thread (1, 8)) into the [K, M]-shaped output tile
 *   (per-thread (8, 1)).
 *
 *   Global [K, M] store: 8 warp-coalesced 128-byte writes per warp
 *   per k_iter (64 threads write 64 consecutive M values at a single K).
 *
 * Why it should be conflict-free:
 *   On the LDS READ side, 64 threads of a warp each do a single
 *   ds_read_b128 at a different m_t with the SAME K-range. Without
 *   XOR the physical offsets would be m_t * kK * sizeof(T) = m_t * 64
 *   bytes apart -- stride 64 B = 16 banks, so 32 bank-conflicts across
 *   the warp. With the [M, K] XOR descriptor (pack on K, swizzle on M),
 *   those 64 addresses are spread across all 32 banks. Same argument
 *   applies on the write side.
 *
 * Tutorial 04 is the naive counterpart: it creates a SECOND [K, M]
 * descriptor by swapping merge order, keeping pack on K but reading
 * along the "wrong" vector axis (M instead of K), which the compiler
 * folds into `ds_read offset:128` sequences that bypass XOR. Here we
 * keep vectorization on K throughout.
 *
 * Expected: same SQ_LDS_BANK_CONFLICT (0) and same SQ_INSTS_LDS (128)
 * as 07. The removed `transpose_tile2d` compiled to nothing anyway.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct XorCrossWarpWindowReinterpretKernel
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
    // SINGLE [M, K] XOR LDS descriptor. Matches the production pattern in
    // gemm_universal_pipeline_ag_bg_cr_policy.hpp, including the
    // architecture-aware `NBanks` and `RowMul`. This is important on
    // MI355 (gfx950) because it has 64 LDS banks, not 32 like MI300
    // (gfx942). If you hardcode 32 you get an undersized XOR scramble
    // and residual conflicts on MI355.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        constexpr auto DataTypeSize = sizeof(DataType);
        constexpr index_t NBanks    = get_n_lds_banks(); // 32 on gfx942, 64 on gfx950
        constexpr index_t LdsBandwidthBytes = NBanks * get_n_dwords_per_128b();
        constexpr index_t MLdsLayerRaw      = LdsBandwidthBytes / kK / DataTypeSize;
        constexpr index_t MLdsLayer         = MLdsLayerRaw < 1 ? 1 : MLdsLayerRaw;
        constexpr index_t RowMul            = (NBanks == 64) ? 2 : 1;

        // Step 1: (B, A, C) natural row-major order.
        //   B = M / MLdsLayer       (outermost, stride kK*MLdsLayer bytes-worth of elements)
        //   A = K / KPack * Layer   (middle, stride KPack)
        //   C = KPack               (innermost, stride 1 = vector axis)
        constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kM / MLdsLayer>{},           // B
                       number<kK / kKPack * MLdsLayer>{},  // A
                       number<kKPack>{}),                  // C
            make_tuple(number<kK * MLdsLayer>{},           // stride B
                       number<kKPack>{},                   // stride A
                       number<1>{}),                       // stride C
            number<kKPack>{},
            number<1>{});

        // Step 2: XOR on (B, A). xor_t gives A_low = A XOR (B % A).
        // On gfx950 (64 banks) RowMul=2 doubles the B-length the XOR considers,
        // which matches the wider physical bank row.
        constexpr auto lds_desc_permuted = transform_tensor_descriptor(
            lds_desc_0,
            make_tuple(make_xor_transform(
                           make_tuple(number<kM / MLdsLayer * RowMul>{},   // B_eff
                                      number<kK / kKPack * MLdsLayer>{})), // A
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0, 1>{}, sequence<2>{}));

        // Step 3: Unmerge A -> (Layer, K/Pack). Upper: (B=0, L=1, K0=2, C=3).
        constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
            lds_desc_permuted,
            make_tuple(
                make_pass_through_transform(number<kM / MLdsLayer>{}),
                make_unmerge_transform(make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
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

    // CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    // {
    // return make_naive_tensor_descriptor_packed(
    //     make_tuple(number<kM>{}, number<kK>{}));
    // }

    // --------------------------------------------------------------------
    // WRITE distribution (M-partitioned by warp).
    //
    // Warp w covers M=[w*16, w*16+16), all K=[0, 32).
    // Lane L within the warp covers (M2_idx, K0_idx) = (L/4, L%4)
    // -- i.e. 16 distinct M rows x 4 K-groups within the warp.
    // Per-thread Y shape: (M0=1, K1=8) => 8 K values at a fixed M.
    //
    // This is the standard row-major copy distribution for [M, K].
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeWriteDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType); // 8 for FP16
        constexpr index_t K0 = kK / K1;               // 4
        constexpr index_t M2 = 64 / K0;               // 16
        constexpr index_t M1 = kBlockSize / 64;       // 4 (warps)
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
    // READ distribution (K-partitioned by warp).
    //
    // Warp w covers K=[w*8, w*8+8), all M=[0, 64).
    // Lane L within the warp covers (M1_idx, M2_idx) = (L/8, L%8)
    // -- i.e. 64 distinct M rows (all of them) at a fixed K-group.
    // Per-thread Y shape: (M0=1, K1=8) => 8 K values at a fixed M.
    //
    // Key property: this reads DIFFERENT data than the write distribution
    // put into each thread. Specifically, warp 0 now reads K=[0,8) for
    // M=0..63 -- those M's were written by all four write-warps. Data
    // genuinely flows through LDS between warps.
    //
    // The read is logically "transposed" from the write in the sense that
    // warp-partitioning swaps from M-axis to K-axis. If this access were
    // done against a plain row-major LDS (no XOR), 64 lanes reading at
    // stride kK*sizeof(T) = 64 bytes would collide on 16 banks each,
    // producing ~32 bank conflicts per read. With the XOR descriptor the
    // 64 addresses are spread across all 32 banks.
    //
    // Note on the shape structure: we use M = (M0=1, M1=8, M2=8) here
    // instead of (1, 4, 16) so that M1*M2 = 64 lanes partition all of M
    // within a single warp (P1 -> (M1, M2)). The total M=64 is unchanged.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeReadDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType); // 8
        constexpr index_t K0 = kK / K1;               // 4 (== number of warps)
        constexpr index_t M0 = 1;
        constexpr index_t M1 = 8;
        constexpr index_t M2 = 8;
        static_assert(M0 * M1 * M2 == kM, "M partition must cover all 64 M rows");
        static_assert(K0 == kBlockSize / 64, "K0 must equal #warps so warp w gets K0_idx=w");

        // Hs = (M-group {1, 8, 8}, K-group {4, 8})
        // P0 -> Hs[1].minor[0] = K0 = 4   (warp id -> K-group id)
        // P1 -> (Hs[0].minor[1]=M1=8, Hs[0].minor[2]=M2=8) = 64 lanes
        // Y0 -> Hs[0].minor[0] = M0 = 1
        // Y1 -> Hs[1].minor[1] = K1 = 8
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<2>, sequence<1, 1>>,
                                       tuple<sequence<0>, sequence<1, 2>>,
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
        [[maybe_unused]] auto amir = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);

        constexpr auto dist_mk_write = MakeWriteDistributionMK();
        constexpr auto dist_mk_read  = MakeReadDistributionMK();
        // Note: no dist_km_shuffled here. The output window carries the
        // "transpose" via its descriptor instead of via a register relabel.

        auto lds_window_write =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk_write);
        auto lds_window_read =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk_read);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // [1] Global [M, K] load, K-vectorized (M-partitioned across warps).
            //     Pass the guaranteed last-dim vector length/stride so the
            //     compiler can fuse the 8 × b16 loads into one b128.
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<kKPack>{},  // 8 elements contiguous on last dim
                number<1>{});      // stride 1 on last dim
            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);
            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk_write);

            auto reg_write = load_tile(gmem_window_in);

            // [2] Store to LDS [M, K] XOR with the M-partitioned write
            //     distribution (K-vectorized, matches XOR pack axis).
            store_tile(lds_window_write, reg_write);
            block_sync_lds();

            // [3] Read from LDS [M, K] XOR with the K-partitioned read
            //     distribution. Cross-warp exchange happens here. Still
            //     K-vectorized, still no bank conflicts.
            auto reg_read = load_tile(lds_window_read);
            block_sync_lds();

            // [4] NO transpose_tile2d. The output global buffer is
            //     [K, M] row-major in memory (strides (M, 1) for (K, M)).
            //     We build a window that describes it as an [M, K]-
            //     shaped view with strides (1, M): element at logical
            //     (m, k) in this view resolves to offset 1*m + M*k,
            //     which is exactly position of output[k, m] in the
            //     [K, M] buffer.
            //
            //     With this descriptor, reg_read's (M, K) distribution
            //     maps directly to the correct global addresses -- no
            //     register relabel needed.
            const auto gmem_desc_out_reinterp = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}), // [M, K] logical shape
                make_tuple(number<1>{}, M));            // strides (1, M_global) -- swapped
            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out_reinterp);
            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kM, kK), {0, 0}, dist_mk_read);

            // [5] Store directly from reg_read. Same 8 buffer_store_short
            //     per thread as 07, same warp coalescing, same timing.
            store_tile(gmem_window_out, reg_read);
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.08: XOR + cross-warp LDS transpose, NO transpose_tile2d\n";
    std::cout << "========================================\n\n";

    std::cout << "Single [M, K] XOR descriptor.\n";
    std::cout << "WRITE distribution: M-partitioned (warp w gets M=[w*16, w*16+16))\n";
    std::cout << "READ  distribution: K-partitioned (warp w gets K=[w*8,  w*8+8))\n";
    std::cout << "Both K-vectorized (per-thread (1, 8)) so XOR stays in effect.\n";
    std::cout << "LDS actually moves data between warps this time.\n\n";

    constexpr index_t M = 65536;
    constexpr index_t K = 256;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);

    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            h_input[m * K + k] = static_cast<DataType>(m * 10 + k);

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
    constexpr index_t lds_size = XorCrossWarpWindowReinterpretKernel<DataType>::GetStaticLdsSize();

    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          XorCrossWarpWindowReinterpretKernel<DataType>{},
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
                          XorCrossWarpWindowReinterpretKernel<DataType>{},
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

    bool passed         = true;
    index_t error_count = 0;

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

    std::cout << "\nExpected: ZERO bank conflicts despite cross-warp LDS traffic.\n";
    std::cout << "Verify with:\n";
    std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -- ./aa_tutorial_14_08_xor_cross_warp_window_reinterpret\n";

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
