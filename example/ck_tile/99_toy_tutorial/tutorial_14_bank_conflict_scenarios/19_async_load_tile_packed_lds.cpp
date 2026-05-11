// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.19: async_load_tile variant of tutorial 08
 *
 * Same [M, K] -> [K, M] transpose as 08, but the DRAM->LDS half is
 * replaced with a single `async_load_tile` so the global read writes
 * straight into LDS without staging through VGPRs. On gfx950 this
 * compiles to one `buffer_load_dwordx4 ... offen offset:N lds` per
 * lane (verified in tutorial 18.B).
 *
 * Pipeline diff vs 08:
 *
 *   08:  load_tile(gmem)        // global -> VGPRs (vmcnt)
 *        store_tile(lds_xor,_)  // VGPRs  -> LDS XOR (lgkmcnt)
 *        block_sync_lds         // s_waitcnt lgkmcnt(0); s_barrier
 *        load_tile(lds_xor)     // LDS XOR -> VGPRs (cross-warp)
 *
 *   19:  async_load_tile(lds_async, gmem) // global -> LDS direct (vmcnt)
 *        async_load_fence(0)              // s_waitcnt vmcnt(0)
 *        s_barrier                        // workgroup barrier
 *        load_tile(lds_packed)            // LDS packed -> VGPRs (cross-warp)
 *
 * Why the LDS layout had to change
 * --------------------------------
 * In tutorial 08 the LDS layout was [M, K] with an XOR transform on
 * (B, A) to defeat the K-partitioned read-side bank conflicts. Two
 * subtleties of `async_load_tile` force us to drop the XOR here:
 *
 *   (a) The library variant we use (`async_load_tile`, the non-`_raw`
 *       form) requires the LDS-side window to be in the SAME
 *       X-coordinate space as the DRAM-side window. Both have to be
 *       2-D `[M, K]` for the per-warp address arithmetic
 *       `window_origin + warp_X_coord` in `async_load_with_offset` to
 *       type-check. (The `_raw` variant takes a 3-D
 *       [issues, warps, lanes] descriptor and uses m0-based
 *       addressing instead; that path is what the production gemm
 *       and FMHA pipelines use.)
 *
 *   (b) The hardware pattern of `buffer_load_dwordx4 ... lds` is
 *       fixed: lane L of a wave writes its 16-byte vector to
 *       `m0 + L * 16` bytes. The library sets `m0` once per wave to
 *       `lds_base + warp_X_coord`. So the LDS layout that ends up in
 *       memory is determined by the DRAM-side distribution, not by
 *       the LDS descriptor: whatever per-lane logical position the
 *       dist_mk_write distribution assigns to lane L is what lands
 *       at byte `warp_base + L*16` in LDS.
 *
 * For tutorial 08's `dist_mk_write` (M-partitioned by warp,
 * K-vectorized: per-thread (1, 8) fp16), lane L of warp w covers
 * logical (m = w*16 + L/4, K = (L%4)*8 + i) for i in [0, 8). The
 * packed [M, K] byte offset for that element is
 *   w*1024 + (L/4)*64 + (L%4)*16 + i*2
 * The hardware writes that lane at byte offset
 *   w*1024 + L*16 + i*2
 * These are equal because (L/4)*64 + (L%4)*16 == 16*L for any L in
 * [0, 64). So a packed 2-D [M, K] LDS view EXACTLY captures what the
 * async write deposits -- no XOR transform is permitted because the
 * hardware lays out lanes in linear order.
 *
 * Bank-conflict trade-off
 * -----------------------
 * Tutorial 08's selling point was that the [M, K] XOR descriptor
 * spread the cross-warp K-partitioned read across all 32 (or 64) LDS
 * banks. Without the XOR, this kernel re-introduces those bank
 * conflicts on the LDS read: 64 lanes of a warp each issue
 * `ds_read_b128` at stride kK*sizeof(fp16) = 64 bytes apart, which
 * collides on 16 banks (32 banks) or 32 banks (64 banks). Verify with
 *   rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
 *     -- ./aa_tutorial_14_19_async_load_tile_packed_lds
 * and compare against 08's zero-conflict numbers. The lesson is that
 * `async_load_tile` is a write-side optimisation and is orthogonal to
 * (and currently incompatible with) the [M, K] XOR descriptor pattern
 * used to defeat read-side conflicts -- you have to choose, or build a
 * 3-D-XOR LDS descriptor like the production gemm policy does.
 *
 * Sync change
 * -----------
 * The async path bumps `vmcnt` (the load is issued on the global
 * memory unit even though the destination is LDS), not `lgkmcnt`. So
 * the wait between the async load and the read is `s_waitcnt vmcnt(0)`
 * + `s_barrier`, which `async_load_fence(0)` and
 * `__builtin_amdgcn_s_barrier()` produce respectively. Note that
 * `block_sync_lds()` (used after the LDS read to gate the next
 * iteration's overwrite) still does the lgkmcnt path.
 *
 * Hardware: gfx950 (CDNA). `buffer_load_dwordx4 ... lds` is supported
 * on CDNA (gfx9 family); the `flat_load ... lds` variant is not.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct AsyncLoadTilePackedLdsKernel
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
    // Single packed [M, K] LDS descriptor used by BOTH the async write
    // (no per-thread distribution -- the dram-side distribution drives
    // each lane's deposit position via the hardware's
    // `m0 + lane_id * vector_size` rule) AND the read side (with the
    // K-partitioned cross-warp distribution).
    //
    // No XOR. The async-load hardware lays out lanes linearly in LDS,
    // and a packed [M, K] view of those bytes is exactly what the
    // dist_mk_write distribution produces -- see top-of-file proof.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        return make_naive_tensor_descriptor_packed(
            make_tuple(number<kM>{}, number<kK>{}));
    }

    // --------------------------------------------------------------------
    // Same write-side distribution as 08: M-partitioned by warp, K-
    // vectorized per-thread (M0=1, K1=8). The async load consumes this
    // distribution from the dram window.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeWriteDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kK / K1;
        constexpr index_t M2 = 64 / K0;
        constexpr index_t M1 = kBlockSize / 64;
        constexpr index_t M0 = kM / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // --------------------------------------------------------------------
    // Same read-side distribution as 08: K-partitioned by warp; lane L
    // within the warp owns M-row L (M = M1*M2 with M1=8, M2=8). Per-
    // thread Y shape (M0=1, K1=8) so each thread issues a single
    // `ds_read_b128` at a different m_t. With packed (no-XOR) LDS this
    // is exactly the conflict pattern from tutorial 04.
    // --------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeReadDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kK / K1;
        constexpr index_t M0 = 1;
        constexpr index_t M1 = 8;
        constexpr index_t M2 = 8;
        static_assert(M0 * M1 * M2 == kM, "M partition must cover all 64 M rows");
        static_assert(K0 == kBlockSize / 64, "K0 must equal #warps so warp w gets K0_idx=w");

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
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, lds_desc_mk);

        constexpr auto dist_mk_write = MakeWriteDistributionMK();
        constexpr auto dist_mk_read  = MakeReadDistributionMK();

        // 2-D LDS write window with no per-thread distribution. The
        // dram-side distribution drives the lane mapping; the
        // library's `async_load_with_offset` reduces it to one m0
        // value per warp (lds_origin + warp_X_coord), and the
        // hardware places lane L at m0 + L*16 bytes.
        auto lds_window_write =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0});

        // 2-D LDS read window with the K-partitioned cross-warp
        // distribution. Same pattern as 08, but now over a packed
        // (no-XOR) layout -- bank conflicts are expected here.
        auto lds_window_read =
            make_tile_window(lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk_read);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // [1] Global [M, K] view of the input slab. Same descriptor
            //     as 08: K-vectorized last dim so the async path can
            //     fuse the 8 fp16 per thread into one
            //     `buffer_load_dwordx4 ... offen ... lds`.
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}),
                number<kKPack>{},
                number<1>{});
            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            // `make_tile_window_raw` calls `init_raw()` on the underlying
            // buffer view, which builds the V# (4-SGPR buffer resource
            // descriptor) needed by `buffer_load_dwordx4 ... lds`.
            auto gmem_window_in = make_tile_window_raw(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk_write);

            // [2] Direct global -> LDS: one `buffer_load_dwordx4 ... lds`
            //     per lane. No VGPR staging, no `ds_write`.
            async_load_tile(lds_window_write, gmem_window_in);

            // [3] Drain `vmcnt` and barrier so all warps see the LDS
            //     content before the cross-warp read.
            async_load_fence(0);
            __builtin_amdgcn_s_barrier();

            // [4] LDS -> VGPRs (cross-warp K-partitioned). On packed LDS
            //     this incurs bank conflicts -- the trade-off for using
            //     the async write path.
            auto reg_read = load_tile(lds_window_read);
            block_sync_lds();

            // [5] Output store: same descriptor reinterpret trick as 08
            //     so `reg_read`'s (M, K) distribution maps directly to
            //     the correct (k, m) global addresses without a register
            //     relabel.
            const auto gmem_desc_out_reinterp = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(number<1>{}, M));
            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out_reinterp);
            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kM, kK), {0, 0}, dist_mk_read);

            store_tile(gmem_window_out, reg_read);
        }
    }
};

bool run_test(int num_iters = 1, int num_warmup = 0)
{
    std::cout << "\n========================================\n";
    std::cout << "Tutorial 14.19: async_load_tile + packed LDS (no XOR)\n";
    std::cout << "========================================\n\n";

    std::cout << "Same [M, K] -> [K, M] transpose as 08, but DRAM->LDS goes\n";
    std::cout << "through `async_load_tile` (one buffer_load_dwordx4 ... lds\n";
    std::cout << "per lane, no VGPR staging). LDS layout is packed [M, K]\n";
    std::cout << "(no XOR), so the cross-warp K-partitioned LDS read\n";
    std::cout << "incurs bank conflicts -- expected and explained in the\n";
    std::cout << "top-of-file comment.\n\n";

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
    constexpr index_t lds_size = AsyncLoadTilePackedLdsKernel<DataType>::GetStaticLdsSize();

    for(int i = 0; i < num_warmup; i++)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          AsyncLoadTilePackedLdsKernel<DataType>{},
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
                          AsyncLoadTilePackedLdsKernel<DataType>{},
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

    std::cout << "\nObservations to verify with rocprofv3:\n";
    std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
    std::cout << "    -- ./aa_tutorial_14_19_async_load_tile_packed_lds\n";
    std::cout << "  - SQ_LDS_BANK_CONFLICT > 0 expected (no XOR on the read).\n";
    std::cout << "  - SQ_INSTS_LDS pattern differs from 08: no `ds_write`\n";
    std::cout << "    for the LDS-write half (it's the global-issued LDS\n";
    std::cout << "    deposit from `buffer_load_dwordx4 ... lds`).\n";
    std::cout << "  Compare against tutorial 14.08 for the conflict-free\n";
    std::cout << "  baseline.\n";

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
