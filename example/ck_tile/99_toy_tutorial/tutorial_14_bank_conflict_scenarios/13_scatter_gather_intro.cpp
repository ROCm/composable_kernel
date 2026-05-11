// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.13: Minimal tile_scatter_gather inspector
 *
 * Companion to 14.12. 14.12 showed sweep_tile over a regular
 * static_distributed_tensor that had been produced by a normal
 * tile_window + load_tile. This one swaps the *window* for
 * tile_scatter_gather, which overrides the thread's natural bottom-
 * tensor row coordinate with a runtime-indexed "page offset" along one
 * gather H-dim. It is the core primitive behind:
 *
 *   - paged attention K/V loads (page_idx = physical page base per seq step)
 *   - MoE input gather (page_idx = sorted_token_id * stride_M)
 *   - scatter writeback (same mechanism with store_tile instead of load_tile)
 *
 * Mental model:
 *   normal tile_window                 tile_scatter_gather
 *   ------------------                 -------------------
 *   bottom_coord = origin + x_coord    bottom_coord = origin + x_coord (M dim zeroed)
 *   (x_coord derived from p, y via                        + page_idx[y_gather_slot]
 *    the distribution adaptor)
 *
 * Concretely the window drops whatever row coordinate the distribution
 * would have computed along HsGatherDim, and substitutes page_idx[ys_gather_slot]
 * as an *element* offset from the window origin. Lanes / warps still
 * partition the non-gather axes normally.
 *
 * API surface covered:
 *   (1) statically_indexed_array<index_t, R_M>                 -- page_idx type
 *   (2) make_tile_scatter_gather(view, lengths, origin,
 *                                dist, page_idx)               -- factory
 *   (3) load_tile(gather_window)                               -- triggers per-slot gather
 *   (*) HsGatherDim / YsGatherDims                             -- defaults: H[0], Y[0]
 *
 * Setup:
 *   Physical [M=8, K=64] int32 buffer with row i filled with 1000*i + col.
 *   page_idx = [5*K, 2*K, 7*K, 0*K]  (pick physical rows 5, 2, 7, 0 in that order)
 *   Distribution keeps all 4 M rows in Y (R_M=4), partitions K across 64 lanes.
 *   => Lane L sees 4 scalars = [5000+L, 2000+L, 7000+L, 0+L].
 *
 * Compile-cost note (same trick as 14.12):
 *   Stage the per-access result into a plain int array inside any lambda
 *   and do printf in a runtime for-loop outside. Don't call
 *   get_x_indices_from_distributed_indices in the lambda.
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

// --------------------------------------------------------------------------
// Tile distribution: puts the M axis entirely in Y (R_M = 4), K entirely in
// lane id (T_K = 64). The gather dim is H[0] = M. Y-slot 0 is the M-Y
// (which is what YsGatherDims=sequence<0> picks up by default).
//
//   M axis (H[0]): (R_M=4, W_M=1, T_M=1, V_M=1)   -> M-tile = 4
//   K axis (H[1]): (R_K=1, W_K=1, T_K=64, V_K=1)  -> K-tile = 64
//   lanes = T_M * T_K = 1 * 64 = 64
//   per-thread Y = (R_M, V_K) = (4, 1) = 4 scalars
// --------------------------------------------------------------------------
CK_TILE_HOST_DEVICE constexpr auto make_gather_distribution()
{
    constexpr index_t R_M = 4, W_M = 1, T_M = 1,  V_M = 1;
    constexpr index_t R_K = 1, W_K = 1, T_K = 64, V_K = 1;
    static_assert(T_M * T_K == 64, "wave64");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<R_M, W_M, T_M, V_M>,
                  sequence<R_K, W_K, T_K, V_K>>,
            tuple<sequence<1, 2>, sequence<1, 2>>, // Ps major (warp / lane)
            tuple<sequence<1, 1>, sequence<2, 2>>, // Ps minor (W_*, T_*)
            sequence<1, 2>,                        // Ys major: Y0 -> H[0]=M, Y1 -> H[1]=K
            sequence<0, 3>>{});                    // Ys minor: Y0=R_M, Y1=V_K
}

using Dist = decltype(make_gather_distribution());

// 4 per-thread scalars from the default sweep (span0 * span1 = 4*1).
static constexpr index_t kPerThread = 4;

// Gather parameters known at compile time so we can keep the kernel
// signature simple. Real pipelines pass these via kargs.
static constexpr index_t kM = 8;
static constexpr index_t kK = 64;
static constexpr index_t kTileM = 4;

// ------------------------------ kernel ------------------------------------

__global__ void scatter_gather_intro_kernel(const int* __restrict__ src)
{
    const bool dbg = (threadIdx.x == 0 && blockIdx.x == 0);
    const bool dbg_lane1 = (threadIdx.x == 1 && blockIdx.x == 0);

    if(dbg) printf("=== Tutorial 14.13: tile_scatter_gather inspector ===\n\n");

    // -----------------------------------------------------------------
    // (0) plain tensor view over the [M, K] global buffer, row-major
    //     strides (K, 1). No window yet.
    // -----------------------------------------------------------------
    const auto src_view = make_naive_tensor_view<address_space_enum::global>(
        src,
        make_tuple(kM, kK),
        make_tuple(kK, 1),
        number<1>{},  // inner vector dim on K
        number<1>{}); // vector length (demo doesn't need to vectorize)

    // -----------------------------------------------------------------
    // (1) page_idx: one entry per R_M Y-slot. Each entry is an ELEMENT
    //     offset (not bytes) along the gather H-dim (M). For a row-major
    //     [M, K] descriptor the per-M-row stride is K, so
    //     page_idx[i] = physical_row_i * K.
    //
    //     Note: the size MUST match R_M (the Y length along the gather
    //     dim). Here the per-thread M iteration is entirely in R_M, so
    //     every lane sees the same 4 slots of page_idx. In a real
    //     pipeline with T_M != 1 each lane would compute its own slice
    //     of the full page_idx list.
    // -----------------------------------------------------------------
    statically_indexed_array<index_t, kTileM> page_idx;
    page_idx(number<0>{}) = 5 * kK; // tile row 0 <- physical row 5
    page_idx(number<1>{}) = 2 * kK; // tile row 1 <- physical row 2
    page_idx(number<2>{}) = 7 * kK; // tile row 2 <- physical row 7
    page_idx(number<3>{}) = 0 * kK; // tile row 3 <- physical row 0

    if(dbg)
    {
        printf("(1) page_idx (element offsets along M, row-major stride = %d):\n", kK);
        printf("    slot 0 = %d   (-> row 5)\n", static_cast<int>(page_idx[number<0>{}]));
        printf("    slot 1 = %d   (-> row 2)\n", static_cast<int>(page_idx[number<1>{}]));
        printf("    slot 2 = %d   (-> row 7)\n", static_cast<int>(page_idx[number<2>{}]));
        printf("    slot 3 = %d   (-> row 0)\n\n", static_cast<int>(page_idx[number<3>{}]));
    }

    // -----------------------------------------------------------------
    // (2) Build the scatter-gather window. Window origin is (0, 0); the
    //     window's natural [kTileM x kK] rows are going to be IGNORED on
    //     the gather dim and replaced by origin + page_idx[y].
    //
    //     Defaults: HsGatherDim=0 (first H = M), YsGatherDims=sequence<0>
    //     (first Y slot = R_M).
    // -----------------------------------------------------------------
    constexpr auto dist = make_gather_distribution();
    auto gather_window  = make_tile_scatter_gather(
        src_view,
        make_tuple(number<kTileM>{}, number<kK>{}),
        multi_index<2>{0, 0}, // origin at (0, 0)
        dist,
        page_idx);

    // -----------------------------------------------------------------
    // (3) load_tile. Each access step i of the internal SFC reads
    //     src[origin_m + page_idx[i], lane_k_for_this_thread].
    // -----------------------------------------------------------------
    const auto y = load_tile(gather_window);

    // -----------------------------------------------------------------
    // (4) Stage the per-thread buffer to plain ints, then printf outside
    //     any compile-time expansion (same pattern as 14.12).
    // -----------------------------------------------------------------
    int staged[kPerThread] = {};
    static_for<0, kPerThread, 1>{}([&](auto i) {
        staged[i] = y.get_thread_buffer()[number<i>{}];
    });

    if(dbg)
    {
        printf("(2) lane 0 thread_buf_ after load_tile(gather_window):\n");
        printf("    (expected [5000, 2000, 7000, 0] -- row i column 0 = 1000*i + 0)\n");
        for(int i = 0; i < kPerThread; ++i)
            printf("    slot[%d] = %d\n", i, staged[i]);
        printf("\n");
    }

    if(dbg_lane1)
    {
        printf("(3) lane 1 thread_buf_ (same rows, column 1):\n");
        printf("    (expected [5001, 2001, 7001, 1])\n");
        for(int i = 0; i < kPerThread; ++i)
            printf("    slot[%d] = %d\n", i, staged[i]);
        printf("\n");
    }

    // -----------------------------------------------------------------
    // (5) Sanity sweep_tile: confirm that at Y-slot i the value is
    //     exactly what page_idx says it should be. Staging lets us keep
    //     printf outside the lambda so we don't bloat compile time.
    // -----------------------------------------------------------------
    int sweep_vals[kPerThread] = {};
    int seq = 0;
    sweep_tile(y, [&](auto idx) { sweep_vals[seq++] = y[idx]; });

    if(dbg)
    {
        printf("(4) sweep_tile order matches Y-slot order:\n");
        for(int i = 0; i < kPerThread; ++i)
            printf("    sweep[%d] = %d\n", i, sweep_vals[i]);
    }
}

int main()
{
    printf("=== Tutorial 14.13: launching 1 warp (64 lanes) over a gathered tile ===\n");
    printf("Physical buffer [M=%d, K=%d], row i col j = 1000*i + j\n", kM, kK);
    printf("Gather rows [5, 2, 7, 0] via tile_scatter_gather + page_idx\n\n");

    std::vector<int> h_src(kM * kK);
    for(index_t m = 0; m < kM; ++m)
        for(index_t k = 0; k < kK; ++k)
            h_src[m * kK + k] = 1000 * m + k;

    DeviceMem d_src(kM * kK * sizeof(int));
    d_src.ToDevice(h_src.data(), kM * kK * sizeof(int));

    hipLaunchKernelGGL(scatter_gather_intro_kernel, dim3(1), dim3(64), 0, nullptr,
                       static_cast<const int*>(d_src.GetDeviceBuffer()));
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
