// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.11: Minimal SFC + tile-distribution inspector
 *
 * Goal
 * ----
 * Play with ck_tile::space_filling_curve (SFC) as it is used in
 * transpose_tile.hpp, with the smallest plausible setup:
 *
 *   - A 2D Y tile_distribution_encoding (NDimY = 2), the shape
 *     transpose_tile2d is allowed to operate on.
 *   - Pull y_lengths out of StaticTileDistribution::ys_to_d_descriptor.
 *   - Build SFC_Y with the same DimAccessOrder = 0..N-1 and a
 *     scalars_per_access policy we pick, then enumerate every
 *     SFC access "start" idx_y_start and its y_in_desc offset.
 *
 * Not included on purpose:
 *   - No HBM / LDS / block sync / global load / store.
 *   - No kernel dispatch over many blocks.
 *   - A single-thread HIP kernel that printfs from the GPU, so it
 *     looks like transpose_tile runtime but stays minimal.
 *
 * How to read the output
 * ----------------------
 * For each scalars_per_access we print:
 *     num_access
 *     iAccess  idx_y_start=(y0, y1)  in_offset (from y_in_desc)
 * Compare with the Python debugger (space_filling_curve_debug.py) at
 * the same y_lengths and scalars_per_access to confirm the indices
 * match. The in_offset column shows what calculate_offset does on
 * top of idx_y_start; it is what transpose_tile2d_impl_in_thread
 * uses at lines `y_in_desc.calculate_offset(idx_y_in)`.
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// ---------------------------------------------------------------------------
// Step 1: a clean, canonical (Repeat, WarpPerBlock, ThreadPerWarp, Vector)
// static tile distribution -- the shape used, for instance, by the
// add_rmsnorm2d_rdquant pipeline. This gives a 2D X tile (M x N) where each
// axis is hierarchically split as:
//
//   Hs[i] = (Repeat_i, WarpPerBlock_i, ThreadPerWarp_i, Vector_i)
//
// Ps and Ys are wired so that:
//   P0 (warp id)  -> Hs[0].minor[1] + Hs[1].minor[1]    (WarpPerBlock_M, WarpPerBlock_N)
//   P1 (lane id)  -> Hs[0].minor[2] + Hs[1].minor[2]    (ThreadPerWarp_M, ThreadPerWarp_N)
//   Y0, Y1        -> Hs[0].minor[0] + Hs[0].minor[3]    (Repeat_M, Vector_M)
//   Y2, Y3        -> Hs[1].minor[0] + Hs[1].minor[3]    (Repeat_N, Vector_N)
//
// So NDimY = 4 and y_lengths = (Repeat_M, Vector_M, Repeat_N, Vector_N).
//
// Numbers are tiny on purpose (a 1-warp, 4-lane toy "block") so every SFC
// access is easy to read in the printf dump.
// ---------------------------------------------------------------------------
CK_TILE_HOST_DEVICE constexpr auto make_tiny_rmsnorm_like_distribution()
{
    constexpr index_t Repeat_M        = 1;
    constexpr index_t WarpPerBlock_M  = 1;
    constexpr index_t ThreadPerWarp_M = 8;
    constexpr index_t Vector_M        = 2;

    constexpr index_t Repeat_N        = 2;
    constexpr index_t WarpPerBlock_N  = 1;
    constexpr index_t ThreadPerWarp_N = 8;
    constexpr index_t Vector_N        = 4;

    // AMD CDNA/GCN wave size is 64, so the lane partition must cover 64 lanes
    static_assert(ThreadPerWarp_M * ThreadPerWarp_N == 64, "toy 1-warp block = 64 lanes");
    static_assert(WarpPerBlock_M * WarpPerBlock_N == 1, "1 warp for the toy");

    // Per-tile shape on each axis: Repeat * WarpPerBlock * ThreadPerWarp * Vector
    //   M-tile = 1 * 1 * 8 * 2 = 16
    //   N-tile = 2 * 1 * 8 * 4 = 64
    // Per-thread Y block = (Repeat_M, Vector_M, Repeat_N, Vector_N) = (1, 2, 2, 4)
    //   -> 1 * 2 * 2 * 4 = 16 scalars per thread

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>, // empty R (no replication)
            tuple<sequence<Repeat_M, WarpPerBlock_M, ThreadPerWarp_M, Vector_M>,
                  sequence<Repeat_N, WarpPerBlock_N, ThreadPerWarp_N, Vector_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>,  // Ps major
            tuple<sequence<1, 1>, sequence<2, 2>>,  // Ps minor
            sequence<1, 1, 2, 2>,                   // Ys major
            sequence<0, 3, 0, 3>>{});               // Ys minor
}

// ---------------------------------------------------------------------------
// Step 2: same SFC_Y the transpose_tile.hpp impl builds.
//
//   y_lengths           = ys_to_d_descriptor().get_lengths()
//   DimAccessOrder      = 0..NDimY-1   (identity, same as transpose_tile)
//   scalars_per_access  = supplied by caller (template arg below)
//
// This mirrors:
//   using SFC_Y = space_filling_curve<decltype(y_lengths),
//                                     arithmetic_sequence_gen<0, NDimY, 1>::type,
//                                     scalars_per_access>;
// ---------------------------------------------------------------------------
template <typename YLengthsSeq, typename ScalarsPerAccessSeq>
using sfc_y_t = space_filling_curve<YLengthsSeq,
                                    typename arithmetic_sequence_gen<0, YLengthsSeq::size(), 1>::type,
                                    ScalarsPerAccessSeq>;

template <index_t... YL, index_t... SP>
__device__ void enumerate_sfc(sequence<YL...> y_lengths,
                              sequence<SP...> scalars_per_access,
                              const char*     label)
{
    using SFC = sfc_y_t<decltype(y_lengths), decltype(scalars_per_access)>;
    constexpr index_t num_access = SFC::get_num_of_access();
    constexpr index_t NDimY      = decltype(y_lengths)::size();

    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("[%s] y_lengths=(", label);
        ((printf("%d,", YL)), ...);
        printf(") scalars_per_access=(");
        ((printf("%d,", SP)), ...);
        printf(")  num_access=%d\n", num_access);
    }

    constexpr auto dist      = make_tiny_rmsnorm_like_distribution();
    constexpr auto y_in_desc = dist.get_ys_to_d_descriptor();

    static_for<0, num_access, 1>{}([&](auto iAccess) {
        constexpr auto idx_y_start = SFC::get_index(iAccess);

        constexpr auto idx_y_in = generate_tuple(
            [&](auto ii) { return idx_y_start[ii].value; }, number<NDimY>{});

        constexpr index_t in_offset = y_in_desc.calculate_offset(idx_y_in);

        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("  iAccess=%-3d idx_y_start=(", static_cast<int>(iAccess.value));
            static_for<0, NDimY, 1>{}([&](auto d) {
                printf("%d%s",
                       static_cast<int>(idx_y_start[d].value),
                       d.value + 1 == NDimY ? "" : ",");
            });
            printf(")  in_offset=%d\n", static_cast<int>(in_offset));
        }
    });

    // -----------------------------------------------------------------
    // get_forward_step(i) == get_step_between(i, i+1)   (Y multi-index
    // delta from iAccess i to iAccess i+1).  get_step_between(h, t)
    // is the absolute Y delta from accessing h to accessing t.
    //
    // These are what real CK Tile code uses to *move* a tile_window
    // between SFC steps (see `idx_diff_ys` in tile_window.hpp), rather
    // than recomputing absolute offsets via calculate_offset each time.
    //
    // CHEAP PRINT PATTERN: do all the compile-time SFC work inside a
    // static_for whose body only writes ints into a plain runtime array.
    // Then print with a normal runtime `for` loop -- no printf inside
    // static_for means no per-iteration printf template instantiations,
    // which is what previously exploded compile time.
    // -----------------------------------------------------------------
    constexpr index_t num_fwd = num_access > 0 ? num_access - 1 : 0;
    int fwd_steps[num_fwd > 0 ? num_fwd : 1][NDimY];

    static_for<0, num_fwd, 1>{}([&](auto i) {
        constexpr auto step = SFC::get_forward_step(i);
        static_for<0, NDimY, 1>{}([&](auto d) {
            fwd_steps[i.value][d.value] = static_cast<int>(step[d].value);
        });
    });

    int step_mid[NDimY] = {};
    int step_end[NDimY] = {};
    if constexpr(num_access >= 2)
    {
        constexpr auto s_mid =
            SFC::get_step_between(number<0>{}, number<(num_access / 2)>{});
        constexpr auto s_end =
            SFC::get_step_between(number<0>{}, number<(num_access - 1)>{});
        static_for<0, NDimY, 1>{}([&](auto d) {
            step_mid[d.value] = static_cast<int>(s_mid[d].value);
            step_end[d.value] = static_cast<int>(s_end[d].value);
        });
    }

    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("  -- forward_step[i -> i+1]  (Y multi-index delta):\n");
        for(int i = 0; i < num_fwd; ++i)
        {
            printf("    step[%d->%d] = (", i, i + 1);
            for(int d = 0; d < NDimY; ++d)
                printf("%d%s", fwd_steps[i][d], d + 1 == NDimY ? "" : ",");
            printf(")\n");
        }
        if constexpr(num_access >= 2)
        {
            printf("  -- non-adjacent get_step_between:\n");
            printf("    step[0->%d] = (", static_cast<int>(num_access / 2));
            for(int d = 0; d < NDimY; ++d)
                printf("%d%s", step_mid[d], d + 1 == NDimY ? "" : ",");
            printf(")\n");
            printf("    step[0->%d] = (", static_cast<int>(num_access - 1));
            for(int d = 0; d < NDimY; ++d)
                printf("%d%s", step_end[d], d + 1 == NDimY ? "" : ",");
            printf(")\n");
        }
    }
}

__global__ void sfc_intro_kernel()
{
    constexpr auto dist = make_tiny_rmsnorm_like_distribution();
    // ys_to_d_descriptor lengths: that's our y_lengths.
    constexpr auto y_lengths = to_sequence(dist.get_ys_to_d_descriptor().get_lengths());
    constexpr index_t NDimY  = decltype(y_lengths)::size();
    static_assert(NDimY == 4, "rmsnorm-like encoding has 4 Y dims (Repeat_M, Vector_M, Repeat_N, Vector_N)");

    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("y_lengths = (");
        static_for<0, NDimY, 1>{}([&](auto d) {
            printf("%d%s",
                   static_cast<int>(y_lengths[d].value),
                   d.value + 1 == NDimY ? "" : ",");
        });
        printf(")\n");
    }

    // --- A: finest grain (all ones) -> exhaustive snake walk over Y
    enumerate_sfc(y_lengths, sequence<1, 1, 1, 1>{}, "A: scalar-wise (1,1,1,1)");

    // --- B: one access covering the entire Y block
    enumerate_sfc(y_lengths, y_lengths, "B: full-tile one-access");

    // --- C: "contiguous Vector_N only" -- common real-world choice:
    // walk (Repeat_M, Vector_M, Repeat_N) scalar-wise, but each access
    // consumes the full Vector_N chunk along the last Y axis.
    enumerate_sfc(y_lengths,
                  sequence<1, 1, 1, y_lengths[number<3>{}].value>{},
                  "C: vector on last Y (Vector_N)");

    // --- D: vector on Vector_M as well; pulls together (1,Vector_M,1,Vector_N)
    enumerate_sfc(y_lengths,
                  sequence<1,
                           y_lengths[number<1>{}].value,
                           1,
                           y_lengths[number<3>{}].value>{},
                  "D: vectors on both Vector_M and Vector_N");
}

int main()
{
    printf("=== Tutorial 14.11: SFC + tile-distribution inspector ===\n");
    printf("Mirrors transpose_tile.hpp's SFC setup with a 2D-Y static\n");
    printf("tile distribution, then enumerates every access start.\n\n");

    hipLaunchKernelGGL(sfc_intro_kernel, dim3(1), dim3(1), 0, nullptr);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
