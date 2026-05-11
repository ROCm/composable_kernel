// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.12: Minimal sweep_tile inspector
 *
 * Companion to 14.11. Where 14.11 poked at a StaticTileDistribution through
 * the raw space_filling_curve, this one uses the higher-level "sweep" API
 * that every real pipeline uses (rmsnorm2d / softmax / fmha / etc.).
 *
 * API surface covered:
 *   (1) StaticTileDistribution::get_distributed_spans()
 *   (2) sweep_tile_span(span, f)                   -- single-span loop
 *   (3) sweep_tile(tensor, f)                      -- full tile, Unpacks=(1,1)
 *                                                     (this demo also maps to
 *                                                     the (M, N) tile coord via
 *                                                     get_x_indices_from_distributed_indices)
 *   (4) sweep_tile(tensor, f, sequence<1, 2>{})    -- unpack 2 along X1 (pairs on N)
 *   (5) tile_sweeper<Tens, F, Unpacks>             -- get_num_of_access() +
 *                                                     operator()(number<i>{}) dispatch
 *   (6) sweep_tile(tensor, f, sequence<2, 1>{})    -- unpack 2 along X0 (pairs on M).
 *                                                     Mirror of (4) on the other axis;
 *                                                     useful for comparing which axis
 *                                                     the pair stride lands on.
 *   (*) Using the lambda's distributed_index to read/write
 *         static_distributed_tensor (y(idx), y[idx])
 *
 * Compile-cost note (same trick as 14.11)
 * ---------------------------------------
 * Each `sweep_tile(...)` instantiates a whole nested template recursion,
 * and every call-site of `printf(...)` / `get_x_indices_from_distributed_indices(...)`
 * inside a sweep lambda multiplies that cost. The fast pattern:
 *
 *   1. Inside the sweep lambda, store the interesting values into plain
 *      runtime arrays (ints / floats). No printf. No get_x_indices...
 *      unless you really want the tile coord -- then call it once and
 *      store the result, don't do it per-demo.
 *   2. After the sweep returns, run a normal runtime `for` loop and
 *      printf the arrays. printf then lives outside all template
 *      expansions, so it's instantiated once per format string.
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// --------------------------------------------------------------------------
// Tiny 2-D tile distribution: NDimX = 2, NDimY = 4
//
//   M axis (X0): (Repeat_M, Warp_M, Lane_M, Vector_M) = (2, 1, 4, 1)
//                -> M-tile = 8,   per-thread Y on M = (R_M, V_M) = (2, 1)
//   N axis (X1): (Repeat_N, Warp_N, Lane_N, Vector_N) = (1, 1, 16, 4)
//                -> N-tile = 64,  per-thread Y on N = (R_N, V_N) = (1, 4)
//
//   lanes = Lane_M * Lane_N = 4 * 16 = 64   (AMD wave64)
//   per-thread Y block lengths = (R_M, V_M, R_N, V_N) = (2, 1, 1, 4)
//                              = 2 * 1 * 1 * 4 = 8 scalars per thread
//
//   get_distributed_spans():
//     span0 (M) ::Impl = sequence<R_M, V_M> = sequence<2, 1>  -> 2 distr. idxs
//     span1 (N) ::Impl = sequence<R_N, V_N> = sequence<1, 4>  -> 4 distr. idxs
//   -> full sweep = 2 * 4 = 8 accesses per thread.
// --------------------------------------------------------------------------
CK_TILE_HOST_DEVICE constexpr auto make_tiny_distribution()
{
    constexpr index_t R_M = 2, W_M = 1, T_M = 4,  V_M = 1;
    constexpr index_t R_N = 1, W_N = 1, T_N = 16, V_N = 4;
    static_assert(T_M * T_N == 64, "wave64");
    static_assert(W_M * W_N == 1, "1-warp toy block");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<R_M, W_M, T_M, V_M>,
                  sequence<R_N, W_N, T_N, V_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>, // Ps major (warp / lane)
            tuple<sequence<1, 1>, sequence<2, 2>>, // Ps minor
            sequence<1, 1, 2, 2>,                  // Ys major
            sequence<0, 3, 0, 3>>{});              // Ys minor
}

using Dist  = decltype(make_tiny_distribution());
using DTens = static_distributed_tensor<float, Dist>;

// Known total number of accesses for the default sweep (span0 * span1).
// Used to size the staging arrays.
static constexpr index_t kFullNumAccess = 8; // 2 (span0) * 4 (span1)

// Per-access row staged by the sweep lambdas. `tile_distributed_index<a,b>`
// is size-2 here since span.Impl = (Repeat, Vector), so we store its 2
// components per X dim. y holds the distributed-tensor value; xm/xn are
// the tile-space (M, N) coord (set only in demo (3), left 0 elsewhere).
struct AccessRow
{
    int di0a, di0b;  // X0 (M-side) distributed index
    int di1a, di1b;  // X1 (N-side) distributed index
    int xm, xn;      // tile-space coord; 0 if not computed
    float y;
};

// ------------------------------ kernel ------------------------------------

__global__ void sweep_intro_kernel()
{
    const bool dbg = (threadIdx.x == 0 && blockIdx.x == 0);

    if(dbg) printf("=== Tutorial 14.12: sweep_tile inspector ===\n\n");

    // -----------------------------------------------------------------
    // (1) distributed spans
    // -----------------------------------------------------------------
    constexpr auto spans = DTens::get_distributed_spans();
    using Impl0          = typename decltype(spans[number<0>{}])::Impl; // sequence<2,1>
    using Impl1          = typename decltype(spans[number<1>{}])::Impl; // sequence<1,4>

    if(dbg)
    {
        printf("(1) get_distributed_spans()\n");
        printf("    span0.Impl = sequence<%d,%d>   (Y dims from X0 = M)\n",
               static_cast<int>(Impl0{}[number<0>{}].value),
               static_cast<int>(Impl0{}[number<1>{}].value));
        printf("    span1.Impl = sequence<%d,%d>   (Y dims from X1 = N)\n",
               static_cast<int>(Impl1{}[number<0>{}].value),
               static_cast<int>(Impl1{}[number<1>{}].value));
        printf("    per-thread scalars = %d   (== span0 * span1)\n\n",
               static_cast<int>(DTens::get_thread_buffer_size()));
    }

    // -----------------------------------------------------------------
    // (2) sweep_tile_span: walk one span at a time. Stage into 2 ints
    //     per call, print out the runtime loop.
    // -----------------------------------------------------------------
    constexpr int kSpan0Size = 2;
    constexpr int kSpan1Size = 4;
    int s0_a[kSpan0Size] = {}, s0_b[kSpan0Size] = {};
    int s1_a[kSpan1Size] = {}, s1_b[kSpan1Size] = {};

    {
        int i = 0;
        sweep_tile_span(spans[number<0>{}], [&](auto d0) {
            using I = typename decltype(d0)::Impl;
            s0_a[i] = static_cast<int>(I{}[number<0>{}].value);
            s0_b[i] = static_cast<int>(I{}[number<1>{}].value);
            ++i;
        });
    }
    {
        int i = 0;
        sweep_tile_span(spans[number<1>{}], [&](auto d1) {
            using I = typename decltype(d1)::Impl;
            s1_a[i] = static_cast<int>(I{}[number<0>{}].value);
            s1_b[i] = static_cast<int>(I{}[number<1>{}].value);
            ++i;
        });
    }

    if(dbg)
    {
        printf("(2) sweep_tile_span(span0, f)\n");
        for(int i = 0; i < kSpan0Size; ++i)
            printf("    d0=di<%d,%d>\n", s0_a[i], s0_b[i]);
        printf("    sweep_tile_span(span1, f)\n");
        for(int i = 0; i < kSpan1Size; ++i)
            printf("    d1=di<%d,%d>\n", s1_a[i], s1_b[i]);
        printf("\n");
    }

    // Distributed tensor to write/read through a sweep. sweep (3) below
    // writes every element before any later sweep reads.
    DTens y;

    // -----------------------------------------------------------------
    // (3) sweep_tile(tensor, f)   -- defaults to Unpacks = sequence<1,1>
    //     We stage each access into an AccessRow[] and print afterwards.
    //     This demo is the one that also maps to the tile (M,N) coord.
    // -----------------------------------------------------------------
    AccessRow rows3[kFullNumAccess] = {};
    {
        int seq = 0;
        sweep_tile(y, [&](auto idx) {
            y(idx) = 100.f + static_cast<float>(seq);

            using I0 = typename decltype(idx[number<0>{}])::Impl;
            using I1 = typename decltype(idx[number<1>{}])::Impl;

            const auto x = get_x_indices_from_distributed_indices(Dist{}, idx);

            rows3[seq] = AccessRow{
                static_cast<int>(I0{}[number<0>{}].value),
                static_cast<int>(I0{}[number<1>{}].value),
                static_cast<int>(I1{}[number<0>{}].value),
                static_cast<int>(I1{}[number<1>{}].value),
                static_cast<int>(x[number<0>{}]),
                static_cast<int>(x[number<1>{}]),
                y[idx],
            };
            ++seq;
        });
    }

    if(dbg)
    {
        printf("(3) sweep_tile(y, [](auto idx){...})   // Unpacks = (1,1)\n");
        for(int i = 0; i < kFullNumAccess; ++i)
        {
            const auto& r = rows3[i];
            printf("    seq=%d  idx=(di<%d,%d>, di<%d,%d>)  x=(%d,%d)  y=%g\n",
                   i, r.di0a, r.di0b, r.di1a, r.di1b, r.xm, r.xn,
                   static_cast<double>(r.y));
        }
        printf("\n");
    }

    // -----------------------------------------------------------------
    // (4) sweep_tile(tensor, f, sequence<1, 2>): unpack 2 on X1 (N).
    //     Each call takes 2 idxs (same M, adjacent N). We stage pairs.
    // -----------------------------------------------------------------
    static constexpr int kNumPair12 = kFullNumAccess / 2; // 4 groups
    AccessRow rows4a[kNumPair12] = {};
    AccessRow rows4b[kNumPair12] = {};
    {
        int gp = 0;
        sweep_tile(
            y,
            [&](auto idx_a, auto idx_b) {
                using I0a = typename decltype(idx_a[number<0>{}])::Impl;
                using I1a = typename decltype(idx_a[number<1>{}])::Impl;
                using I0b = typename decltype(idx_b[number<0>{}])::Impl;
                using I1b = typename decltype(idx_b[number<1>{}])::Impl;

                rows4a[gp] = AccessRow{
                    static_cast<int>(I0a{}[number<0>{}].value),
                    static_cast<int>(I0a{}[number<1>{}].value),
                    static_cast<int>(I1a{}[number<0>{}].value),
                    static_cast<int>(I1a{}[number<1>{}].value),
                    0, 0, y[idx_a],
                };
                rows4b[gp] = AccessRow{
                    static_cast<int>(I0b{}[number<0>{}].value),
                    static_cast<int>(I0b{}[number<1>{}].value),
                    static_cast<int>(I1b{}[number<0>{}].value),
                    static_cast<int>(I1b{}[number<1>{}].value),
                    0, 0, y[idx_b],
                };
                ++gp;
            },
            sequence<1, 2>{});
    }

    if(dbg)
    {
        printf("(4) sweep_tile(y, f, sequence<1,2>)    // pairs on X1\n");
        for(int i = 0; i < kNumPair12; ++i)
        {
            printf("    grp=%d\n", i);
            printf("      a: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   rows4a[i].di0a, rows4a[i].di0b,
                   rows4a[i].di1a, rows4a[i].di1b,
                   static_cast<double>(rows4a[i].y));
            printf("      b: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   rows4b[i].di0a, rows4b[i].di0b,
                   rows4b[i].di1a, rows4b[i].di1b,
                   static_cast<double>(rows4b[i].y));
        }
        printf("\n");
    }

    // -----------------------------------------------------------------
    // (5) tile_sweeper<DT, F, Unpacks>: functor form with
    //     get_num_of_access() and per-access dispatch via sw(number<i>{}).
    //     Stage into rows5_each[]; also stage a second full sweep from
    //     sw() into rows5_full[].
    // -----------------------------------------------------------------
    AccessRow rows5_each[kFullNumAccess] = {};
    int        rows5_each_order[kFullNumAccess] = {};
    AccessRow rows5_full[kFullNumAccess] = {};
    int        rows5_full_count = 0;

    using Unpacks = sequence<1, 1>;
    auto sweep_body_each = [&](auto idx) {
        // This path is reached via sw(number<i>{}); we don't know i
        // inside the body here, so use a running counter.
        const int seq = rows5_full_count; // repurpose as a counter into rows5_each
        using I0 = typename decltype(idx[number<0>{}])::Impl;
        using I1 = typename decltype(idx[number<1>{}])::Impl;
        rows5_each[seq] = AccessRow{
            static_cast<int>(I0{}[number<0>{}].value),
            static_cast<int>(I0{}[number<1>{}].value),
            static_cast<int>(I1{}[number<0>{}].value),
            static_cast<int>(I1{}[number<1>{}].value),
            0, 0, y[idx],
        };
        rows5_each_order[seq] = seq;
        ++rows5_full_count;
    };
    tile_sweeper<DTens, decltype(sweep_body_each), Unpacks> sw(sweep_body_each);

    const int num_acc = static_cast<int>(decltype(sw)::get_num_of_access());

    // Per-access dispatch.
    static_for<0, decltype(sw)::get_num_of_access(), 1>{}([&](auto i) {
        sw(i);
    });

    // Snapshot of the per-access phase (rows5_each has all 8 entries).
    AccessRow rows5_each_snap[kFullNumAccess];
    for(int i = 0; i < kFullNumAccess; ++i) rows5_each_snap[i] = rows5_each[i];

    // Full-sweep form sw() -- reuse the body but write to rows5_full.
    // We swap which array the body writes into by using a separate
    // counter + lambda.
    rows5_full_count = 0;
    auto sweep_body_full = [&](auto idx) {
        using I0 = typename decltype(idx[number<0>{}])::Impl;
        using I1 = typename decltype(idx[number<1>{}])::Impl;
        rows5_full[rows5_full_count] = AccessRow{
            static_cast<int>(I0{}[number<0>{}].value),
            static_cast<int>(I0{}[number<1>{}].value),
            static_cast<int>(I1{}[number<0>{}].value),
            static_cast<int>(I1{}[number<1>{}].value),
            0, 0, y[idx],
        };
        ++rows5_full_count;
    };
    tile_sweeper<DTens, decltype(sweep_body_full), Unpacks> sw_full(sweep_body_full);
    sw_full(); // full-sweep form

    if(dbg)
    {
        printf("(5) tile_sweeper<DT, F, Unpacks>\n");
        printf("    tile_sweeper::get_num_of_access() = %d\n", num_acc);
        for(int i = 0; i < num_acc; ++i)
        {
            const auto& r = rows5_each_snap[i];
            printf("    sw(number<%d>{})  body: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   i, r.di0a, r.di0b, r.di1a, r.di1b,
                   static_cast<double>(r.y));
        }
        printf("    sw()  // equivalent to sweep_tile(y, body):\n");
        for(int i = 0; i < rows5_full_count; ++i)
        {
            const auto& r = rows5_full[i];
            printf("      body: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   r.di0a, r.di0b, r.di1a, r.di1b,
                   static_cast<double>(r.y));
        }
        printf("\n");
    }

    // -----------------------------------------------------------------
    // (6) sweep_tile(tensor, f, sequence<2, 1>): mirror of (4) -- now the
    //     unpack-2 is on X0 (M) instead of X1 (N). Each call receives 2
    //     idxs that share the X1 (N) components and differ in X0 (M).
    //
    //     Compare with (4): span0.Impl = sequence<2,1> (R_M=2, V_M=1), so
    //     unpacking 2 on X0 pairs the two Repeat_M steps. span1.Impl =
    //     sequence<1,4> with unpack 1 means we still walk N one-at-a-time,
    //     so the group count is the same (kFullNumAccess/2 = 4 groups).
    // -----------------------------------------------------------------
    static constexpr int kNumPair21 = kFullNumAccess / 2; // 4 groups
    AccessRow rows6a[kNumPair21] = {};
    AccessRow rows6b[kNumPair21] = {};
    {
        int gp = 0;
        sweep_tile(
            y,
            [&](auto idx_a, auto idx_b) {
                using I0a = typename decltype(idx_a[number<0>{}])::Impl;
                using I1a = typename decltype(idx_a[number<1>{}])::Impl;
                using I0b = typename decltype(idx_b[number<0>{}])::Impl;
                using I1b = typename decltype(idx_b[number<1>{}])::Impl;

                rows6a[gp] = AccessRow{
                    static_cast<int>(I0a{}[number<0>{}].value),
                    static_cast<int>(I0a{}[number<1>{}].value),
                    static_cast<int>(I1a{}[number<0>{}].value),
                    static_cast<int>(I1a{}[number<1>{}].value),
                    0, 0, y[idx_a],
                };
                rows6b[gp] = AccessRow{
                    static_cast<int>(I0b{}[number<0>{}].value),
                    static_cast<int>(I0b{}[number<1>{}].value),
                    static_cast<int>(I1b{}[number<0>{}].value),
                    static_cast<int>(I1b{}[number<1>{}].value),
                    0, 0, y[idx_b],
                };
                ++gp;
            },
            sequence<2, 1>{});
    }

    if(dbg)
    {
        printf("(6) sweep_tile(y, f, sequence<2,1>)    // pairs on X0 (M)\n");
        for(int i = 0; i < kNumPair21; ++i)
        {
            printf("    grp=%d\n", i);
            printf("      a: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   rows6a[i].di0a, rows6a[i].di0b,
                   rows6a[i].di1a, rows6a[i].di1b,
                   static_cast<double>(rows6a[i].y));
            printf("      b: (di<%d,%d>, di<%d,%d>)  y=%g\n",
                   rows6b[i].di0a, rows6b[i].di0b,
                   rows6b[i].di1a, rows6b[i].di1b,
                   static_cast<double>(rows6b[i].y));
        }
    }
}

int main()
{
    printf("=== Tutorial 14.12: launching 1 warp (64 lanes) ===\n");
    printf("Only lane 0 prints; the other lanes run the sweep too so\n");
    printf("partition indices (lane id) are well defined for\n");
    printf("get_x_indices_from_distributed_indices(...)\n\n");

    hipLaunchKernelGGL(sweep_intro_kernel, dim3(1), dim3(64), 0, nullptr);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
