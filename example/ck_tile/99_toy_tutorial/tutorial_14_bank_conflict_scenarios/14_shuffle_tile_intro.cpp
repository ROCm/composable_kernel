// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.14: Minimal shuffle_tile inspector
 *
 * Companion to 14.12/14.13. shuffle_tile(out, in) is an **in-register
 * transpose** between two tile distributions that agree on everything
 * except the Y ordering (Ys_to_Rhs_major / Ys_to_Rhs_minor). No LDS, no
 * cross-lane traffic, no memory traffic at all -- just reshuffle of
 * each thread's register file so a consumer expecting a different Y
 * layout can pick up the same logical scalars.
 *
 * Canonical use case: FMHA loads V from DRAM with "vector on N" per
 * thread, then hands it to mfma which wants "vector on M". shuffle_tile
 * does that transpose inside the register file via transpose_vectors.
 *
 * Compile-time contract (see shuffle_tile.hpp):
 *   In and Out encodings must share
 *     - rs_lengths_          (no replication here -- empty sequence)
 *     - hs_lengthss_         (per-X H layouts)
 *     - ps_to_rhss_major_    (P-dim routing)
 *     - ps_to_rhss_minor_
 *     - NDimY                (same number of Y dims)
 *   ...and differ only in ys_to_rhs_major_ / ys_to_rhs_minor_ (Y order).
 *
 * Setup (per-thread 4x4 transpose, single SFC access):
 *   Hs[0] = M: (R_M=1, W_M=1, T_M=1,  V_M=4)  -> M-tile = 4
 *   Hs[1] = N: (R_N=1, W_N=1, T_N=16, V_N=4)  -> N-tile = 64
 *   lanes = T_M * T_N = 1 * 16 = 16 ... we launch 16 lanes (1 warp-ish).
 *
 *   In  encoding: Y-order (VM, VN)   (last Y = VN, thread_buf_ contiguous in VN)
 *   Out encoding: Y-order (VN, VM)   (last Y = VM, thread_buf_ contiguous in VM)
 *
 * Each thread fills IN slot (vm*4 + vn) with the two-digit decimal
 * value 10*vm + vn. After shuffle_tile, OUT slot (vn*4 + vm) should
 * hold the SAME semantic value 10*vm + vn -- i.e. the printed OUT
 * thread_buf_ is the transpose of the printed IN thread_buf_.
 *
 * Expected thread_buf_ dumps from lane 0:
 *   IN :  0  1  2  3  10 11 12 13  20 21 22 23  30 31 32 33
 *   OUT:  0 10 20 30   1 11 21 31   2 12 22 32   3 13 23 33
 *
 * Compile-cost note (same trick as 14.12 / 14.13):
 *   Stage each buffer to a plain int[16] and printf in a runtime
 *   for-loop outside any compile-time expansion.
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>

using namespace ck_tile;

static constexpr index_t R_M = 1, W_M = 1, T_M = 1,  V_M = 4;
static constexpr index_t R_N = 1, W_N = 1, T_N = 16, V_N = 4;

// --------------------------------------------------------------------------
// Two distributions: same R/H/P, Y-order swapped.
//
// (Ys_to_Rhs_major, Ys_to_Rhs_minor) picks which H-axis slot each Y dim
// maps to. Encoding of H = (R, W, T, V):
//   rh_minor 0 = R, 1 = W, 2 = T, 3 = V.
// Input  Y = (VM, VN) -> major (1, 2),  minor (3, 3)
// Output Y = (VN, VM) -> major (2, 1),  minor (3, 3)
// --------------------------------------------------------------------------
CK_TILE_HOST_DEVICE constexpr auto make_in_dist()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<R_M, W_M, T_M, V_M>,
                  sequence<R_N, W_N, T_N, V_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>,
            tuple<sequence<1, 1>, sequence<2, 2>>,
            sequence<1, 2>,  // Y = (VM, VN)
            sequence<3, 3>>{});
}

CK_TILE_HOST_DEVICE constexpr auto make_out_dist()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<R_M, W_M, T_M, V_M>,
                  sequence<R_N, W_N, T_N, V_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>,
            tuple<sequence<1, 1>, sequence<2, 2>>,
            sequence<2, 1>,  // Y = (VN, VM)  <-- only change
            sequence<3, 3>>{});
}

using InDist  = decltype(make_in_dist());
using OutDist = decltype(make_out_dist());
using InTens  = static_distributed_tensor<int, InDist>;
using OutTens = static_distributed_tensor<int, OutDist>;

static constexpr index_t kPerThread = V_M * V_N; // 4 * 4 = 16

// ------------------------------ kernel ------------------------------------

__global__ void shuffle_intro_kernel()
{
    const bool dbg = (threadIdx.x == 0 && blockIdx.x == 0);

    if(dbg) printf("=== Tutorial 14.14: shuffle_tile inspector ===\n\n");

    // -----------------------------------------------------------------
    // (1) Create the IN tensor and fill raw slot (vm*VN + vn) with
    //     value (10*vm + vn). Every thread fills the same values; the
    //     demo is about the register-order permutation, not per-lane
    //     content.
    //
    //     IN thread_buf_ layout (Y = (VM, VN), packed row-major):
    //       slot s = vm * V_N + vn
    // -----------------------------------------------------------------
    InTens in;

    static_for<0, V_M, 1>{}([&](auto vm) {
        static_for<0, V_N, 1>{}([&](auto vn) {
            constexpr auto slot = number<vm.value * V_N + vn.value>{};
            in.get_thread_buffer()(slot) = 10 * vm.value + vn.value;
        });
    });

    // -----------------------------------------------------------------
    // (2) shuffle_tile: physical register reshuffle so the same logical
    //     (vm, vn) scalar now sits at OUT's Y-layout position. Internally
    //     this uses transpose_vectors on a VM x VN block per thread.
    // -----------------------------------------------------------------
    OutTens out;
    shuffle_tile(out, in);

    // -----------------------------------------------------------------
    // (3) Stage raw buffers and printf outside compile-time expansion.
    // -----------------------------------------------------------------
    int in_raw [kPerThread] = {};
    int out_raw[kPerThread] = {};
    static_for<0, kPerThread, 1>{}([&](auto i) {
        in_raw[i]  = in.get_thread_buffer()[i];
        out_raw[i] = out.get_thread_buffer()[i];
    });

    if(dbg)
    {
        printf("(1) IN thread_buf_ (Y = (VM, VN), slot = vm*%d + vn):\n", V_N);
        for(int vm = 0; vm < V_M; ++vm)
        {
            printf("    vm=%d :", vm);
            for(int vn = 0; vn < V_N; ++vn)
                printf(" %3d", in_raw[vm * V_N + vn]);
            printf("\n");
        }
        printf("\n");

        printf("(2) OUT thread_buf_ (Y = (VN, VM), slot = vn*%d + vm):\n", V_M);
        printf("    (should be IN transposed: out[vn*%d + vm] == in[vm*%d + vn])\n",
               V_M, V_N);
        for(int vn = 0; vn < V_N; ++vn)
        {
            printf("    vn=%d :", vn);
            for(int vm = 0; vm < V_M; ++vm)
                printf(" %3d", out_raw[vn * V_M + vm]);
            printf("\n");
        }
        printf("\n");

        // Runtime check: same scalar at semantically-equal positions.
        int mismatches = 0;
        for(int vm = 0; vm < V_M; ++vm)
            for(int vn = 0; vn < V_N; ++vn)
                if(in_raw[vm * V_N + vn] != out_raw[vn * V_M + vm]) ++mismatches;

        printf("(3) semantic check: %d mismatches (expect 0)\n", mismatches);
    }
}

int main()
{
    printf("=== Tutorial 14.14: launching 16 lanes (T_M*T_N=%d) ===\n", T_M * T_N);
    printf("Only lane 0 prints; shuffle_tile is a per-thread register move,\n");
    printf("so other lanes execute the same permutation on their own data.\n\n");

    hipLaunchKernelGGL(shuffle_intro_kernel, dim3(1), dim3(T_M * T_N), 0, nullptr);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
