// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.21: load_tile, the smallest possible example
 *
 * Goal: exactly one focused kernel that
 *   1. defines a static_tile_distribution,
 *   2. wraps a global float buffer in a tile_window, and
 *   3. calls load_tile to fill a per-thread register file.
 *
 * Designed to be small enough to compile and launch under the build-debug
 * tree (-O0 -fno-inline -mcmodel=large -ggdb3) so you can attach a debugger
 * and step through the load + inspect the per-lane register slots that come
 * back. There are no extra sweeps, no reductions, no element-wise helpers,
 * no `__shared__` staging arrays -- just the load and a tiny, lane-gated
 * per-slot print so the structure is obvious.
 *
 * Source layout: a flat [M=8, N=64] row-major float buffer with
 *   src[m, n] = m*64 + n          // values 0..511 in memory order
 * Each lane gets 8 of those 512 floats. The mapping (which slot <- which
 * tensor coord) is fixed by the distribution; with the toy distribution
 * defined below it works out to:
 *
 *   lane_M = lane_id / 16,  lane_N = lane_id % 16
 *
 *   slot k=0..3:  m = lane_M,      n = 4*lane_N + k       (R_M=0 half)
 *   slot k=4..7:  m = lane_M + 4,  n = 4*lane_N + (k-4)   (R_M=1 half)
 *
 *   value(m, n)   = m*64 + n
 *   => slots[0..3] = 64*lane_M       + 4*lane_N + (0..3)
 *      slots[4..7] = 64*lane_M       + 4*lane_N + (0..3) + 256
 *
 * So inside one thread:
 *   slots[0..3] are 4 contiguous floats (== one buffer_load_dwordx4)
 *   slots[4..7] are 4 contiguous floats 256 elements later (one more dwordx4)
 *
 * Build target:  aa_tutorial_14_21_load_tile_intro
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// --------------------------------------------------------------------------
// Distribution: same shape as 14.12 / 14.20 so you can read them side by side.
//   1 warp = 64 lanes (Lane_M=4, Lane_N=16)
//   per-thread Y = (Repeat_M=2, Vector_M=1, Repeat_N=1, Vector_N=4)
//                = 8 floats per thread => 8 VGPRs per thread for this tile
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
            tuple<sequence<1, 2>, sequence<1, 2>>,
            tuple<sequence<1, 1>, sequence<2, 2>>,
            sequence<1, 1, 2, 2>,
            sequence<0, 3, 0, 3>>{});
}

static constexpr index_t kTileM = 8;
static constexpr index_t kTileN = 64;
// Per-thread Y dimensions (R_M*V_M, R_N*V_N) = (2*1, 1*4) = 8 floats per thread.

// --------------------------------------------------------------------------
// The kernel: one block, one warp (64 lanes). Each lane:
//   * has a tile_window over the whole [M, N] global buffer
//   * calls load_tile  -> fills its 8 register slots
//   * (lane 0 only) prints expected vs. actual for itself
//
// Set a breakpoint on the line marked "<-- BREAK HERE" and inspect the
// `loaded.thread_buf_` array; under -O0 the 8 floats live in named VGPRs
// associated with that local variable.
// --------------------------------------------------------------------------
__global__ void load_tile_kernel(const float* __restrict__ src)
{
    // 1. Wrap raw pointer as a 2-D logical view: [M=8, N=64], row-major.
    //    The two trailing arguments tell the framework which dim is the
    //    inner-vector dim and how many elements to vectorize over.
    auto src_view = make_naive_tensor_view<address_space_enum::global>(
        src,
        make_tuple(kTileM, kTileN),
        make_tuple(kTileN, 1),
        number<1>{},   // inner vector dim = N
        number<4>{});  // V_N = 4 -> dwordx4 friendly

    // 2. Anchor a tile_window of [kTileM, kTileN] at (0, 0) using our
    //    distribution. The window owns the per-lane offsets needed by load_tile.
    constexpr auto dist = make_tiny_distribution();
    auto src_window     = make_tile_window(
        src_view,
        make_tuple(number<kTileM>{}, number<kTileN>{}),
        multi_index<2>{0, 0},
        dist);

    // 3. Issue the load. Result is a static_distributed_tensor<float, dist>;
    //    its underlying storage is a thread_buffer<float, 8> per lane.
    auto loaded = load_tile(src_window);   // <-- BREAK HERE: 8 floats land in this lane's VGPRs

    // 4. Lane 0 prints its slots and the predicted values, so a quick run
    //    confirms the layout matches the comment block at the top of the file.
    if(threadIdx.x == 0)
    {
        const int lane_M = 0;   // lid / 16
        const int lane_N = 0;   // lid % 16
        const float exp0 = static_cast<float>(64 * lane_M + 4 * lane_N + 0);
        const float exp4 = exp0 + 256.0f;

        printf("=== Tutorial 14.21: load_tile (lane 0 only) ===\n");
        printf("  src[m,n] = m*64 + n   (values 0..511 in memory)\n");
        printf("  lane 0 -> (lane_M, lane_N) = (%d, %d)\n", lane_M, lane_N);
        printf("  expected slots[0..3] = %g %g %g %g\n",
               static_cast<double>(exp0 + 0),
               static_cast<double>(exp0 + 1),
               static_cast<double>(exp0 + 2),
               static_cast<double>(exp0 + 3));
        printf("  expected slots[4..7] = %g %g %g %g\n",
               static_cast<double>(exp4 + 0),
               static_cast<double>(exp4 + 1),
               static_cast<double>(exp4 + 2),
               static_cast<double>(exp4 + 3));
        printf("  actual   slots[0..7] = %g %g %g %g  %g %g %g %g\n",
               static_cast<double>(loaded.get_thread_buffer()[number<0>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<1>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<2>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<3>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<4>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<5>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<6>{}]),
               static_cast<double>(loaded.get_thread_buffer()[number<7>{}]));
    }
}

int main()
{
    // Build host buffer: src[m, n] = m*64 + n   (values 0..511).
    std::vector<float> h_src(kTileM * kTileN);
    for(index_t m = 0; m < kTileM; ++m)
        for(index_t n = 0; n < kTileN; ++n)
            h_src[m * kTileN + n] = static_cast<float>(m * kTileN + n);

    DeviceMem d_src(kTileM * kTileN * sizeof(float));
    d_src.ToDevice(h_src.data(), kTileM * kTileN * sizeof(float));

    // 1 block * 64 threads = 1 warp -- minimal launch.
    hipLaunchKernelGGL(load_tile_kernel, dim3(1), dim3(64), 0, nullptr,
                       static_cast<const float*>(d_src.GetDeviceBuffer()));

    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
