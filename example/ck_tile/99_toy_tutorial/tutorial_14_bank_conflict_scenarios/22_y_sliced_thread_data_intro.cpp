// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.22: get_y_sliced_thread_data / set_y_sliced_thread_data
 *
 * What it does
 * ------------
 * A static_distributed_tensor stores its per-lane data in a thread_buffer<T, N>.
 * The Y-coordinate system has one axis per (Repeat | Lane | Vector) factor of
 * the X->Y mapping. For the toy distribution shared with 14.20 / 14.21:
 *
 *   Y = (R_M=2, V_M=1, R_N=1, V_N=4)  =>  prod = 8 slots / lane
 *
 * Slots are laid out PACKED in lexicographic Y order:
 *
 *   off(rm, vm, rn, vn) = ((rm*V_M + vm)*R_N + rn)*V_N + vn
 *                       =  rm*4 + vn         // because V_M=1, R_N=1
 *
 * For lane 0 the 8 slots correspond to these (m, n) cells of the full tile:
 *
 *   slot 0 : (rm=0, vn=0) -> (m=0, n=0)
 *   slot 1 : (rm=0, vn=1) -> (m=0, n=1)
 *   slot 2 : (rm=0, vn=2) -> (m=0, n=2)
 *   slot 3 : (rm=0, vn=3) -> (m=0, n=3)
 *   slot 4 : (rm=1, vn=0) -> (m=4, n=0)
 *   slot 5 : (rm=1, vn=1) -> (m=4, n=1)
 *   slot 6 : (rm=1, vn=2) -> (m=4, n=2)
 *   slot 7 : (rm=1, vn=3) -> (m=4, n=3)
 *
 * The API
 * -------
 *   tensor.get_y_sliced_thread_data(YOrigin, YLengths)
 *     -> thread_buffer<T, prod(YLengths)>
 *
 *   for each idx in static_ford<YLengths>:
 *       dst_off = packed_desc(YLengths).offset(idx)
 *       src_off = thread_buffer_desc.offset(idx + YOrigin)
 *       dst[dst_off] = thread_buf_[src_off]
 *
 *   set_y_sliced_thread_data(YOrigin, YLengths, slice) is the symmetric
 *   writeback (same loop with the assignment reversed).
 *
 * Why?
 * ----
 * Real pipelines use it to operate on just one repeat or one row of the
 * per-thread tile without touching the rest -- e.g. softmax denominators
 * (one M row at a time), partial accumulator updates, or hand-rolled fusion
 * with another in-register helper.
 *
 * What this tutorial shows
 * ------------------------
 * Each lane fills its 8 slots with `100*m + n` so the values are easy to
 * recognize. The demo is split into 5 small kernels (each fits under
 * build-debug -O0 -fno-inline). Lane 0 of each kernel prints:
 *
 *   [A] sec_full_kernel    : full thread_buf_ (all 8 slots).
 *   [B] sec_B_kernel       : slice (origin=(0,0,0,0), len=(1,1,1,4))
 *                            -> M-row 0 of this lane (slots 0..3).
 *   [C] sec_C_kernel       : slice (origin=(1,0,0,0), len=(1,1,1,4))
 *                            -> M-row 4 of this lane (slots 4..7).
 *   [D] sec_D_kernel       : slice (origin=(0,0,0,0), len=(2,1,1,2))
 *                            -> first 2 N elements of BOTH M rows.
 *                            Y points (rm,vm,rn,vn): (0,0,0,0),(0,0,0,1),
 *                            (1,0,0,0),(1,0,0,1) -> per-thread slots 0,1,4,5.
 *                            Slice IS a contiguous Y rectangle; it is NOT
 *                            contiguous in the flat thread_buf_.
 *   [E] sec_E_write_kernel : mutate the [B] slice to (-1,-2,-3,-4) and call
 *                            set_y_sliced_thread_data; observe that only
 *                            slots 0..3 changed.
 *
 * Build target:  aa_tutorial_14_22_y_sliced_thread_data_intro
 */

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>

using namespace ck_tile;

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

using Dist  = decltype(make_tiny_distribution());
using DTens = static_distributed_tensor<float, Dist>;

// Direct-poke fill so the kernels don't pay for a sweep_tile expansion.
CK_TILE_DEVICE void fill_known(DTens& t)
{
    auto& b = t.get_thread_buffer();
    b(number<0>{}) = 0.0f;     // (m=0, n=0)
    b(number<1>{}) = 1.0f;     // (m=0, n=1)
    b(number<2>{}) = 2.0f;     // (m=0, n=2)
    b(number<3>{}) = 3.0f;     // (m=0, n=3)
    b(number<4>{}) = 400.0f;   // (m=4, n=0)
    b(number<5>{}) = 401.0f;   // (m=4, n=1)
    b(number<6>{}) = 402.0f;   // (m=4, n=2)
    b(number<7>{}) = 403.0f;   // (m=4, n=3)
}

// --------------------------------------------------------------------------
// [A] full thread_buf_
// --------------------------------------------------------------------------
__global__ void sec_full_kernel()
{
    DTens t;
    fill_known(t);
    if(threadIdx.x != 0) return;

    const auto& b = t.get_thread_buffer();
    printf("=== Tutorial 14.22 ===\n");
    printf("    Per-lane Y = (R_M=2, V_M=1, R_N=1, V_N=4) -> 8 slots / lane.\n");
    printf("    Lane 0 owns (m,n) cells (0,0..3) in slots 0..3, (4,0..3) in slots 4..7.\n");
    printf("    Filled with 100*m + n so each value identifies its (m, n).\n\n");
    printf("[A] full thread_buf_:\n");
    printf("    thread_buf_[0..7] = %g %g %g %g | %g %g %g %g\n",
           static_cast<double>(b[number<0>{}]),
           static_cast<double>(b[number<1>{}]),
           static_cast<double>(b[number<2>{}]),
           static_cast<double>(b[number<3>{}]),
           static_cast<double>(b[number<4>{}]),
           static_cast<double>(b[number<5>{}]),
           static_cast<double>(b[number<6>{}]),
           static_cast<double>(b[number<7>{}]));
}

// --------------------------------------------------------------------------
// [B] slice = M-row 0 of this lane (slots 0..3)
// --------------------------------------------------------------------------
__global__ void sec_B_kernel()
{
    DTens t;
    fill_known(t);

    constexpr auto orig = sequence<0, 0, 0, 0>{};
    constexpr auto len  = sequence<1, 1, 1, 4>{};
    auto slice = t.get_y_sliced_thread_data(orig, len);   // thread_buffer<float, 4>

    if(threadIdx.x != 0) return;
    printf("\n[B] slice (origin=(0,0,0,0), len=(1,1,1,4)) -> M-row 0, slots 0..3\n");
    printf("    expects 4 floats: 0 1 2 3\n");
    printf("    slice = %g %g %g %g\n",
           static_cast<double>(slice[number<0>{}]),
           static_cast<double>(slice[number<1>{}]),
           static_cast<double>(slice[number<2>{}]),
           static_cast<double>(slice[number<3>{}]));
}

// --------------------------------------------------------------------------
// [C] slice = M-row 4 of this lane (slots 4..7)
// --------------------------------------------------------------------------
__global__ void sec_C_kernel()
{
    DTens t;
    fill_known(t);

    constexpr auto orig = sequence<1, 0, 0, 0>{};
    constexpr auto len  = sequence<1, 1, 1, 4>{};
    auto slice = t.get_y_sliced_thread_data(orig, len);

    if(threadIdx.x != 0) return;
    printf("\n[C] slice (origin=(1,0,0,0), len=(1,1,1,4)) -> M-row 4, slots 4..7\n");
    printf("    expects 4 floats: 400 401 402 403\n");
    printf("    slice = %g %g %g %g\n",
           static_cast<double>(slice[number<0>{}]),
           static_cast<double>(slice[number<1>{}]),
           static_cast<double>(slice[number<2>{}]),
           static_cast<double>(slice[number<3>{}]));
}

// --------------------------------------------------------------------------
// [D] slice = first 2 N elements of BOTH M rows.
//     Y points (0,0,0,0),(0,0,0,1),(1,0,0,0),(1,0,0,1) -> slots 0,1,4,5.
//     The slice IS a contiguous Y rectangle; it is NOT contiguous in the
//     flat thread_buf_. The output buffer holds it in packed Y order.
// --------------------------------------------------------------------------
__global__ void sec_D_kernel()
{
    DTens t;
    fill_known(t);

    constexpr auto orig = sequence<0, 0, 0, 0>{};
    constexpr auto len  = sequence<2, 1, 1, 2>{};
    auto slice = t.get_y_sliced_thread_data(orig, len);

    if(threadIdx.x != 0) return;
    printf("\n[D] slice (origin=(0,0,0,0), len=(2,1,1,2)) -> first 2 N of BOTH M rows\n");
    printf("    Y points (rm,vm,rn,vn): (0,0,0,0)(0,0,0,1)(1,0,0,0)(1,0,0,1)\n");
    printf("    pulled from slots:        0          1         4         5\n");
    printf("    expects 4 floats: 0 1 400 401\n");
    printf("    slice = %g %g %g %g\n",
           static_cast<double>(slice[number<0>{}]),
           static_cast<double>(slice[number<1>{}]),
           static_cast<double>(slice[number<2>{}]),
           static_cast<double>(slice[number<3>{}]));
}

// --------------------------------------------------------------------------
// [E] mutate the [B] slice and write it back via set_y_sliced_thread_data.
// --------------------------------------------------------------------------
__global__ void sec_E_write_kernel()
{
    DTens t;
    fill_known(t);

    constexpr auto orig = sequence<0, 0, 0, 0>{};
    constexpr auto len  = sequence<1, 1, 1, 4>{};
    auto slice = t.get_y_sliced_thread_data(orig, len);
    slice(number<0>{}) = -1.0f;
    slice(number<1>{}) = -2.0f;
    slice(number<2>{}) = -3.0f;
    slice(number<3>{}) = -4.0f;
    t.set_y_sliced_thread_data(orig, len, slice);

    if(threadIdx.x != 0) return;
    const auto& b = t.get_thread_buffer();
    printf("\n[E] mutate slice_B to (-1,-2,-3,-4) and set_y_sliced_thread_data back\n");
    printf("    expected thread_buf_: -1 -2 -3 -4 | 400 401 402 403\n");
    printf("    actual   thread_buf_: %g %g %g %g | %g %g %g %g\n",
           static_cast<double>(b[number<0>{}]),
           static_cast<double>(b[number<1>{}]),
           static_cast<double>(b[number<2>{}]),
           static_cast<double>(b[number<3>{}]),
           static_cast<double>(b[number<4>{}]),
           static_cast<double>(b[number<5>{}]),
           static_cast<double>(b[number<6>{}]),
           static_cast<double>(b[number<7>{}]));
    printf("    (slots 4..7 are unchanged because the slice did not cover them)\n");
}

static int launch_and_sync(const char* tag)
{
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize after %s failed: %s\n",
                tag, hipGetErrorString(err));
        return 1;
    }
    return 0;
}

int main()
{
    hipLaunchKernelGGL(sec_full_kernel,    dim3(1), dim3(64), 0, nullptr);
    if(launch_and_sync("sec_full"))    return 1;
    hipLaunchKernelGGL(sec_B_kernel,       dim3(1), dim3(64), 0, nullptr);
    if(launch_and_sync("sec_B"))       return 1;
    hipLaunchKernelGGL(sec_C_kernel,       dim3(1), dim3(64), 0, nullptr);
    if(launch_and_sync("sec_C"))       return 1;
    hipLaunchKernelGGL(sec_D_kernel,       dim3(1), dim3(64), 0, nullptr);
    if(launch_and_sync("sec_D"))       return 1;
    hipLaunchKernelGGL(sec_E_write_kernel, dim3(1), dim3(64), 0, nullptr);
    if(launch_and_sync("sec_E_write")) return 1;
    return 0;
}
