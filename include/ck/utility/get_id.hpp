// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/get_shift.hpp"

namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
    // warpSize is defined by HIP
    return warpSize;
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

// get_wave_id() does the same thing as get_warp_local_1d_id(), except that
// it tries to save the result in sgpr
#if defined(__gfx90a__)
__device__ inline index_t get_wave_id()
{
    int thread_id = threadIdx.x;
    int wave_id;
    constexpr int shift = get_shift<warpSize>();

    // clang-format off
    __asm__ volatile("v_readfirstlane_b32 s16, %1      \n\
                      s_lshr_b32 %0, s16, %2"
                      : "=s"(wave_id)
                      : "v"(thread_id), "i"(shift)
                      : "s16");
    // clang-format on

    return wave_id;
};
#else
__device__ inline index_t get_wave_id() { return get_warp_local_1d_id(); };
#endif

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

} // namespace ck
