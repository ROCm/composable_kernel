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
static __device__ inline index_t get_wave_id()
{
    int thread_id = threadIdx.x;
    int tmp_int;
    int wave_id;
    constexpr index_t shift = get_shift<warpSize>();

    // clang-format off
    __asm__ volatile("v_lshrrev_b32 %1, %3, %2                       \n\
                      v_readfirstlane_b32 %0, %1"
                      : "=s"(wave_id), "=v"(tmp_int)
                      : "v"(thread_id), "i"(shift));
    // clang-format on

    return wave_id;
};
#else
static __device__ inline index_t get_wave_id() { return get_warp_local_1d_id(); };
#endif

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

} // namespace ck
