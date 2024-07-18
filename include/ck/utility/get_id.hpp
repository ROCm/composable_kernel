// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
    // warpSize is defined by HIP
    return warpSize;
}

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
__device__ index_t get_lane_id() { return __lane_id(); }

__device__ index_t get_warp_id()
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size());
}

__device__ index_t get_thread_id() { return threadIdx.x; }

__device__ index_t get_block_id() { return blockIdx.x; }

} // namespace ck
