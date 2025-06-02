// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include <assert.h>

namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
    if (__builtin_is_constant_evaluated()) {
// this is only for cases where the use of constexpr cannot be avoided
#if defined(__GFX9__)
        return 64;
#else
        return 32;
#endif
    } else {
        hipDeviceProp_t props;
        [[maybe_unused]] hipError_t status = hipGetDeviceProperties(&props, get_device_id());

        assert(status == hipSuccess && "Failed to get device properties when trying to get warp size");
        return props.warpSize;
    }
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

} // namespace ck
