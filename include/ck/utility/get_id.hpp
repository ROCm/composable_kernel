#pragma once
#include "config.hpp"

#ifndef CK_NOGPU
namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
    // warpSize is defined by HIP
    return warpSize;
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

} // namespace ck
#endif