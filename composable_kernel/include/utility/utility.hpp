#ifndef CK_UTILITY_HPP
#define CK_UTILITY_HPP

#include "config.hpp"

namespace ck {

__device__ constexpr index_t get_wave_size() { return CK_GPU_WAVE_SIZE; }

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_wave_local_1d_id() { return threadIdx.x / get_wave_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

} // namespace ck

#endif
