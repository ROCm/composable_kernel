// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <optional>
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/ck.hpp"

CK_DECLARE_ENV_VAR_BOOL(CK_AUTO_DEDUCE_SPLIT_K);

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceProperties
{
  DeviceProperties()
  {
    hipDeviceProp_t dev_prop;
    hipDevice_t dev;
    hip_check_error(hipGetDevice(&dev));
    hip_check_error(hipGetDeviceProperties(&dev_prop, dev));

    num_cu_ = dev_prop.multiProcessorCount;
    max_threads_per_block_ = dev_prop.maxThreadsPerBlock;
    max_shared_memory_per_block_ = dev_prop.sharedMemPerBlock;
  };
  int num_cu_;
  int max_threads_per_block_;
  int max_shared_memory_per_block_;
};

template <ck::index_t BlockSize>
std::optional<int> get_k_batch_value(const hipFunction_t& kernel, size_t grid_size, size_t dynSharedMemPerBlk = 0)
{
    static DeviceProperties properties;
    if(ck::EnvIsEnabled(CK_ENV(CK_AUTO_DEDUCE_SPLIT_K)))
    {
      const int num_cu = properties.num_cu_;
      int occupancy = 0;
      hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, dynSharedMemPerBlk));
      const int split_k = std::ceil((occupancy * num_cu) / (1.0 *grid_size));
      return split_k;
    }
    return std::nullopt;
}

template<ck::index_t MPerBlock,
         ck::index_t NPerBlock>
ck::index_t get_k_batch_value(ck::index_t split_k, ck::index_t M, ck::index_t N, ck::index_t conv_G)
{
    static DeviceProperties properties;
    if(ck::EnvIsEnabled(CK_ENV(CK_AUTO_DEDUCE_SPLIT_K)))
    {
      const int num_cu = properties.num_cu_;
      constexpr int occupancy = 1;
      const auto M0 = math::integer_divide_ceil(M, MPerBlock);
      const auto N0 = math::integer_divide_ceil(N, NPerBlock);
      const auto n_output_tiles = M0 * N0;
      const auto k_batch = std::ceil((occupancy * num_cu) / (1.0 * n_output_tiles * conv_G));
      if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
      {
          std::cout << "[SPLIT-K AUTODEDUCE] Overriding user deinfed split_k value " << split_k << " to optimal value " << k_batch << " for K-batch."<< std::endl;
      }
      return k_batch;
    }
    return split_k;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
