// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/ck.hpp"

CK_DECLARE_ENV_VAR_UINT64(CK_SPLIT_K_BATCH_SIZE)

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
  };
  int num_cu_;
};

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size, ck::index_t K_size, ck::index_t conv_G /*, ck::index_t multiplier*/)
{
    static DeviceProperties device_properties;
    constexpr ck::index_t k_batch_min = 1;
    constexpr ck::index_t batch_size_min = 512;

    const int num_cu = device_properties.num_cu_;
    const auto k_batch_max = math::integer_divide_ceil(K_size, batch_size_min);
    // Ensure that we do not exceed the maximum capacity. This would lead to wave quantization.
    const auto optimal_split = static_cast<ck::index_t>(std::floor((max_occupancy * num_cu) / (1.0 * grid_size * conv_G)));
    auto k_batch = 1;
    if (optimal_split > 1)
    {
      //The optimal value of k_batch is a multiple of the optimal_split.
      //We need to find the optimal number K values per batch - this gives the optimal k_batch value.
      auto target_batch_size = static_cast<ck::index_t>(ck::EnvValue(CK_ENV(CK_SPLIT_K_BATCH_SIZE)));
      if (target_batch_size < k_batch_min)
      {
        target_batch_size = k_batch_min;
      }
      k_batch = optimal_split;
      const auto current_batch_size = math::integer_divide_ceil(K_size, k_batch);
      if (current_batch_size > target_batch_size)
      {
        // If the current batch size is larger than the target batch size, we need to increase k_batch.
        const ck::index_t multiplier = std::max(1, math::integer_divide_ceil(K_size, target_batch_size * optimal_split));
        k_batch = optimal_split * multiplier;
      }
    }
    
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Output grid size (M tiles x N tiles x Conv groups):  " << grid_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] K-dim size:  " << K_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Conv groups:  " << conv_G << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split value:  " << optimal_split << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Maximum k_batch value:  " << k_batch_max << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << " for K-batch."<< std::endl;
    }
    return k_batch;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
