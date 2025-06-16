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

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size, ck::index_t k_size)
{
    static DeviceProperties device_properties;
    const int num_cu = device_properties.num_cu_;
    auto k_batch = 1;
    //constexpr ck::index_t min_k_per_batch = 16;
    //const auto max_split_k = math::integer_divide_ceil(k_size, min_k_per_batch);

    const auto optimal_split = static_cast<ck::index_t>(std::floor((max_occupancy * num_cu) / (grid_size)));
    if (optimal_split > 1)
    {
      //k_batch = std::min(optimal_split, max_split_k);
      k_batch = optimal_split;
    }
    
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] K-dim size:  " << k_size << std::endl;
      //std::cout << "[SPLIT-K AUTODEDUCE] Max split-k value:  " << max_split_k << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split value:  " << optimal_split << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << " for K-batch."<< std::endl;
    }
    return k_batch;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
