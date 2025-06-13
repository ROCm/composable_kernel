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

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size)
{
    static DeviceProperties device_properties;
    const int num_cu = device_properties.num_cu_;
    auto k_batch = 1;

    constexpr ck::index_t num_waves = 1;
    const auto optimal_split = static_cast<ck::index_t>(std::floor((max_occupancy * num_cu) / (num_waves * grid_size)));
    if (optimal_split > 1)
    {
      k_batch = optimal_split;
    }
    
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Output grid size (M tiles x N tiles):  " << grid_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split value:  " << optimal_split << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << " for K-batch."<< std::endl;
    }
    return k_batch;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
