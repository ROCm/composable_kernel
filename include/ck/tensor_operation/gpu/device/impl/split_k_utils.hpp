// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/ck.hpp"

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

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size, ck::index_t K_size, ck::index_t conv_G)
{
    static DeviceProperties device_properties;
    constexpr ck::index_t k_batch_min = 1;
    constexpr ck::index_t batch_size_min = 16;

    const int num_cu = device_properties.num_cu_;
    const auto k_batch_max = math::integer_divide_ceil(K_size, batch_size_min);
    auto k_batch = static_cast<ck::index_t>(std::ceil((max_occupancy * num_cu) / (1.0 * grid_size))); // Exclude th egrid size from the occupancy calculation
    k_batch = std::min(std::max(k_batch_min, k_batch), k_batch_max);
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Output grid size (M tiles x N tiles x Conv groups):  " << grid_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] K-dim size:  " << K_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Conv groups:  " << conv_G << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Maximum k_batch value:  " << k_batch_max << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << " for K-batch."<< std::endl;
    }
    return k_batch;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
