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

template<
        ck::index_t MPerBlock,
        ck::index_t NPerBlock>
ck::index_t get_k_batch_value(ck::index_t split_k, int max_occupancy, ck::index_t M, ck::index_t N, ck::index_t conv_G)
{
    static DeviceProperties device_properties;
    // For now, assume that negative (or zero) value signals automatic computation of the split_k value.
    if(split_k <= 0)
    {
      const int num_cu = device_properties.num_cu_;
      const auto M0 = math::integer_divide_ceil(M, MPerBlock);
      const auto N0 = math::integer_divide_ceil(N, NPerBlock);
      const auto n_output_tiles = M0 * N0;
      const auto k_batch = std::ceil((max_occupancy * num_cu) / (1.0 * n_output_tiles * conv_G));
      if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
      {
        std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Overriding user deinfed split_k value " << split_k << " to optimal value " << k_batch << " for K-batch."<< std::endl;
      }
      return k_batch;
    }
    return split_k;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
