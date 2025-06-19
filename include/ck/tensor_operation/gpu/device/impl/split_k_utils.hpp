// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/utility/number.hpp"
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
    max_num_active_wavefronts_per_cu_ = dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize;
    wavefront_size_ = dev_prop.warpSize;
  };
  int num_cu_;
  int max_num_active_wavefronts_per_cu_;
  int wavefront_size_;
};

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size, ck::index_t blockSize, bool enable_oversubscription = true)
{
    static DeviceProperties device_properties;
    const int num_cu = device_properties.num_cu_;
    auto k_batch = 1;

    const ck::index_t oversubscription = enable_oversubscription
      ? static_cast<ck::index_t>(std::round((1.0 *device_properties.max_num_active_wavefronts_per_cu_ * device_properties.wavefront_size_) / blockSize))
      : 1;

    const auto optimal_split = static_cast<ck::index_t>(std::floor((1.0 *max_occupancy * num_cu) / (grid_size)));
    if (optimal_split > 1)
    {
      k_batch = oversubscription * optimal_split;
    }
    
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Block size:  " << blockSize << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Oversubscription factor:  " << oversubscription << " (oversubscription enabled = " << std::to_string(enable_oversubscription) << ")"<< std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split value:  " << optimal_split << std::endl;
      std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << " for K-batch."<< std::endl;
    }
    return k_batch;
}

template <ck::index_t NDimSpatial>
inline index_t get_bwd_weight_gemm_k(const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths)
{
  static constexpr auto I1 = Number<1>{};

  // The input array has elements in the order: G, N, K, Do, Ho, Wo
  // GemmK = N * Do * Ho * Wo for the BWD weight pass.
  constexpr index_t spatial_offset = 3; 
  const index_t DoHoWo = std::accumulate(begin(a_g_n_k_wos_lengths) + spatial_offset,
                                      end(a_g_n_k_wos_lengths),
                                      index_t{1},
                                      std::multiplies<>{});
  const auto gemmK = a_g_n_k_wos_lengths[I1] * DoHoWo;
  return gemmK;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
