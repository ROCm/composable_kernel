// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <hip/hip_runtime.h>
#include "ck/utility/env.hpp"
#include "ck/utility/number.hpp"
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
    max_num_active_wavefronts_per_cu_ = dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize;
    wavefront_size_ = dev_prop.warpSize;
  };
  int num_cu_;
  int max_num_active_wavefronts_per_cu_;
  int wavefront_size_;
};

inline ck::index_t get_k_batch_value(int max_occupancy, ck::index_t grid_size)
{
    static DeviceProperties device_properties;
    const int num_cu = device_properties.num_cu_;
    ck::index_t k_batch = 1;

    const auto optimal_split = static_cast<ck::index_t>(std::round((1.0 *max_occupancy * num_cu) / (grid_size)));
    if (optimal_split > 1)
    {
      k_batch = optimal_split;
    }
    
    if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
      std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
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

inline ck::index_t get_closest_full_wave_value(ck::index_t split_k_value, ck::index_t grid_size)
{
  static DeviceProperties device_properties;
  const int num_compute_units = device_properties.num_cu_;

  if ((split_k_value * grid_size) % num_compute_units == 0)
  {
    return split_k_value;
  }

  int best_split_k_modification =0;
  int min_cost = std::numeric_limits<int>::max();
  for (int k = -split_k_value + 1; k < num_compute_units - split_k_value; ++k)
  {
    const auto remainder = ((k + split_k_value) * grid_size) % num_compute_units;
    const auto wave_quant_cost = (N-remainder)*(N-remainder);
    const auto relative_k_change = 100.0 * (k/split_k_value);
    const auto k_change_cost = relative_k_change * relative_k_change;
    const auto cost = k_change_cost + 2.0*wave_quant_cost;
    if (cost < min_cost)
    {
      min_cost = cost;
      best_split_k_modification = k;
    }
    else if (std::abs(cost - min_cost) < std::numeric_limits<double>::epsilon())
    {
      // For equally good candidates, select the one with smaller absolute value
      if (std::abs(k) < std::abs(best_split_k_modification))
      {
        best_split_k_modification = k;
      }
      else if (std::abs(k) == std::abs(best_split_k_modification))
      {
        // If absolute values are equal, prefer the larger one
        best_split_k_modification = std::max(best_split_k_modification, k);
      }
    }
  }

  return split_k_value + best_split_k_modification;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
