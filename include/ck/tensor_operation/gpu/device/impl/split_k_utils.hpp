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

inline ck::index_t get_best_occupancy_k_batch_value(int max_occupancy, ck::index_t grid_size)
{
    static DeviceProperties device_properties;
    const int num_cu = device_properties.num_cu_;
    ck::index_t k_batch = 1;

    const auto optimal_split = static_cast<ck::index_t>(std::floor((1.0 *max_occupancy * num_cu) / (grid_size)));
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
inline auto get_bwd_weight_gemm_sizes(
  const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
  const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths)
{
  static constexpr auto I1 = Number<1>{};
  static constexpr auto I2 = Number<2>{};

  // The input array has elements in the order: G, N, K, Do, Ho, Wo
  // GemmK = N * Do * Ho * Wo for the BWD weight pass.
  constexpr index_t spatial_offset = 3; 
  const index_t DoHoWo = std::accumulate(begin(a_g_n_k_wos_lengths) + spatial_offset,
                                      end(a_g_n_k_wos_lengths),
                                      index_t{1},
                                      std::multiplies<>{});
  const auto gemmK = a_g_n_k_wos_lengths[I1] * DoHoWo;

  // The GEMM M dimension is the number of output channels.
  const auto gemmM = e_g_k_c_xs_lengths[I1];

  // The output array has elements in the order: G, K, C, X, Y, Z
  // GemmN = C * X * Y * Z for the BWD weight pass.
  const index_t XYZ = std::accumulate(begin(e_g_k_c_xs_lengths) + spatial_offset,
                                      end(e_g_k_c_xs_lengths),
                                      index_t{1},
                                      std::multiplies<>{});
  const auto gemmN = e_g_k_c_xs_lengths[I2] * XYZ;
  return std::make_tuple(gemmM, gemmN, gemmK);
}

inline ck::index_t get_optimized_k_batch_value(int max_occupancy, ck::index_t grid_size_mn, ck::index_t grid_size_k)
{
  static DeviceProperties device_properties;
  const int num_compute_units = device_properties.num_cu_;
  const auto nproc = max_occupancy * num_compute_units;
  
  const auto max_split_k = grid_size_k;
  double best_score = 0;
  ck::index_t best_split_k = 1;

  for (ck::index_t split_k = 1; split_k <= max_split_k; ++split_k)
  {
    const auto k_tiles_per_split = (grid_size_k + split_k - 1) / split_k;

    // Tail loop cost - the split_k may not divide grid_k evenly, leading to a tail loop
    const auto tail_loop_score = (grid_size_k % split_k == 0) ? 0 : 1.0 / (grid_size_k % split_k);

    // Check load balance - how evenly can we distribute the splits
    const auto total_blocks = split_k * grid_size_mn;
    const auto load_balance_score = total_blocks % nproc == 0 ? 1.0 : (nproc - (total_blocks % nproc)) / nproc;

    // Compute cache locality score based on k-tile size per split
    // Smaller k_tiles_per_split generally means better cache reuse within each split
    const auto cache_locality_score = k_tiles_per_split > 0 ? 1.0 / k_tiles_per_split : 0;

    // Synchronization overhead increases with more splits, i.e., the score gets lower
    const auto sync_overhead_score = 1.0 / (1.0 + split_k);

    const auto total_score = 0.5*load_balance_score + 0.3*cache_locality_score + 0.1*sync_overhead_score + 0.1*tail_loop_score;

    if (total_score > best_score)
    {
      best_score = total_score;
      best_split_k = split_k;
    }
  }

  if (ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
  {
    std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  " << max_occupancy << std::endl;
    std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size_mn << std::endl;
    std::cout << "[SPLIT-K AUTODEDUCE] K grid size:  " << grid_size_k << std::endl;
    std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << best_split_k << " for K-batch."<< std::endl;
  }

  return best_split_k;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
