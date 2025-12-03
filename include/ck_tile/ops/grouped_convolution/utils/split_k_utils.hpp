// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <numeric>

#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

//
// Return actual kernel that can be passed e.g. to  hipOccupancyMaxActiveBlocksPerMultiprocessor.
// The KernelImpl should be a class without non-static data member, or let's say
// can be instantiate with "KernelImpl{}"
//
// the "static __device__ operator()(some_arg)" is the entry point of KernelImpl
//
template <int MinBlockPerCu,
          typename KernelImpl,
          typename KernelArgs>
CK_TILE_HOST auto
make_kernel(KernelImpl)
{
    const auto kernel = []() {
        return kentry<MinBlockPerCu, KernelImpl, KernelArgs>;
    }();
    return kernel;
}

template <typename Kernel, index_t BlockSize>
CK_TILE_HOST index_t get_max_occupancy_for_kernel()
{
    constexpr int dynamic_smem_size = 0;
    constexpr int min_blocks_per_cu = 1;

    using KernelArgs = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized;
    const auto& kernel = make_kernel<min_blocks_per_cu, Kernel, KernelArgs>(Kernel{});

    int max_occupancy = 0;
    hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_occupancy,
                kernel,
                BlockSize,
                dynamic_smem_size));

    return static_cast<index_t>(max_occupancy);
}

CK_TILE_HOST index_t get_best_occupancy_k_batch_value(index_t max_occupancy, index_t grid_size)
{
    static const index_t num_cus = get_num_cus();
    const index_t max_capacity = max_occupancy * num_cus;

    index_t k_batch = 1;
    const auto optimal_split =
        static_cast<index_t>(std::floor((1.0 * max_capacity) / grid_size));
    if(optimal_split > 1)
    {
        k_batch = optimal_split;
    }

    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
    {
        std::cout << "[SPLIT-K AUTODEDUCE] Max active thread blocks per CU for GEMM kernel:  "
                  << max_occupancy << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Output grid size:  " << grid_size << std::endl;
        std::cout << "[SPLIT-K AUTODEDUCE] Optimal split-k value " << k_batch << std::endl;
    }
    return k_batch;
}

}
