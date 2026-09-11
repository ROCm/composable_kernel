// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/stream_utility.hpp"

#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/// @brief Grouped GEMM kernel using output Tile Looping algorithm
///
/// @par This kernel does not require any knowledge about input data sizes (GEMM M/N/K)
///       It requires only the number of groups to launch. Other information like
///       data pointers and GEMM sizes, packed into gemm kernel args may be all dynamic
///       (known only at kernel run-time).
///
/// @note This kernel does not support SplitK.

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedGemmTileLoop : public DeviceGroupedGemm<ALayout,
                                                            BLayout,
                                                            DsLayout,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            DsDataType,
                                                            EDataType,
                                                            AElementwiseOperation,
                                                            BElementwiseOperation,
                                                            CDEElementwiseOperation>
{
};

template <ck::index_t BlockSize>
struct TileLoopKernelConfig
{
    // The oversubscription factor for the number of blocks that can simultaneously reside on
    // GPU.
    static constexpr int BLOCK_SUBSCRIPTION_FACTOR = 1;
    // static constexpr int BLOCK_WAVES               = BlockSize / get_warp_size();
    static constexpr int CU_SIMDS = 4;
    // Assume we want to have at most 2 waves per SIMD
    // static constexpr int CU_BLOCKS = math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);
    static int GetCuBlocks()
    {
        int BLOCK_WAVES = BlockSize / get_warp_size();
        return ck::math::integer_divide_floor(2 * CU_SIMDS, BLOCK_WAVES);
    }

    template <typename KernelFunction>
    static int CalculateMaxOccupancyGridSize(const KernelFunction& kernel,
                                             const StreamConfig& stream_config)
    {
        // Calculate max number of workgroups that can simultaneously reside on the CU.
        int occ_num_blocks = GetKernelOccupancy(kernel);
        int cu_count       = getAvailableComputeUnitCount(stream_config);

        if(stream_config.log_level_ > 0)
        {
            std::cout << "MaxActiveBlocksPerCU: " << occ_num_blocks
                      << ", available CUs count: " << cu_count << ", occup. grid size: "
                      << ck::math::min(occ_num_blocks, GetCuBlocks()) * cu_count << std::endl;
        }

        return cu_count * ck::math::min(occ_num_blocks, GetCuBlocks());
    }

    template <typename KernelFunction>
    static int GetKernelOccupancy(const KernelFunction& kernel)
    {
        int occupancy = 0;
        ck::hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize, 0));
        return std::max(occupancy, 1);
    }

    static int GetComputeUnitCount()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        ck::hip_check_error(hipGetDevice(&dev));
        ck::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        return dev_prop.multiProcessorCount;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
