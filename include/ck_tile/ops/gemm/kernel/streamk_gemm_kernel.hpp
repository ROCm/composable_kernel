// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

/// @brief The Stream K GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref StreamKKernel "StreamKKernel" when creating the kernel
///      arguments object. It contains all necessary information required to build proper kernel
///      arguments and launch the kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
struct StreamKHostArgs : public ck_tile::UniversalGemmHostArgs<>
{
    CK_TILE_HOST explicit StreamKHostArgs(const void* a_ptr_,
                                          const void* b_ptr_,
                                          void* c_ptr_,
                                          index_t M_,
                                          index_t N_,
                                          index_t K_,
                                          index_t stride_A_,
                                          index_t stride_B_,
                                          index_t stride_C_,
                                          StreamKReductionStrategy reduction_strategy_,
                                          uint32_t num_sk_blocks_ = 0xffffffff)
        : UniversalGemmHostArgs<>({a_ptr_},
                                  {b_ptr_},
                                  {/*ds_ptr*/},
                                  c_ptr_,
                                  /*k_batch_ =*/1,
                                  M_,
                                  N_,
                                  K_,
                                  {stride_A_},
                                  {stride_B_},
                                  {/*stride_Ds_*/},
                                  stride_C_),
          reduction_strategy{reduction_strategy_},
          num_sk_blocks{num_sk_blocks_}
    {
    }

    ck_tile::StreamKReductionStrategy reduction_strategy;
    uint32_t num_sk_blocks;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    /// @brief  Specify the layout configurations for A, B, and C
    using ALayout = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout = remove_cvref_t<typename GemmPipeline::CLayout>;

    /// @brief  Specify the data type configurations for A, B, and C
    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    /// @brief  ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, ALayout>::value &&
                      !is_detected<is_tuple, ADataType>::value,
                  "ALayout and ADataType must be scalars.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, BLayout>::value &&
                      !is_detected<is_tuple, BDataType>::value,
                  "BLayout and BDataType must be scalars.");

    /// @brief  CLayout and CDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, CDataType>::value,
                  "CLayout and CDataType must be scalars.");

    struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<>
    {
        /// @brief  The strategy used by work groups to compute final results in C tensor.
        StreamKReductionStrategy reduction_strategy;
        /// @brief  The number of stream k blocks.
        uint32_t num_sk_blocks;
        /// @brief  A pointer to a buffer in device memory for accumulating partial via reduction
        /// strategy.
        void* workspace_ptr;
        /// @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
        /// the C tensor.
        TilePartitioner tile_partitioner;
    };

    using KernelArgs = StreamKKernelArgs;
    using Kernel     = StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        using WarpTile = typename P_::BlockGemmShape::WarpTile;

        return concat('_', "streamk", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', WarpTile::at(number<0>{}), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    /// @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
    /// @return The grid size.
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner) -> dim3
    {
        return tile_partitioner.GridSize();
    }

    /// @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
    /// @return The maximum occupancy grid size.
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return UniversalGemmKernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args)
    {
        uint32_t occupancy = static_cast<uint32_t>(Occupancy());
        uint32_t num_cu    = static_cast<uint32_t>(NumCU());

        return StreamKKernelArgs{{host_args.as_ptr,
                                  host_args.bs_ptr,
                                  host_args.ds_ptr,
                                  host_args.e_ptr,
                                  host_args.M,
                                  host_args.N,
                                  host_args.K,
                                  host_args.stride_As,
                                  host_args.stride_Bs,
                                  host_args.stride_Ds,
                                  host_args.stride_E,
                                  host_args.k_batch},
                                 host_args.reduction_strategy,
                                 host_args.num_sk_blocks,
                                 // The workspace pointer is set to nullptr because we must first
                                 // instantiate the TilePartitioner to get the necessary size
                                 /*workspace_ptr =*/nullptr,
                                 TilePartitioner{static_cast<uint32_t>(host_args.M),
                                                 static_cast<uint32_t>(host_args.N),
                                                 static_cast<uint32_t>(host_args.K),
                                                 num_cu,
                                                 occupancy,
                                                 host_args.num_sk_blocks}};
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const typename UniversalGemmKernel::KernelArgs& kargs)
    {
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    /// @brief Computes the buffer size needed to store accumulation results for Stream K.
    /// @return The buffer size needed.
    CK_TILE_HOST static uint32_t GetWorkSpaceSize(const StreamKKernelArgs& kargs)
    {
        // For reduction, we need to determine the amount of device space for acculumation
        // results and semaphores.
        if(kargs.reduction_strategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            return kargs.tile_partitioner.GetWorkSpaceSize(sizeof(CDataType));
        }

        // Otherwise, no additional space is needed since blocks atomically store their results.
        return 0;
    }

    /// @brief Sets the kargs' current workspace_ptr to the given workspace_ptr.
    /// @note Assumes that the given workspace_ptr points to allocated device memory.
    CK_TILE_HOST static void SetWorkSpacePointer(StreamKKernelArgs& kargs, void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    // Temporary placeholder to support the Occupancy() static function.
    // Since the Occupancy function uses kentry, this class must have an operator() function
    CK_TILE_DEVICE void operator()(StreamKKernelArgs /*kargs*/) const {}

    private:
    CK_TILE_HOST static int NumCU()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        int num_cu = dev_prop.multiProcessorCount;

        return num_cu;
    }

    /// @brief Computes the occupancy (i.e. maximum number of active blocks per CU) for the kernel
    /// @return The occupancy
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static int Occupancy()
    {
        int occupancy;

        // Since occupancy of 1 is valid for stream k, we set min_num_block_per_cu to 1
        constexpr int min_block_per_cu = 1;
        const auto kernel              = kentry<min_block_per_cu, Kernel, KernelArgs>;

        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kBlockSize, 0));

        return occupancy;
    }
};

} // namespace ck_tile
