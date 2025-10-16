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

    /// @brief Constructs kernel arguments for the Stream-K kernel.
    /// @param host_args Stream-K host arguments.
    /// @param num_cu Number of compute units (CUs). The default is the number of CUs on the device.
    /// The caller may select their own to assist with test reproducibility, etc.
    /// @param occupancy The maximum number of active blocks per CU for this kernel. The caller may
    /// select their own to assist with test reproducibility, etc.
    /// @return The kernel arguments for Stream-K.
    CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args,
                                                         int num_cu    = NumCU(),
                                                         int occupancy = Occupancy())
    {
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
                                                 static_cast<uint32_t>(num_cu),
                                                 static_cast<uint32_t>(occupancy),
                                                 host_args.num_sk_blocks}};
    }

    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunGemm(const std::array<const ADataType*, UniversalGemmKernel::NumATensor>& as_ptr,
            const std::array<const BDataType*, UniversalGemmKernel::NumBTensor>& bs_ptr,
            const std::array<const void*, UniversalGemmKernel::NumDTensor>& ds_ptr,
            CDataType* c_ptr,
            void* smem_ptr_0,
            const typename UniversalGemmKernel::KernelArgs& kargs,
            const index_t num_loop,
            const index_t block_idx_m,
            const index_t block_idx_n,
            const index_t k_size)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            UniversalGemmKernel::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, c_ptr, kargs, k_size);

        const auto& gemm_pad_views = UniversalGemmKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            UniversalGemmKernel::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& as_block_window = gemm_tile_windows.at(UniversalGemmKernel::I0);
        const auto& bs_block_window = gemm_tile_windows.at(UniversalGemmKernel::I1);
        const auto& ds_block_window = gemm_tile_windows.at(UniversalGemmKernel::I2);

        // Since num_loop can vary per WG and per iteration of the Stream-K while loop, we compute
        // has_hot_loop and tail_num here. This is a similar pattern used by grouped GEMM. In this
        // case, we call the GemmPipeline's operator() function that takes both has_hot_loop and
        // tail_num.
        const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto& c_block_tile = GemmPipeline{}(as_block_window[UniversalGemmKernel::I0],
                                                  bs_block_window[UniversalGemmKernel::I0],
                                                  num_loop,
                                                  has_hot_loop,
                                                  tail_num,
                                                  smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(UniversalGemmKernel::I3);

            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    CK_TILE_HOST static bool IsSupportedArgument(const StreamKKernelArgs& kargs)
    {
        if(kargs.reduction_strategy == StreamKReductionStrategy::Reduction)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("CK Tile Stream-K only supports the atomic reduction strategy.");
            }
            return false;
        }
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

    /// @brief Entry point for the Stream-K Kernel, performing the main Stream-K loop.
    CK_TILE_DEVICE void operator()(StreamKKernelArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];

        uint32_t block_idx = ck_tile::get_block_1d_id();

        bool is_padding_block =
            amd_wave_read_first_lane(block_idx >= kargs.tile_partitioner.sk_num_blocks &&
                                     block_idx < kargs.tile_partitioner.dp_start_block_idx);

        // Padding blocks make it such that the DP blocks are aligned with the number of CUs; they
        // should not partake in the GEMM
        if(is_padding_block)
            return;

        // Determine the K offset of the first and final macro tile in the A and B tensors along the
        // K dimension.
        uint32_t iter_start, iter_end;
        kargs.tile_partitioner.GetBlockItr(block_idx, iter_start, iter_end);

        // Main Stream-K loop
        while(true)
        {
            // Determine the number of macro tiles in A and B this WG is resposible for in the
            // current C macro tile.
            uint32_t current_iter_length = amd_wave_read_first_lane(
                kargs.tile_partitioner.GetCurrentIterLength(iter_start, iter_end));

            // Determine the 1D tile_idx and the iter_offset for this WG.
            // The tile_idx is the 1D macro tile index in the C tensor.
            // The iter_offset is the starting macro tile index in the K dimension for the WG in the
            // current iteration of the while loop.
            uint32_t tile_idx, iter_offset;
            kargs.tile_partitioner.GetTileIdxWithOffset(iter_start, tile_idx, iter_offset);

            // Get the 2D tile index in the C tensor for this WG using the 1D index (i.e. tile_idx)
            auto spatial_idx = kargs.tile_partitioner.GetOutputTileIndex(tile_idx);

            // Get the offsets in A, B, C tensors.
            index_t i_m         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I0] *
                                               TilePartitioner::MPerBlock);
            index_t i_n         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I1] *
                                               TilePartitioner::NPerBlock);
            auto [i_k_a, i_k_b] = GetKOffsets<ALayout, BLayout>(
                static_cast<index_t>(iter_offset), kargs.stride_As[0], kargs.stride_Bs[0]);

            // Determine the total size along the K dimension the WG is using in this iteration
            // (used to construct tensor views).
            index_t k_size = static_cast<index_t>(current_iter_length * TilePartitioner::KPerBlock);

            // Update pointer offsets for A, B, and C.
            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
            const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
            CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

            // Run the GEMM pipeline and Epilogue.
            RunGemm({a_ptr},
                    {b_ptr},
                    {/*ds_ptr*/},
                    c_ptr,
                    smem_ptr_0,
                    kargs,
                    current_iter_length,
                    i_m,
                    i_n,
                    k_size);

            // Prepare for next Stream-K loop iteration.
            iter_start += current_iter_length;
            if(iter_end <= iter_start)
                break;
            block_sync_lds();
        }
    }

    private:
    /// @brief Computes the K offsets in the A and B tensors given iter_offset, where iter_offset is
    /// the starting macro tile index in the K dimension for the workgroup.
    /// @return A tuple containing the offsets into the A and B tensors accounting for the layouts
    /// of A and B.
    /// @note The default case is that A is assumed to be row major and B is assumed to be column
    /// major.
    template <typename ALayout, typename BLayout>
    CK_TILE_DEVICE static tuple<index_t, index_t>
    GetKOffsets(index_t iter_offset, index_t stride_a, index_t stride_b)
    {
        index_t stride_offset_a;
        index_t stride_offset_b;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            stride_offset_a = stride_a;
        }
        else
        {
            stride_offset_a = 1;
        }

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_offset_b = stride_b;
        }
        else
        {
            stride_offset_b = 1;
        }

        index_t base_offset = iter_offset * TilePartitioner::KPerBlock;

        return make_tuple(base_offset * stride_offset_a, base_offset * stride_offset_b);
    }

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
