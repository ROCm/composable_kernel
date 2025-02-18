// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_gemm_xdl_splitk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                       const index_t group_count)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx94__))
    constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
    __shared__ uint8_t p_shared[shared_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_id >= gemm_desc_ptr[group_id].block_start_ &&
             block_id < gemm_desc_ptr[group_id].block_end_)) &&
          left <= right)
    {
        if(block_id < gemm_desc_ptr[group_id].block_start_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    const auto karg          = gemm_desc_ptr[group_id].karg_;
    auto splitk_batch_offset = GridwiseGemm::SplitKBatchOffset(karg);

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
        karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
        karg.p_c_grid,
        p_shared,
        karg);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false,
          // MultipleD not supported for now.
          enable_if_t<is_same_v<DsLayout, ck::Tuple<>> && is_same_v<DsDataType, ck::Tuple<>>,
                      bool> = false>

struct DeviceGroupedGemmXdlSplitKCShuffle : public DeviceGroupedGemmSplitK<ALayout,
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
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using GridwiseGemm = GridwiseGemm_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        ELayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    using Block2ETileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;

    using GroupedGemmBlock2ETileMap = OffsettedBlockToCTileMap<Block2ETileMap>;
    using KernelArgument            = typename GridwiseGemm::Argument;
    using PassThrough               = ck::tensor_operation::element_wise::PassThrough;

    struct GemmTransKernelArg
    {
        KernelArgument karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        // GemmTransKernelArg() = default;
        GemmTransKernelArg(KernelArgument&& karg,
                           GroupedGemmBlock2ETileMap&& b2c_map,
                           index_t block_start,
                           index_t block_end)
            : karg_{karg},
              block_2_ctile_map_{b2c_map},
              block_start_{block_start},
              block_end_{block_end}
        {
        }
    };

    static constexpr index_t DefaultKBatch = 1;

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_a_grid,
                 std::vector<const void*>& p_b_grid,
                 std::vector<void*>& p_c_grid,
                 std::vector<GemmDesc>& gemm_descs)
            : Argument(p_a_grid, p_b_grid, p_c_grid, gemm_descs, DefaultKBatch)
        {
            // TODO: use occupancy api to calculate appropriate batch size.
        }

        Argument(std::vector<const void*>& p_a_grid,
                 std::vector<const void*>& p_b_grid,
                 std::vector<void*>& p_c_grid,
                 std::vector<GemmDesc>& gemm_descs,
                 index_t kbatch)
            : K_BATCH{kbatch}
        {
            grid_size_   = 0;
            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_a_grid.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_b_grid.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_c_grid.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_a_grid/b/c.size");
            }

            gemm_kernel_args_.reserve(group_count_);

            skipped_group_count_ = 0;

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(M * N * K == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;
                const index_t stride_c = gemm_descs[i].stride_C_;

                const auto local_b2c_tile_map = Block2ETileMap{M, N, 4};
                index_t grid_size_grp         = local_b2c_tile_map.CalculateGridSize(M, N);
                grid_size_grp *= K_BATCH;

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                KernelArgument karg{type_convert<const ADataType*>(p_a_grid[i]),
                                    type_convert<const BDataType*>(p_b_grid[i]),
                                    type_convert<EDataType*>(p_c_grid[i]),
                                    M,
                                    N,
                                    K,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    K_BATCH};

                gemm_kernel_args_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);
            }
        }

        /**
         * @brief      Recalculate group grid size for all gemms and update B2C maps.
         *
         * @param[in]  kbatch  The new splitK parameter value.
         */
        void UpdateKBatch(index_t kbatch)
        {
            K_BATCH    = kbatch;
            grid_size_ = 0;

            for(std::size_t i = 0; i < gemm_kernel_args_.size(); ++i)
            {

                auto& karg = gemm_kernel_args_[i].karg_;

                const auto local_b2c_tile_map = Block2ETileMap{karg.M, karg.N, 4};
                index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(karg.M, karg.N);
                grid_size_grp *= K_BATCH;

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;
                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                karg.KBatch                             = K_BATCH;
                gemm_kernel_args_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_kernel_args_[i].block_start_       = block_start;
                gemm_kernel_args_[i].block_end_         = block_end;
            }
        }

        //  private:
        index_t K_BATCH;
        index_t group_count_;
        index_t skipped_group_count_;

        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t grid_size_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto& karg0 = arg.gemm_kernel_args_[0].karg_;
            index_t k_grain0  = karg0.KBatch * KPerBlock;
            index_t K_split0  = (karg0.K + k_grain0 - 1) / k_grain0 * KPerBlock;

            bool all_have_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split0);
            const auto tail_num             = GridwiseGemm::CalculateKBlockLoopTailNum(K_split0);
            bool all_have_kbatch_gt_one     = karg0.KBatch > 1;

            for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& karg = arg.gemm_kernel_args_[i].karg_;

                if(stream_config.log_level_ > 0)
                {
                    karg.Print();
                }

                index_t k_grain = karg.KBatch * KPerBlock;
                index_t K_split = (karg.K + k_grain - 1) / k_grain * KPerBlock;

                bool not_all_have_main_k0_block_loop_same =
                    all_have_main_k_block_loop xor
                    GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

                bool not_all_have_tail_num_same =
                    (tail_num == GridwiseGemm::CalculateKBlockLoopTailNum(K_split));

                if(not_all_have_main_k0_block_loop_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same value for main_k0_block_loop! in " << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                if(not_all_have_tail_num_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same TailNumber value! in " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                if(!GridwiseGemm::CheckValidity(karg))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }

            hip_check_error(
                hipMemcpyAsync(arg.p_workspace_,
                               arg.gemm_kernel_args_.data(),
                               arg.gemm_kernel_args_.size() * sizeof(GemmTransKernelArg),
                               hipMemcpyHostToDevice,
                               stream_config.stream_id_));

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    KernelArgument arg_ = arg.gemm_kernel_args_[0].karg_;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);

                    auto size_a_buffer =
                        a_grid_desc_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType);
                    auto size_b_buffer =
                        b_grid_desc_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType);

                    ck::utility::RotatingMemWrapper<Argument> rotating_mem(
                        arg_, stream_config.rotating_count, size_a_buffer, size_b_buffer);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        // TODO: should be loop here through all groups
                        for(const auto& trans_arg : arg.gemm_kernel_args_)
                        {
                            const auto& karg = trans_arg.karg_;
                            if(karg.KBatch > 1)
                                hipGetErrorString(
                                    hipMemsetAsync(karg.p_c_grid,
                                                   0,
                                                   karg.M * karg.N * sizeof(EDataType),
                                                   stream_config.stream_id_));
                        }
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(arg.grid_size_),
                        dim3(BlockSize),
                        0,
                        cast_pointer_to_constant_address_space(arg.p_workspace_),
                        arg.gemm_kernel_args_.size());
                }
                else
                {
                    // TODO: should be loop here through all groups
                    for(const auto& trans_arg : arg.gemm_kernel_args_)
                    {
                        const auto& karg = trans_arg.karg_;
                        if(karg.KBatch > 1)
                            hipGetErrorString(hipMemsetAsync(karg.p_c_grid,
                                                             0,
                                                             karg.M * karg.N * sizeof(EDataType),
                                                             stream_config.stream_id_));
                    }

                    ave_time = launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(arg.grid_size_),
                        dim3(BlockSize),
                        0,
                        cast_pointer_to_constant_address_space(arg.p_workspace_),
                        arg.gemm_kernel_args_.size());
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(all_have_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                           GemmTransKernelArg,
                                                           true,
                                                           InMemoryDataOperationEnum::AtomicAdd,
                                                           minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                           GemmTransKernelArg,
                                                           true,
                                                           InMemoryDataOperationEnum::Set,
                                                           minimum_occupancy>;
                        Run(kernel);
                    }
                }

                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        if(tail_num == TailNumber::One)
                        {
                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::AtomicAdd,
                                                               minimum_occupancy,
                                                               TailNumber::One>;
                            Run(kernel);
                        }

                        //// TODO: Fix below as above!

                        else if(tail_num == TailNumber::Full)
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::AtomicAdd,
                                                               minimum_occupancy,
                                                               TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(tail_num == TailNumber::Two)
                            {

                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(tail_num == TailNumber::Three)
                            {

                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(tail_num == TailNumber::Four)
                            {
                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(tail_num == TailNumber::Five)
                            {
                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(tail_num == TailNumber::Six)
                            {
                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(tail_num == TailNumber::Seven)
                            {
                                const auto kernel = kernel_grouped_gemm_xdl_splitk<
                                    GridwiseGemm,
                                    GemmTransKernelArg,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }

                    else
                    {
                        if(tail_num == TailNumber::One)
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::Set,
                                                               minimum_occupancy,
                                                               TailNumber::One>;
                            Run(kernel);
                        }
                        else if(tail_num == TailNumber::Full)
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::Set,
                                                               minimum_occupancy,
                                                               TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(tail_num == TailNumber::Two)
                            {

                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(tail_num == TailNumber::Three)
                            {

                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(tail_num == TailNumber::Four)
                            {

                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(tail_num == TailNumber::Five)
                            {

                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(tail_num == TailNumber::Six)
                            {

                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(tail_num == TailNumber::Seven)
                            {
                                const auto kernel =
                                    kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   true,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   minimum_occupancy,
                                                                   TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        if(tail_num == TailNumber::Odd)
                        {

                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                GemmTransKernelArg,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {

                            const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                GemmTransKernelArg,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(tail_num == TailNumber::Odd)
                        {

                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                                 GemmTransKernelArg,
                                                                 true,
                                                                 InMemoryDataOperationEnum::Set,
                                                                 minimum_occupancy,
                                                                 TailNumber::Odd>;
                            Run(kernel);
                        }

                        else
                        {

                            const auto kernel =
                                kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                                 GemmTransKernelArg,
                                                                 true,
                                                                 InMemoryDataOperationEnum::Set,
                                                                 minimum_occupancy,
                                                                 TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }

                else
                {
                    if(all_have_kbatch_gt_one)
                    {
                        if(tail_num == TailNumber::Odd)
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::AtomicAdd,
                                                               minimum_occupancy,
                                                               TailNumber::Odd>;
                            Run(kernel);
                        }

                        else
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::AtomicAdd,
                                                               minimum_occupancy,
                                                               TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(tail_num == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::Set,
                                                               minimum_occupancy,
                                                               TailNumber::Odd>;
                            Run(kernel);
                        }

                        else
                        {

                            const auto kernel =
                                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                               GemmTransKernelArg,
                                                               true,
                                                               InMemoryDataOperationEnum::Set,
                                                               minimum_occupancy,
                                                               TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                           GemmTransKernelArg,
                                                           false,
                                                           InMemoryDataOperationEnum::AtomicAdd,
                                                           minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                           GemmTransKernelArg,
                                                           false,
                                                           InMemoryDataOperationEnum::Set,
                                                           minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        if((ck::type_convert<ck::index_t>(arg.gemm_kernel_args_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The group count is not equal to sum of skipped groups "
                             "and kernel args size!"
                          << std::endl;
            }
            return false;
        }

        if(std::is_same_v<EDataType, ck::bhalf_t> && arg.K_BATCH > 1 && !is_bf16_atomic_supported())
        {
            return false;
        }

        bool supported = true;
        for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
        {
            const auto& a = arg.gemm_kernel_args_[i].karg_;

            bool group_arg_valid = GridwiseGemm::CheckValidity(a);
            if(not group_arg_valid)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[" << __func__ << "] group id: " << i
                              << " has invalid GridwiseGemm settings!" << std::endl;
                    a.Print();
                }
            }
            supported = supported && group_arg_valid;
        }
        return supported;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_a_grid,
                             std::vector<const void*>& p_b_grid,
                             std::vector<std::array<const void*, NumDTensor>>&,
                             std::vector<void*>& p_c_grid,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CDEElementwiseOperation)
    {
        return Argument{p_a_grid, p_b_grid, p_c_grid, gemm_descs};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_a_grid,
                        std::vector<const void*>& p_b_grid,
                        std::vector<std::array<const void*, NumDTensor>>&,
                        std::vector<void*>& p_c_grid,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation) override
    {
        return std::make_unique<Argument>(p_a_grid, p_b_grid, p_c_grid, gemm_descs);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGemmXdlUniversal"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(ELayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_)
        {
            return p_arg_->gemm_kernel_args_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffle::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return GetWorkSpaceSize(p_arg);
    }

    // TODO: deperecation notice.
    static void SetKBatchSize(Argument& arg, index_t kbatch) { arg.UpdateKBatch(kbatch); }

    // polymorphic
    void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->UpdateKBatch(kbatch);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffle::Argument structure!");
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return this->SetWorkSpacePointer(p_arg, p_dev_kernel_args);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
