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
          typename AElementwiseOperation   = ck::tensor_operation::element_wise::PassThrough,
          typename BElementwiseOperation   = ck::tensor_operation::element_wise::PassThrough,
          typename CDEElementwiseOperation = ck::tensor_operation::element_wise::PassThrough>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_xdl_splitk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                       const index_t group_count,
                                       const AElementwiseOperation a_element_op,
                                       const BElementwiseOperation b_element_op,
                                       const CDEElementwiseOperation c_element_op)
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

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
        gemm_desc_ptr[group_id].karg_,
        static_cast<void*>(p_shared),
        gemm_desc_ptr[group_id].block_2_ctile_map_,
        a_element_op,
        b_element_op,
        c_element_op);
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
          bool PermuteB                               = false>

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
        ComputeTypeB>;

    using Block2ETileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;

    using GroupedGemmBlock2ETileMap = OffsettedBlockToCTileMap<Block2ETileMap>;
    using KernelArgument            = typename GridwiseGemm::Argument;
    using PassThrough               = ck::tensor_operation::element_wise::PassThrough;

    struct GemmTransKernelArg
    {
        KernelArgument karg_;
        // GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(KernelArgument&& karg,
                           //    GroupedGemmBlock2ETileMap&& b2c_map,
                           index_t block_start,
                           index_t block_end)
            : karg_{karg},
              //   block_2_ctile_map_{b2c_map},
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

                if(M == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;
                const index_t stride_c = gemm_descs[i].stride_C_;

                index_t gdx, gdy, gdz;
                std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(M, N, K_BATCH);

                const auto local_b2c_tile_map = Block2ETileMap{gdx, gdy, gdz};
                const index_t grid_size_grp   = local_b2c_tile_map.CalculateGridSize(M, N);
                // const index_t grid_size_grp = gdx * gdy * gdz;

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                // auto grouped_block_2_ctile_map =
                //     GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

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

                // gemm_kernel_args_.emplace_back(
                // std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);

                gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
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

                // const index_t m_padded = GridwiseGemm::CalculateMPadded(karg.M);
                // const index_t n_padded = GridwiseGemm::CalculateNPadded(karg.N);

                index_t gdx, gdy, gdz;
                std::tie(gdx, gdy, gdz) =
                    GridwiseGemm::CalculateGridSize(karg.M, karg.N, karg.KBatch);

                const auto local_b2c_tile_map = Block2ETileMap{gdx, gdy, gdz};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(karg.M, karg.N);
                // const index_t grid_size_grp = gdx * gdy * gdz;

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;
                // auto grouped_block_2_ctile_map =
                //     GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                karg.KBatch = K_BATCH;
                // gemm_kernel_args_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_kernel_args_[i].block_start_ = block_start;
                gemm_kernel_args_[i].block_end_   = block_end;
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
            bool all_have_main_k_block_loop{true};
            bool all_have_kbatch_gt_one;

            for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& karg = arg.gemm_kernel_args_[i].karg_;

                all_have_kbatch_gt_one = karg.KBatch > 1;

                index_t k_grain = arg.gemm_kernel_args_[i].karg_.KBatch * KPerBlock;
                index_t K_split =
                    (arg.gemm_kernel_args_[i].karg_.K + k_grain - 1) / k_grain * KPerBlock;

                all_have_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

                if(stream_config.log_level_ > 0)
                {
                    karg.Print();
                }

                auto kbatch = karg.KBatch;

                if(!GridwiseGemm::CheckValidity(karg))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                bool not_all_have_kbatch_value_same = all_have_kbatch_gt_one xor (kbatch > 1);

                if(not_all_have_kbatch_value_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same kbatch value (=1 or >1)! "
                        << "group [" << i << "], kbatch: " << kbatch
                        << ", group [0], kbatch: " << arg.gemm_kernel_args_[0].karg_.KBatch
                        << " in " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
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
                if(all_have_kbatch_gt_one)
                {
                    for(const auto& trans_arg : arg.gemm_kernel_args_)
                    {
                        const auto& karg = trans_arg.karg_;
                        hip_check_error(hipMemsetAsync(karg.p_c_grid,
                                                       0,
                                                       karg.M * karg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));
                    }
                }

                for(const auto& trans_arg : arg.gemm_kernel_args_)
                {
                    const auto& karg = trans_arg.karg_;
                    ave_time += launch_and_time_kernel(
                        stream_config, kernel, dim3(arg.grid_size_), dim3(BlockSize), 0, karg);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            // Calculate TailNumber for one
            auto calculate_tail_number = [&]() {
                index_t k_grain = arg.gemm_kernel_args_[0].karg_.KBatch * KPerBlock;
                index_t K_split =
                    (arg.gemm_kernel_args_[0].karg_.K + k_grain - 1) / k_grain * KPerBlock;
                return GridwiseGemm::CalculateKBlockLoopTailNum(K_split);
            };

            auto all_have_same_tail_number = [&]() {
                // Calculate TailNumber for one
                auto tail_number = calculate_tail_number();

                // Calculate TailNumber for every other arg and compare
                for(size_t i = 1; i < arg.gemm_kernel_args_.size(); ++i)
                {

                    index_t k_grain = arg.gemm_kernel_args_[i].karg_.KBatch * KPerBlock;
                    index_t K_split =
                        (arg.gemm_kernel_args_[i].karg_.K + k_grain - 1) / k_grain * KPerBlock;

                    if(tail_number != GridwiseGemm::CalculateKBlockLoopTailNum(K_split))
                    {
                        return false;
                    }
                }

                return true;
            };

            auto throw_error = [&]() {
                std::ostringstream err;
                err << "Not all gemms have same TailNumber value! ";
                throw std::runtime_error(err.str());
            };

            if(all_have_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::AtomicAdd,
                                                        minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
                        if(calculate_tail_number() == TailNumber::One)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::One>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }

                        else if(calculate_tail_number() == TailNumber::Full)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Full>;
                                Run(kernel);
                            }

                            else
                            {
                                throw_error();
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(calculate_tail_number() == TailNumber::Two)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Two>;
                                    Run(kernel);
                                }

                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(calculate_tail_number() == TailNumber::Three)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Three>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(calculate_tail_number() == TailNumber::Four)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Four>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(calculate_tail_number() == TailNumber::Five)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Five>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(calculate_tail_number() == TailNumber::Six)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Six>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(calculate_tail_number() == TailNumber::Seven)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                        GridwiseGemm,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Seven>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }
                    }

                    else
                    {
                        if(calculate_tail_number() == TailNumber::One)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::One>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }
                        else if(calculate_tail_number() == TailNumber::Full)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Full>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(calculate_tail_number() == TailNumber::Two)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Two>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(calculate_tail_number() == TailNumber::Three)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Three>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(calculate_tail_number() == TailNumber::Four)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Four>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(calculate_tail_number() == TailNumber::Five)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Five>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(calculate_tail_number() == TailNumber::Six)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Six>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(calculate_tail_number() == TailNumber::Seven)
                            {
                                if(all_have_same_tail_number())
                                {
                                    const auto kernel =
                                        kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Seven>;
                                    Run(kernel);
                                }
                                else
                                {
                                    throw_error();
                                }
                            }
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        if(calculate_tail_number() == TailNumber::Odd)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }
                        else
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3_2lds<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }
                    }
                    else
                    {
                        if(calculate_tail_number() == TailNumber::Odd)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                                     true,
                                                                     InMemoryDataOperationEnum::Set,
                                                                     minimum_occupancy,
                                                                     TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }

                        else
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3_2lds<GridwiseGemm,
                                                                     true,
                                                                     InMemoryDataOperationEnum::Set,
                                                                     minimum_occupancy,
                                                                     TailNumber::Even>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }
                    }
                }

                else
                {
                    if(all_have_kbatch_gt_one)
                    {
                        if(calculate_tail_number() == TailNumber::Odd)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }

                        else
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel = kernel_gemm_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }
                    }
                    else
                    {
                        if(calculate_tail_number() == TailNumber::Odd)
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Odd>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
                        }

                        else
                        {
                            if(all_have_same_tail_number())
                            {
                                const auto kernel =
                                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy,
                                                                TailNumber::Even>;
                                Run(kernel);
                            }
                            else
                            {
                                throw_error();
                            }
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
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        false,
                                                        InMemoryDataOperationEnum::AtomicAdd,
                                                        minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_gemm_xdl_cshuffle_v3<GridwiseGemm,
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
