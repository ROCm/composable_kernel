// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2CTileMap,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_gemm_wmma_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                      const index_t group_count)
{
#if(defined(__gfx11__) || defined(__gfx12__))
    constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
        typename GridwiseGemm::EpilogueCShuffle>();
    __shared__ char p_shared[LDS_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));
        
    // Binary search lookup to find which group this block is part of
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

    // NOTE: Local copy of the arg struct since SplitKBatchOffset verifies and modifies K index
    // and thus needs a non-const reference. It's also not feasible to store this in global
    // memory as different threads would be writing different K values to the same arg struct
    auto karg = gemm_desc_ptr[group_id].karg_;

#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using c_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<c_data_type, ck::half_t> ||
                    std::is_same_v<c_data_type, ck::bhalf_t>)))
    {
#endif
        const auto& block_2_ctile_map = gemm_desc_ptr[group_id].block_2_ctile_map_;

        // Tile index first dimension is the K batch
        auto tile_index =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        auto splitk_batch_offset =
            typename GridwiseGemm::SplitKBatchOffset(karg, tile_index[Number<0>{}]);
        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop,
                                   CGlobalMemoryDataOperation,
                                   TailNum,
                                   Block2CTileMap,
                                   typename GridwiseGemm::EpilogueCShuffle,
                                   1, // Block2CTileMap MBlock index
                                   2  // Block2CTileMap NBlock index
                                   >(static_cast<void*>(p_shared),
                                     splitk_batch_offset,
                                     karg,
                                     block_2_ctile_map,
                                     epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = gemm_descs_const;
    ignore = group_count;
#endif // end of if(defined(__gfx11__) || defined(__gfx12__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGroupedGemm_Wmma_Fixed_Nk : public DeviceGroupedGemmFixedNK<ALayout,
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
    using DeviceOp = DeviceGroupedGemm_Wmma_Fixed_Nk;

    static constexpr index_t NumDTensor = DsDataType::Size();

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        DsDataType,
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
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    using CGridDesc_M_N =
        remove_cvref_t<decltype(GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
            1, 1, 1, 1, 1))>;

    using Block2ETileMap = BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;
    static constexpr index_t B2E_M01 = 8;
    using GroupedGemmBlock2ETileMap = OffsettedBlockToCTileMap<Block2ETileMap>;

    static constexpr index_t DefaultKBatch = 1;
    using KernelArgument                   = typename GridwiseGemm::Argument;

    template <typename KernelArgument_>
    struct GemmTransKernelArgBase
    {
        KernelArgument_ karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArgBase() = default;
        GemmTransKernelArgBase(KernelArgument_&& karg,
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
    using GemmTransKernelArg = GemmTransKernelArgBase<KernelArgument>;

    static constexpr bool CalculateHasMainKBlockLoop(const KernelArgument& karg)
    {
        index_t k_grain = karg.KBatch * KPerBlock;
        index_t K_split = (karg.K + k_grain - 1) / karg.KBatch;
        return GridwiseGemm::CalculateHasMainKBlockLoop(K_split);
    }

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op)
            : Argument(p_As,
                       p_Bs,
                       p_Ds,
                       p_Es,
                       gemm_descs,
                       a_element_op,
                       b_element_op,
                       c_element_op,
                       DefaultKBatch)
        {
            // TODO: use occupancy api to calculate appropriate batch size.
        }

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op,
                 index_t kbatch)
            : group_count_{ck::type_convert<ck::index_t>(gemm_descs.size())},
              gemm_kernel_host_args_{nullptr},
              grid_size_{0},
              k_batch_{kbatch}
        {
            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 ((NumDTensor == 0 && p_Ds.size() == 0) ||
                  group_count_ == ck::type_convert<ck::index_t>(p_Ds.size())) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/d/e.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            const index_t fixed_N = gemm_descs[0].N_;
            const index_t fixed_K = gemm_descs[0].K_;

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(N != fixed_N || K != fixed_K)
                {
                    throw std::runtime_error("wrong! N or K are not fixed across GEMM groups");
                }

                const index_t StrideA = gemm_descs[i].stride_A_;
                const index_t StrideB = gemm_descs[i].stride_B_;
                const index_t StrideE = gemm_descs[i].stride_C_;
                const auto& stride_d_vec = gemm_descs[i].stride_Ds_;

                if(!(NumDTensor == ck::type_convert<ck::index_t>(stride_d_vec.size())))
                {
                    throw std::runtime_error("wrong! stride D mismatch");
                }

                std::array<index_t, NumDTensor> StrideDs;
                if constexpr(NumDTensor > 0)
                {
                    std::copy(stride_d_vec.begin(), stride_d_vec.end(), StrideDs);
                }

                const index_t m_padded = GridwiseGemm::CalculateMPadded(M);
                const index_t n_padded = GridwiseGemm::CalculateNPadded(N);

                const auto e_grid_desc_m_n =
                    GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                        M, m_padded, N, n_padded, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, B2E_M01, k_batch_};

                const index_t grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp_;

                grid_size_ += grid_size_grp_;

                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                auto karg = KernelArgument(std::array<const void*, 1>{p_As[i]},
                                           std::array<const void*, 1>{p_Bs[i]},
                                           p_Ds[i],
                                           type_convert<EDataType*>(p_Es[i]),
                                           M,
                                           N,
                                           K,
                                           std::array<index_t, 1>{StrideA},
                                           std::array<index_t, 1>{StrideB},
                                           StrideDs,
                                           StrideE,
                                           k_batch_,
                                           a_element_op,
                                           b_element_op,
                                           c_element_op,
                                           false);

                gemm_desc_kernel_arg_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);
            }
        }

        /**
         * @brief      Recalculate group grid size for all gemms and update B2C maps.
         *
         * @param[in]  k_batch  The new splitK parameter value.
         */
        void UpdateKBatch(index_t k_batch)
        {
            k_batch_ = k_batch;
            grid_size_ = 0;

            if(k_batch_ < 1)
            {
                throw std::runtime_error("wrong! k_batch must be > 0");
            }

            for(std::size_t i = 0; i < gemm_desc_kernel_arg_.size(); ++i)
            {
                auto& karg = gemm_desc_kernel_arg_[i].karg_;

                const index_t k_read     = GridwiseGemm::CalculateKRead(karg.K, k_batch_);
                const index_t k_padded   = GridwiseGemm::CalculateKPadded(karg.K, k_batch_);
                const index_t ak0_padded = GridwiseGemm::CalculateAK0Padded(karg.K, k_batch_);
                const index_t bk0_padded = GridwiseGemm::CalculateBK0Padded(karg.K, k_batch_);

                const auto c_grid_desc_m_n =
                    GridwiseGemm::template MakeDEGridDescriptor_M_N<ELayout>(
                        karg.M, karg.MPadded, karg.N, karg.NPadded, karg.StrideE);

                const auto local_b2c_tile_map = Block2ETileMap{c_grid_desc_m_n, B2E_M01, k_batch_};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                karg.KRead                                  = k_read;
                karg.KPadded                                = k_padded;
                karg.AK0                                    = ak0_padded;
                karg.BK0                                    = bk0_padded;
                karg.KBatch                                 = k_batch_;
                gemm_desc_kernel_arg_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_desc_kernel_arg_[i].block_start_       = block_start;
                gemm_desc_kernel_arg_[i].block_end_         = block_end;
            }


        }

        //  private:
        index_t group_count_;

        std::vector<GemmTransKernelArg> gemm_desc_kernel_arg_;

        void* gemm_kernel_host_args_;
        index_t grid_size_;

        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg,
                  const StreamConfig& stream_config = StreamConfig{},
                  hipStream_t cpy_stream            = nullptr,
                  hipEvent_t cpy_event              = nullptr)
        {
            using GemmTransKernelArg_ = GemmTransKernelArgBase<typename GridwiseGemm::Argument>;
            static_assert(sizeof(GemmTransKernelArg_) == sizeof(GemmTransKernelArg));

            bool all_have_kbatch_gt_one = arg.gemm_desc_kernel_arg_[0].karg_.KBatch > 1;
            bool all_have_main_k0_block_loop =
                CalculateHasMainKBlockLoop(arg.gemm_desc_kernel_arg_[0].karg_);

            bool not_all_have_main_k0_block_loop_same = false;
            bool not_all_have_kbatch_value_same       = false;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); ++i)
            {
                const auto& karg = reinterpret_cast<const typename GridwiseGemm::Argument&>(
                    arg.gemm_desc_kernel_arg_[i].karg_);

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

                not_all_have_main_k0_block_loop_same |=
                    all_have_main_k0_block_loop xor CalculateHasMainKBlockLoop(karg);
                not_all_have_kbatch_value_same |= all_have_kbatch_gt_one xor (kbatch > 1);
            }

            if(not_all_have_main_k0_block_loop_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same value for main_k0_block_loop! in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                // throw std::runtime_error(err.str());
            }

            if(not_all_have_kbatch_value_same)
            {
                std::ostringstream err;
                err << "Not all gemms have same kbatch value (=1 or >1)! " << " in " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            // If the user provides copy stream and copy event, we assume that they're also
            // responsible for providing allocated host memory (eg. pinned) which
            // would be used to copy kernel arguments to the device.
            if(cpy_stream && cpy_event)
            {
                if(arg.gemm_kernel_host_args_ == nullptr)
                {
                    std::ostringstream err;
                    err << "No memory has been allocated for gemm kernel host args "
                        << "when providing the copy stream and copy event! In " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
                hip_check_error(hipMemcpyAsync(arg.p_workspace_,
                                               arg.gemm_kernel_host_args_,
                                               arg.group_count_ * sizeof(GemmTransKernelArg_),
                                               hipMemcpyHostToDevice,
                                               cpy_stream));

                hip_check_error(hipEventRecord(cpy_event, cpy_stream));

                hip_check_error(hipEventSynchronize(cpy_event));
            }
            else // In this case CK owns memory allocated on host.
            {

                hip_check_error(
                    hipMemcpyAsync(arg.p_workspace_,
                                   arg.gemm_desc_kernel_arg_.data(),
                                   arg.gemm_desc_kernel_arg_.size() * sizeof(GemmTransKernelArg_),
                                   hipMemcpyHostToDevice,
                                   stream_config.stream_id_));
            }

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(all_have_kbatch_gt_one)
                {
                    for(const auto& trans_arg : arg.gemm_desc_kernel_arg_)
                    {

                        const auto& karg = trans_arg.karg_;
                        hip_check_error(hipMemsetAsync(karg.p_e_grid,
                                                       0,
                                                       karg.M * karg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));
                    }
                }

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(arg.grid_size_),
                                           dim3(BlockSize),
                                           0,
                                           cast_pointer_to_constant_address_space(arg.p_workspace_),
                                           arg.gemm_desc_kernel_arg_.size());
            };

            // NOTE: If at least one gemm problem has a main k0 block loop, we include it for all
            if(all_have_main_k0_block_loop || not_all_have_main_k0_block_loop_same)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(all_have_kbatch_gt_one)
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              true,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              true,
                                                              InMemoryDataOperationEnum::Set,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
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
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              false,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              GroupedGemmBlock2ETileMap>;

                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_gemm_wmma_fixed_nk<GridwiseGemm,
                                                              GemmTransKernelArg_,
                                                              false,
                                                              InMemoryDataOperationEnum::Set,
                                                              GroupedGemmBlock2ETileMap>;

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

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }
        if constexpr(std::is_same_v<EDataType, ck::half_t> ||
                     std::is_same_v<EDataType, ck::bhalf_t>)
        {
            if(arg.k_batch_ > 1 && ck::is_gfx11_supported())
            {
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(!std::is_same_v<CDEElementwiseOperation,
                                     ck::tensor_operation::element_wise::PassThrough>)
        {
            if(arg.k_batch_ > 1)
            {
                // Using SplitK and a C element op would require a two stage kernel where the second
                // stage applies the op on the accumulated results
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "C element operators are not supported when using SplitK. Set "
                                 "K_BATCH to 1 or remove the operator."
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        bool supported = true;
        for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); ++i)
        {
            const auto& a        = arg.gemm_desc_kernel_arg_[i].karg_;
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

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation c_element_op)
    {
        return Argument{
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op);
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

        // clang-format off
        str << "DeviceGroupedGemm_Wmma_Fixed_Nk"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }

    // polymorphic
    void SetDeviceKernelArgs(BaseArgument* p_arg, void* kernel_args) const override
    {
        return this->SetWorkSpacePointer(p_arg, kernel_args);
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_)
        {
            return p_arg_->gemm_desc_kernel_arg_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_CShuffleV3::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return GetWorkSpaceSize(p_arg);
    }

    size_t GetHostKernelArgSize(const BaseArgument* p_arg) const { return GetWorkSpaceSize(p_arg); }

    static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatchSize(BaseArgument* p_arg, index_t k_batch) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->UpdateKBatch(k_batch);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Wmma_Fixed_Nk::Argument structure!");
    }


    void SetHostKernelArgsPointer(BaseArgument* p_arg, void* p_host_kernel_args) const
    {
        Argument* pArg_ = dynamic_cast<Argument*>(p_arg);
        if(!pArg_)
        {
            throw std::runtime_error("Failed to cast argument pointer!");
        }

        pArg_->gemm_kernel_host_args_ = p_host_kernel_args;
        std::copy(pArg_->gemm_desc_kernel_arg_.begin(),
                  pArg_->gemm_desc_kernel_arg_.end(),
                  static_cast<GemmTransKernelArg*>(pArg_->gemm_kernel_host_args_));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
