// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp" 
#include "ck_tile/library/tensor_operation_instance/gpu/gemm_configs.hpp"

namespace ck_tile {
namespace ops {

template <ck_tile::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct GroupedConvolutionForwardBaseInvoker
{
    virtual bool IsSupportedArgument(const ck_tile::GroupedConvFwdHostArgs& args) const = 0; 
    virtual float Run(const ck_tile::GroupedConvFwdHostArgs& args, bool time_kernel, int n_warmup, int n_repeat) const = 0;
    virtual std::string GetName(const ck_tile::GroupedConvFwdHostArgs& args) const = 0;
    GroupedConvolutionForwardBaseInvoker() = default;
    GroupedConvolutionForwardBaseInvoker(const GroupedConvolutionForwardBaseInvoker&) = default;
    GroupedConvolutionForwardBaseInvoker& operator=(const GroupedConvolutionForwardBaseInvoker&) = default;
    GroupedConvolutionForwardBaseInvoker(GroupedConvolutionForwardBaseInvoker&&) = default;
    GroupedConvolutionForwardBaseInvoker& operator=(GroupedConvolutionForwardBaseInvoker&&) = default;
    virtual ~GroupedConvolutionForwardBaseInvoker() = default;
};

template <
    ck_tile::index_t NDimSpatial,
    typename InLayout,
    typename WeiLayout,
    typename OutLayout,
    typename InDataType,
    typename WeiDataType,
    typename OutDataType,
    typename InElementwiseOperation,
    typename WeiElementwiseOperation,
    typename OutElementwiseOperation,
    int kBlockPerCu,
    ck_tile::index_t M_Tile,
    ck_tile::index_t N_Tile,
    ck_tile::index_t K_Tile,
    ck_tile::index_t M_Warp,
    ck_tile::index_t N_Warp,
    ck_tile::index_t K_Warp,
    ck_tile::index_t M_Warp_Tile,
    ck_tile::index_t N_Warp_Tile,
    ck_tile::index_t K_Warp_Tile,
    ck_tile::index_t VectorSizeA,
    ck_tile::index_t VectorSizeB,
    ck_tile::index_t VectorSizeC,
    bool DoubleSmemBuffer,
    ck_tile::index_t PipelineVersion>
struct GroupedConvolutionForwardInvoker : 
    public GroupedConvolutionForwardBaseInvoker<NDimSpatial,
                                                        InLayout,
                                                        WeiLayout,
                                                        OutLayout,
                                                        InDataType,
                                                        WeiDataType,
                                                        OutDataType,
                                                        InElementwiseOperation,
                                                        WeiElementwiseOperation,
                                                        OutElementwiseOperation>
{
    using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
            ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
            ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>,
            GemmConfigBase::PermuteA,
            GemmConfigBase::PermuteB>;

    static constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;

    using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfigBase::TileParitionerGroupNum,
                                                       GemmConfigBase::TileParitionerM01>;

    using GroupedConvTraitsType = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                 ConvSpec,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 ck_tile::tuple<>, // = DsLayout
                                                                 OutLayout,
                                                                 VectorSizeA,
                                                                 VectorSizeB,
                                                                 VectorSizeC>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GemmConfigBase::kPadM,
            GemmConfigBase::kPadN,
            GemmConfigBase::kPadK,
            DoubleSmemBuffer,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::AsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::BsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::CLayout,
            GemmConfigBase::TransposeC,
            GemmConfigBase::UseStructuredSparsity,
            false, // Persistent,
            GemmConfigBase::NumWaveGroups,
            GemmConfigBase::Preshuffle>;

    using AccDataType = float;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            true,
            VectorSizeA,
            VectorSizeB>;

    using BaseGemmPipeline = typename PipelineTypeTraits<PipelineVersion>::template UniversalGemmPipeline<GemmPipelineProblem>;
    
    template <bool HasHotLoop, ck_tile::TailNumber TailNumber, ck_tile::memory_operation_enum MemOp>
    auto CreateKernel() const
    {
        constexpr auto scheduler = GemmConfigBase::Scheduler;
    
        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<InDataType,
                                                WeiDataType,
                                                AccDataType,
                                                GemmShape,
                                                GemmUniversalTraits,
                                                scheduler,
                                                HasHotLoop,
                                                TailNumber,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                OutDataType,
                                                true,
                                                VectorSizeA,
                                                VectorSizeB>;

        using GemmPipeline = typename PipelineTypeTraits<PipelineVersion>::template GemmPipeline<UniversalGemmProblem>;

        using CDEElementWise = ck_tile::element_wise::PassThrough;

        using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            InDataType,
            WeiDataType,
            ck_tile::tuple<>, // = DsDataType
            AccDataType,
            OutDataType,
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            ck_tile::tensor_layout::gemm::RowMajor,
            CDEElementWise,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            M_Warp,
            N_Warp,
            M_Warp_Tile,
            N_Warp_Tile,
            K_Warp_Tile,
            GemmConfigBase::TransposeC,
            MemOp,
            1,
            true,
            GroupedConvTraitsType::VectorSizeC>>;

        return ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                            TilePartitioner,
                                                            GemmPipeline,
                                                            ConvEpilogue>{};   
    }

    bool IsSupportedArgument(const ck_tile::GroupedConvFwdHostArgs& args) const override
    {
        if (args.k_batch > 1)
        {
            using Kernel = decltype(CreateKernel<false, ck_tile::TailNumber::Empty, ck_tile::memory_operation_enum::atomic_add>());
            return Kernel::IsSupportedArgument(args);    
        }
        using Kernel = decltype(CreateKernel<false, ck_tile::TailNumber::Empty, ck_tile::memory_operation_enum::set>());
        return Kernel::IsSupportedArgument(args);
    };

    float Run(const ck_tile::GroupedConvFwdHostArgs& args, bool time_kernel, int n_warmup=5, int n_repeat=50) const override
    {
        const ck_tile::index_t gemm_k =
            args.C_ * std::accumulate(args.filter_spatial_lengths_.begin(),
                                      args.filter_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        const auto Run = [&](const auto has_hot_loop_,
                             const auto tail_number_,
                             const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto memory_operation = memory_operation_.value;

            auto kernel = CreateKernel<has_hot_loop_v, tail_number_v, memory_operation>();
            using Kernel = decltype(kernel);
                             
            auto kargs   = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args);
            const dim3 blocks = Kernel::BlockSize();
                   
            ck_tile::stream_config s {nullptr, time_kernel, 1, n_warmup, n_repeat};

            ave_time = ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(kernel, grids, blocks, 0, kargs));

            return ave_time;
        };

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(args.k_batch == 1)
            {
                Run(has_hot_loop_, tail_number_, MemoryOpSet{});
            }
            else
            {
                Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
            }
        };

        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    };

    std::string GetName(const ck_tile::GroupedConvFwdHostArgs& args) const override
    {
        std::stringstream min_occupancy;
        min_occupancy << "_blk_per_cu_" << kBlockPerCu;
        if (args.k_batch > 1)
        {
            using Kernel = decltype(CreateKernel<false, ck_tile::TailNumber::Empty, ck_tile::memory_operation_enum::atomic_add>());
            return Kernel::GetName() + min_occupancy.str();    
        }
        using Kernel = decltype(CreateKernel<false, ck_tile::TailNumber::Empty, ck_tile::memory_operation_enum::set>());
        return Kernel::GetName() + min_occupancy.str();
    };

    GroupedConvolutionForwardInvoker() = default;
    GroupedConvolutionForwardInvoker(const GroupedConvolutionForwardInvoker&) = default;
    GroupedConvolutionForwardInvoker& operator=(const GroupedConvolutionForwardInvoker&) = default;
    GroupedConvolutionForwardInvoker(GroupedConvolutionForwardInvoker&&) = default;
    GroupedConvolutionForwardInvoker& operator=(GroupedConvolutionForwardInvoker&&) = default;
    ~GroupedConvolutionForwardInvoker() override = default;
  };

}
}
