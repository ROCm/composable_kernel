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
struct GroupedConvolutionBackwardWeightBaseInvoker
{
    virtual bool IsSupportedArgument(const ck_tile::GroupedConvBwdWeightHostArgs& args) const = 0; 
    virtual float Run(const ck_tile::GroupedConvBwdWeightHostArgs& args, bool time_kernel) = 0;
    virtual std::string GetName(const ck_tile::GroupedConvBwdWeightHostArgs& args) const = 0;
    GroupedConvolutionBackwardWeightBaseInvoker() = default;
    GroupedConvolutionBackwardWeightBaseInvoker(const GroupedConvolutionBackwardWeightBaseInvoker&) = default;
    GroupedConvolutionBackwardWeightBaseInvoker& operator=(const GroupedConvolutionBackwardWeightBaseInvoker&) = default;
    GroupedConvolutionBackwardWeightBaseInvoker(GroupedConvolutionBackwardWeightBaseInvoker&&) = default;
    GroupedConvolutionBackwardWeightBaseInvoker& operator=(GroupedConvolutionBackwardWeightBaseInvoker&&) = default;
    virtual ~GroupedConvolutionBackwardWeightBaseInvoker() = default;
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
    ck_tile::index_t VectorSizeC>
struct GroupedConvolutionBackwardWeightInvoker : 
    public GroupedConvolutionBackwardWeightBaseInvoker<NDimSpatial,
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
    using CodegenShape_ =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    static constexpr auto ConvSpec_  = ck_tile::ConvolutionSpecialization::Default;

    using TilePartitioner_           = ck_tile::GemmTile1DPartitioner<CodegenShape_>;
    using GroupedConvTraitsType_     = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                ConvSpec_,
                                                                InLayout,
                                                                WeiLayout,
                                                                ck_tile::tuple<>, // = DsLayout
                                                                OutLayout,
                                                                VectorSizeA,
                                                                VectorSizeB,
                                                                VectorSizeC>;

    using AccDataType = float;
    using CDEElementWise = ck_tile::element_wise::PassThrough;

    using CodegenPipelineProblem_ = ck_tile::GemmPipelineProblem<
        InDataType,
        WeiDataType,
        AccDataType,
        CodegenShape_,
        typename GroupedConvTraitsType_::GroupedConvImplicitGemmTraitsBwdWeight,
        ck_tile::element_wise::PassThrough,
        ck_tile::element_wise::PassThrough,
        InDataType,
        true,
        GroupedConvTraitsType_::VectorSizeA,
        GroupedConvTraitsType_::VectorSizeB>;

    using CodegenPipeline_ = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem_>;

    using ConvEpilogueAtomicAdd_ = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                InDataType,
                WeiDataType,
                ck_tile::tuple<>, // = DsDataType,
                AccDataType,
                OutDataType,
                typename GroupedConvTraitsType_::ImplicitGemmDsLayout,
                ck_tile::tensor_layout::gemm::RowMajor,
                CDEElementWise,
                TilePartitioner_::MPerBlock,
                TilePartitioner_::NPerBlock,
                M_Warp,
                N_Warp,
                M_Warp_Tile,
                N_Warp_Tile,
                K_Warp_Tile,
                CodegenPipelineProblem_::TransposeC,
                ck_tile::memory_operation_enum::atomic_add,
                1,
                true,
                GroupedConvTraitsType_::VectorSizeC>>;

    using ConvEpilogueSet_ = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                InDataType,
                WeiDataType,
                ck_tile::tuple<>, // = DsDataType,
                AccDataType,
                OutDataType,
                typename GroupedConvTraitsType_::ImplicitGemmDsLayout,
                ck_tile::tensor_layout::gemm::RowMajor,
                CDEElementWise,
                TilePartitioner_::MPerBlock,
                TilePartitioner_::NPerBlock,
                M_Warp,
                N_Warp,
                M_Warp_Tile,
                N_Warp_Tile,
                K_Warp_Tile,
                CodegenPipelineProblem_::TransposeC,
                ck_tile::memory_operation_enum::set,
                1,
                true,
                GroupedConvTraitsType_::VectorSizeC>>;

    using KernelSplitK = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                           TilePartitioner_,
                                                                           CodegenPipeline_,
                                                                           ConvEpilogueAtomicAdd_>;

    using KernelNonSplitK = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                           TilePartitioner_,
                                                                           CodegenPipeline_,
                                                                           ConvEpilogueSet_>;

    bool IsSupportedArgument(const ck_tile::GroupedConvBwdWeightHostArgs& args) const override
    {
        if (args.k_batch == 1)
        {
            return KernelNonSplitK::IsSupportedArgument(KernelNonSplitK::MakeKernelArgs(args));
        }
        return KernelSplitK::IsSupportedArgument(KernelSplitK::MakeKernelArgs(args));
    };

    template <typename Kernel>
    float RunImpl(const ck_tile::GroupedConvBwdWeightHostArgs& args, bool time_kernel)
    {
        auto kargs        = Kernel::MakeKernelArgs(args);
        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        constexpr int n_warmup = 5;
        constexpr int n_repeat = 50;
        ck_tile::stream_config s {nullptr, time_kernel, 1, n_warmup, n_repeat};
        float avg_time = ck_tile::launch_kernel_time_mask(
            s,
            Kernel::Preprocess(kargs, s),
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return avg_time;
    };

    float Run(const ck_tile::GroupedConvBwdWeightHostArgs& args, bool time_kernel) override
    {
        if (args.k_batch == 1)
        {
            return RunImpl<KernelNonSplitK>(args, time_kernel);
        }
        else
        {
            return RunImpl<KernelSplitK>(args, time_kernel);
        }
    };

    std::string GetName(const ck_tile::GroupedConvBwdWeightHostArgs& args) const override
    {
        std::stringstream min_occupancy;
        min_occupancy << "_blk_per_cu_" << kBlockPerCu;
        if (args.k_batch == 1)
        {
            return KernelNonSplitK::GetName() + min_occupancy.str();
        }
        return KernelSplitK::GetName() + min_occupancy.str();
    };

    GroupedConvolutionBackwardWeightInvoker() = default;
    GroupedConvolutionBackwardWeightInvoker(const GroupedConvolutionBackwardWeightInvoker&) = default;
    GroupedConvolutionBackwardWeightInvoker& operator=(const GroupedConvolutionBackwardWeightInvoker&) = default;
    GroupedConvolutionBackwardWeightInvoker(GroupedConvolutionBackwardWeightInvoker&&) = default;
    GroupedConvolutionBackwardWeightInvoker& operator=(GroupedConvolutionBackwardWeightInvoker&&) = default;
    ~GroupedConvolutionBackwardWeightInvoker() override = default;
  };

}
}
