// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

// Add these missing includes:
// #include "ck_tile/core/tensor_layout.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"
// #include "ck_tile/ops/common/element_wise_operation.hpp"

#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp" 

namespace ck_tile {
namespace ops {

template <typename DeviceOp>
struct DeviceOperationInstanceFactory;

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
    virtual std::string GetName() const = 0;
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
    ck_tile::index_t VectorSizeC,
    bool UseSplitK>
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
                                                                OutLayout, // = DsLayout
                                                                OutLayout,
                                                                VectorSizeA,
                                                                VectorSizeB,
                                                                VectorSizeC>;

    using AccDataType = float;
    using DsDataType = OutDataType;
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

    using MemOp = std::conditional_t<UseSplitK,
                                     ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::atomic_add>,
                                     ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::set>>;
    using ConvEpilogue_ = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                InDataType,
                WeiDataType,
                DsDataType,
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
                MemOp{}.value,
                1,
                true,
                GroupedConvTraitsType_::VectorSizeC>>;

    using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                           TilePartitioner_,
                                                                           CodegenPipeline_,
                                                                           ConvEpilogue_>;

    bool IsSupportedArgument(const ck_tile::GroupedConvBwdWeightHostArgs& args) const override
    {
        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
    };

    float Run(const ck_tile::GroupedConvBwdWeightHostArgs& args, bool time_kernel) override
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

    std::string GetName() const override
    {
        return Kernel::GetName();
    };

    ~GroupedConvolutionBackwardWeightInvoker() override = default;
};

template <ck_tile::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<GroupedConvolutionBackwardWeightBaseInvoker<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck_tile::element_wise::PassThrough,
    ck_tile::element_wise::PassThrough,
    ck_tile::element_wise::PassThrough,
    ComputeTypeA,
    ComputeTypeB>>
{
    using DeviceOp = GroupedConvolutionBackwardWeightBaseInvoker<NumDimSpatial,
                                                InLayout,
                                                WeiLayout,
                                                OutLayout,
                                                InDataType,
                                                WeiDataType,
                                                OutDataType,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ck_tile::element_wise::PassThrough,
                                                ComputeTypeA,
                                                ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

//         if constexpr(NumDimSpatial == 2)
//         {
//             if constexpr(is_same_v<InLayout, GNHWC> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, GNHWK>)
//             {
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NHWGC> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, NHWGK>)
//             {
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float>)
//                 {
//                     static_assert(is_same_v<ComputeTypeA, ComputeTypeB>,
//                                   "Error: ComputeTypeA and ComputeTypeB should be the same");
//                     if constexpr(is_same_v<ComputeTypeA, float>)
//                     {
                        
//                     }
//                 }
// #endif
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKCYX> &&
//                          is_same_v<OutLayout, NGKHW>)
//             {
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
//             }
//             if constexpr(is_same_v<InLayout, NGCHW> && is_same_v<WeiLayout, GKYXC> &&
//                          is_same_v<OutLayout, NGKHW>)
//             {
// #ifdef CK_ENABLE_FP16
//                 if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
//                              is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
//                              is_same_v<ComputeTypeB, half_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_BF16
//                 if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
//                              is_same_v<WeiDataType, ck::bhalf_t> &&
//                              is_same_v<OutDataType, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeA, ck::bhalf_t> &&
//                              is_same_v<ComputeTypeB, ck::bhalf_t>)
//                 {
//                 }
// #endif
// #ifdef CK_ENABLE_FP32
//                 if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
//                              is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
//                              is_same_v<ComputeTypeB, float>)
//                 {
//                 }
// #endif
//             }
//         }

        return op_ptrs;
    }
};

} // namespace ops
} // namespace ck_tile
