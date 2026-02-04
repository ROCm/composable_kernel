// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/tuple.hpp"
#include "convnd_fwd_convscale_add_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

using InDataType       = ck::f8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;
using DsDataType       = float;
using OutDataType      = ck::f8_t;
using AComputeDataType = ck::f8_t;
using BComputeDataType = ck::f8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ConvScaleAdd;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DsLayout,
          typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
        NDimSpatial,           // NDimSpatial
        InLayout,              // ALayout
        WeiLayout,             // BLayout
        ck::Tuple<DsLayout>,   // DsLayout
        OutLayout,             // ELayout
        InDataType,            // ADataType
        WeiDataType,           // BDataType
        AccDataType,           // AccDataType
        CShuffleDataType,      // CShuffleDataType
        ck::Tuple<DsDataType>, // DsDataType
        OutDataType,           // EDataType
        InElementOp,           // AElementwiseOperation
        WeiElementOp,          // BElementwiseOperation
        OutElementOp,          // CDEElementwiseOperation
        ConvSpec,              // ConvForwardSpecialization
        GemmSpec,              // GemmSpecialization
        64,                    // BlockSize
        64,                    // MPerBlock
        64,                    // NPerBlock
        32,                    // KPerBlock
        8,                     // AK1
        8,                     // BK1
        16,                    // MPerWmma
        16,                    // NPerWmma
        4,                     // MRepeat
        2,                     // NRepeat
        S<4, 16, 1>,           // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,            // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,            // ABlockTransferSrcAccessOrder
        2,                     // ABlockTransferSrcVectorDim
        1,                     // ABlockTransferSrcScalarPerVector
        8,                     // ABlockTransferDstScalarPerVector_AK1
        1,                     // ABlockLdsExtraM
        S<4, 16, 1>,           // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,            // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,            // BBlockTransferSrcAccessOrder
        2,                     // BBlockTransferSrcVectorDim
        1,                     // BBlockTransferSrcScalarPerVector
        8,                     // BBlockTransferDstScalarPerVector_BK1
        1,                     // BBlockLdsExtraN
        1,                     // CShuffleMRepeatPerShuffle
        1,                     // CShuffleNRepeatPerShuffle
        S<1, 16, 1, 4>,        // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        1,                     // CDEBlockTransferScalarPerVector_NPerBlock
        ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
        true,                                      // UseThreadTileTransfer
        AComputeDataType,                          // AComputeDataType
        BComputeDataType,                          // BComputeDataType
        1>;                                        // NumGroupsToMerge

#include "run_convnd_fwd_convscale_add_example.inc"

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This kernel support gfx12 only" << std::endl;

        return 0;
    }
    return run_convnd_fwd_example(argc, argv) ? 0 : 1;
}
