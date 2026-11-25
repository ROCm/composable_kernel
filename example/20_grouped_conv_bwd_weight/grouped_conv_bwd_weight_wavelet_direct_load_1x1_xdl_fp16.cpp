// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<
        NDimSpatial,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::NWGC,
                                      ck::tensor_layout::convolution::NHWGC,
                                      ck::tensor_layout::convolution::NDHWGC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                      ck::tensor_layout::convolution::GKYXC,
                                      ck::tensor_layout::convolution::GKZYXC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::NWGK,
                                      ck::tensor_layout::convolution::NHWGK,
                                      ck::tensor_layout::convolution::NDHWGK>>,
        InDataType,                        // InDataType
        WeiDataType,                       // WeiDataType
        OutDataType,                       // OutDataType
        AccDataType,                       // AccDataType
        InElementOp,                       // InElementwiseOperation
        WeiElementOp,                      // WeiElementwiseOperation
        OutElementOp,                      // OutElementwiseOperation
        ConvBwdWeightFilter1x1Stride1Pad0, // ConvolutionBackwardWeightSpecialization
        1,                                 // NumGemmKPrefetchStage
        256,                               // TileLoadThreadGroupSize
        256,                               // TileMathThreadGroupSize
        128,                               // MPerBlock
        128,                               // NPerBlock
        128,                               // K0PerBlock
        8,                                 // K1
        16,                                // MPerXdl
        16,                                // NPerXdl
        4,                                 // MXdlPerWave
        4,                                 // NXdlPerWave
        S<2, 16, 8>,                       // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 2, 1>,                        // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,                        // ABlockTransferSrcAccessOrder
        1,                                 // ABlockTransferSrcVectorDim
        8,                                 // ABlockTransferSrcScalarPerVector
        8,                                 // ABlockTransferDstScalarPerVector_K1
        true,                              // ABlockLdsAddExtraM
        S<2, 16, 8>,                       // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 2, 1>,                        // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,                        // BBlockTransferSrcAccessOrder
        1,                                 // BBlockTransferSrcVectorDim
        8,                                 // BBlockTransferSrcScalarPerVector
        8,                                 // BBlockTransferDstScalarPerVector_K1
        true,                              // BBlockLdsAddExtraN
        1,                                 // CShuffleMXdlPerWavePerShuffle
        2,                                 // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,              // CBlockTransferScalarPerVector_NWaveNPerXdl
        InDataType,
        InDataType,
        1,
        true,
        true>;

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParamFilter1x1Stride1Pad0;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: return !run_grouped_conv_bwd_weight<1>(config, conv_param, 2);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param, 2);
    case 3: return !run_grouped_conv_bwd_weight<3>(config, conv_param, 2);
    default: break;
    }

    return 1;
}
