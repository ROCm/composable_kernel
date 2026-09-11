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
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        1,                    // NumGemmKPrefetchStage
        256,                  // TileLoadThreadGroupSize
        256,                  // TileMathThreadGroupSize
        64,                   // MPerBlock
        64,                   // NPerBlock
        32,                   // K0PerBlock
        8,                    // K1
        32,                   // MPerXdl
        32,                   // NPerXdl
        1,                    // MXdlPerWave
        1,                    // NXdlPerWave
        S<4, 32, 2>,          // ABlockTransferThreadClusterLengths_K0_M_K1
        S<2, 0, 1>,           // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,           // ABlockTransferSrcAccessOrder
        1,                    // ABlockTransferSrcVectorDim
        2,                    // ABlockTransferSrcScalarPerVector
        4,                    // ABlockTransferDstScalarPerVector_K1
        false,                // ABlockLdsAddExtraM
        S<4, 32, 2>,          // BBlockTransferThreadClusterLengths_K0_N_K1
        S<2, 0, 1>,           // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,           // BBlockTransferSrcAccessOrder
        1,                    // BBlockTransferSrcVectorDim
        2,                    // BBlockTransferSrcScalarPerVector
        4,                    // BBlockTransferDstScalarPerVector_K1
        false,                // BBlockLdsAddExtraN
        1,                    // CShuffleMXdlPerWavePerShuffle
        1,                    // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>,       // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        2>;                   // CBlockTransferScalarPerVector_NWaveNPerXdl

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
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    case 3: return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
