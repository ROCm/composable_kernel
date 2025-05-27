// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

//#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

#if 0
template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
        NDimSpatial,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                      ck::tensor_layout::convolution::GNHWC,
                                      ck::tensor_layout::convolution::GNDHWC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                      ck::tensor_layout::convolution::GKYXC,
                                      ck::tensor_layout::convolution::GKZYXC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                      ck::tensor_layout::convolution::GNHWK,
                                      ck::tensor_layout::convolution::GNDHWK>>,
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        4,                    // K0PerBlock
        8,                    // K1
        32,                   // MPerXdl
        32,                   // NPerXdl
        2,                    // MXdlPerWave
        2,                    // NXdlPerWave
        S<1, 4, 16, 4>,       // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 3, 1, 2>,        // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // ABlockTransferSrcAccessOrder
        2,                    // ABlockTransferSrcVectorDim
        1,                    // ABlockTransferSrcScalarPerVector
        1,                    // ABlockTransferDstScalarPerVector_K1
        true,                 // ABlockLdsAddExtraM
        S<1, 4, 16, 4>,       // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // BBlockTransferSrcAccessOrder
        2,                    // BBlockTransferSrcVectorDim
        1,                    // BBlockTransferSrcScalarPerVector
        1,                    // BBlockTransferDstScalarPerVector_K1
        true,                 // BBlockLdsAddExtraN
        1,                    // CShuffleMXdlPerWavePerShuffle
        1,                    // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,       // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        1>; // CBlockTransferScalarPerVector_NWaveNPerXdl

#endif

//       DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    32,     32,   8,   32,   32,    1,    1,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              2,              2,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  1, Scheduler, PipelineVersion, 2>,

 //ConvBwdWeightDefault,
//is_NHWGC_GKYXC_NHWGK
using ALayout  = ck::tensor_layout::convolution::NHWGC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::NHWGK;
//using Scheduler =ck::BlockGemmPipelineScheduler::Intrawave;
//using PipelineVersion =ck::BlockGemmPipelineVersion::v1;
template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<
        NDimSpatial,
        ALayout,
        BLayout,
        ELayout,
        F16,
        F16,
        F16,
        F32,
        PassThrough,
        PassThrough,
        PassThrough,
        ConvBwdWeightDefault,
        64,
        32,//16,
        64,
        32,//64,
        8,
        32, //16,
        32, //16,
        1,
        2, //4,
        S<4, 8, 1>,// S<8, 8, 1>
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        2,
        2,
        false,
        S<4, 16, 1>,
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        2,
        2,
        false,
        1,
        1,
        S<1, 8, 1, 8>,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        2>;
#if 0

       DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    64,     32,   8,   32,   32,    1,    2,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              2,              2,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  1, Scheduler, PipelineVersion, 2>,


64, 16, 16, 32, 8, 16, 16, 1, 1, S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false, S<4, 8, 1>,
    S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false, 1, 1, S<1, 8, 1, 8>, 1,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 8 > ;
#endif

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
    case 1: break;//return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    case 3: break;//return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
