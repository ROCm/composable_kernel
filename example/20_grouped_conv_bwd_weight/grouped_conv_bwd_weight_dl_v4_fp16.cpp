// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "device_grouped_conv_bwd_weight_dl_v4.hpp"


using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

using ALayout = ck::tensor_layout::convolution::GNHWC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::GNHWK;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<NDimSpatial,
                                                                 256,
                                                                  ALayout,
                                                                  BLayout,
                                                                  ELayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  S<28, 28>,
                                                                  5,
                                                                  ck::Tuple<S<1,1>, S<1,1>, S<2,2>>,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp,
                                                                  2,  // N batch
                                                                  1,  // NumWavePerTile
                                                                  4,  // InScalarPerVector
                                                                  4,  // OutScalarPerVector
                                                                  2,  // DstScalarPerVector
                                                                  false>;

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
