// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define DISABLE_OUTPUT_LDS 1
#include "device_grouped_conv_bwd_data_dl_v4.hpp"
#include "common.hpp"

#define ENABLE_CONV_FACTORY 1

// kernel data types
using InKernelDataType  = FP16;
using WeiKernelDataType = FP16;
using AccDataType       = FP32;
using CShuffleDataType  = FP16;
using OutKernelDataType = FP16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<
        NDimSpatial,
        64,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        OutKernelDataType,
        S<28, 28>,
        5,
        ck::Tuple<S<1,1>, S<1,1>, S<2,2>>,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        4,
        4,4,
        4,
        4,
        false>;

using InType = InKernelDataType;
using WeiType = WeiKernelDataType;
using AccType = AccDataType;
using OutType = OutKernelDataType;

using DeviceConvBwdFactory = std::tuple<

//                                                NDimSpatial BlockSize In      Wei        Acc    Out      BlockTileSize FilterSize   FilterParam (dilation, stride, padding)                                     NBatch  SubTileH W  ScalarPerVector(in out)    RequirePadding>
      ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<7, 7>,     5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 2,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 4,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 2,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   5,           ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp,  8,       4, 4,       8, 4,                     false>

    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<7, 7>,     3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       1, 1,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<14, 14>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       2, 2,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<56, 56>,   3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       7, 8,       8, 8,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  2,       14, 16,     8, 8,                     false>

    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<28, 28>,   3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  32,      4, 4,       4, 2,                     false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdDlV4<2, 64,      InType, WeiType, AccType, OutType,  S<112, 112>, 3,           ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp,  8,       7, 8,       8, 8,                     false>
>;
#include "run_grouped_conv_bwd_data_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_bwd_example(argc, argv); }
