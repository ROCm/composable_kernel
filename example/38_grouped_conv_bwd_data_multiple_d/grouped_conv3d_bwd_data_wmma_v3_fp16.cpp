// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp"
#include "common.hpp"

using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
using DsDataType       = ck::Tuple<>;
using InDataType       = FP16;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

// clang-format off
using DeviceConvInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3
// ######| NDimSpatial|   ALayout|   BLayout|   DsLayout|  ELayout|       AData|       BData|     AccData|         CShuffle|       DsData|      EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|         Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffle       | CShuffle    |   CDEBlockTransfer| CDEBlockTransfer|
// ######|            |          |          |           |         |        Type|        Type|        Type|         DataType|         Type|       Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN|          Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|    MRepeat     |     NRepeat |  _MBlock_MPerBlock|  ScalarPerVector|
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |      PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |                |             |                   |                 |
        < NDimSpatial_3D, OutLayout, WeiLayout,   DsLayout, InLayout, OutDataType, WeiDataType, AccDataType, CShuffleDataType,   DsDataType, InDataType, OutElementOp, WeiElementOp,    InElementOp,  ConvBwdDataFilter1x1Stride1Pad0,  true,  true,    64,    16,    64,    32,   8,   8,   16,   16,       1,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,      S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            1,     S<1, 16, 1, 4>,                S<8,8,8>>;

// clang-format on

#include "run_grouped_conv3d_bwd_data_example.inc"

int main(int argc, char* argv[]) { return run_grouped_conv_bwd_data_example(argc, argv); }
