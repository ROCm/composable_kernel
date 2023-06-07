// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "device_grouped_conv2d_fwd_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv2d_fwd_wmma_f16_instances =
    std::tuple<
        // clang-format off
        //########################################|  NumDim|       A|       B|       Ds|       E| AData| BData|         Ds|  EData| AccData| CShuffle|            A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################| Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|   DataType|   Type|    Type| DataType|  Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|        |        |        |         |        |      |      |           |       |        |         |    Operation|   Operation|    Operation|               |               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|        |        |        |         |        |      |      |           |       |        |         |             |            |             |               |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<       2, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,     4,  8,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>
        // clang-format on
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
