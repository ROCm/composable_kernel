// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#define USE_WAVE_TRANSFER

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using BF8  = ck::bf8_t;
using F8   = ck::f8_t;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault = ConvolutionBackwardDataSpecialization::Default;

static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
    ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_cshufflev3_bf16_wave_transfer_instances = std::tuple<
    // clang-format off
          //########################################|     NumDim|       A|       B|          Ds|       E|        AData|         BData| AccData|         CShuffle|             Ds|         EData|           A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MWmma| NWmma|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|     CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|                   Pipeline scheduler |            Pipeline version |
          //########################################|    Spatial|  Layout|  Layout|      Layout|  Layout|         Type|          Type|    Type|         DataType|       DataType|          Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|    |    | WMMA| WMMA|   Per|   Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MWmmaPerWave| NWmmaPerWave|        _MBlock_MWaveMPerWmma| ScalarPerVector|                                      |                             |
          //########################################|           |        |        |            |        |             |              |        |                 |               |              |   Operation|   Operation|    Operation|               |               |      |      |      |      |    |    |     |     |  Wave|  Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|   PerShuffle|        _NBlock_NWaveNPerWmma|  _NWaveNPerWmma|                                      |                             |
          //########################################|           |        |        |            |        |             |              |        |                 |               |              |            |            |             |               |               |      |      |      |      |    |    |     |     |      |      |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |             |                             |                |                                      |                             |
      // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<1,1,1>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>,
        
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    32,    32,   8,   8,   16,   16,       4,       1,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_cshufflev3_f16_wave_transfer_instances = std::tuple<
    // clang-format off
          //########################################|     NumDim|       A|       B|          Ds|       E|        AData|         BData| AccData|         CShuffle|             Ds|         EData|           A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MWmma| NWmma|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|     CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|                   Pipeline scheduler |            Pipeline version |
          //########################################|    Spatial|  Layout|  Layout|      Layout|  Layout|         Type|          Type|    Type|         DataType|       DataType|          Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|    |    | WMMA| WMMA|   Per|   Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MWmmaPerWave| NWmmaPerWave|        _MBlock_MWaveMPerWmma| ScalarPerVector|                                      |                             |
          //########################################|           |        |        |            |        |             |              |        |                 |               |              |   Operation|   Operation|    Operation|               |               |      |      |      |      |    |    |     |     |  Wave|  Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|   PerShuffle|        _NBlock_NWaveNPerWmma|  _NWaveNPerWmma|                                      |                             |
          //########################################|           |        |        |            |        |             |              |        |                 |               |              |            |            |             |               |               |      |      |      |      |    |    |     |     |      |      |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |             |                             |                |                                      |                             |
   
         // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<1,1,1>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>,
        
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    32,    64,   8,   8,   16,   16,       4,       1,        S<8, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,        S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>,BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, false>

    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
