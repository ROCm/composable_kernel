// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32 = float;

#ifdef CK_ENABLE_FP8
using F8 = ck::f8_t;
#endif

#ifdef CK_ENABLE_BF8
using BF8 = ck::bf8_t;
#endif

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

#ifdef CK_ENABLE_FP8

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename OutElementOp>
using device_grouped_conv_fwd_wmma_cshufflev3_binary_outelementop_f8_instances = std::tuple<
// clang-format off
          //########################################|     NumDim|       A|       B|       Ds|        E| AData| BData| AccData| CShuffle|             Ds| EData|           A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MWmma| NWmma|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|     CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
          //########################################|    Spatial|  Layout|  Layout|   Layout|   Layout|  Type|  Type|    Type| DataType|       DataType|  Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|    |    | WMMA| WMMA|   Per|   Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MWmmaPerWave| NWmmaPerWave|        _MBlock_MWaveMPerWmma| ScalarPerVector|
          //########################################|           |        |        |         |         |      |      |        |         |               |      |   Operation|   Operation|    Operation|               |               |      |      |      |      |    |    |     |     |  Wave|  Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|   PerShuffle|        _NBlock_NWaveNPerWmma|  _NWaveNPerWmma|
          //########################################|           |        |        |         |         |      |      |        |         |               |      |            |            |             |               |               |      |      |      |      |    |    |     |     |      |      |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |             |                             |                |
#ifdef CK_ENABLE_FP8
    // generic instance
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    64,    32,   8,   8,   16,   16,     4,     2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,            1,            1,               S<1, 16, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>
// #ifndef ONE_INSTANCE_PER_LIST
//     ,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    32,    32,   8,   8,   16,   16,     4,     1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,   16,   16,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   128,    32,   8,   8,   16,   16,     8,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   256,    32,   8,   8,   16,   16,     4,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,   128,   128,    32,   8,   8,   16,   16,     8,     2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,   16,   16,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    64,    32,   8,   8,   16,   16,     4,     2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 4>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,    64,   128,    32,   8,   8,   16,   16,     4,     2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    64,    32,   8,   8,   16,   16,     4,     2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 4>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>, 
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    64,    32,   8,   8,   16,   16,     4,     1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>, 
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,    64,   128,    32,   8,   8,   16,   16,     2,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    32,    32,   8,   8,   16,   16,     4,     1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 4>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,    32,   128,    32,   8,   8,   16,   16,     2,     2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    32,    32,   8,   8,   16,   16,     4,     1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 4>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,    32,   8,   8,   16,   16,     2,     2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 4>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    96,    64,   8,   8,   16,   16,     4,     3,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 32, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    96,    64,   8,   8,   16,   16,     2,     3,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 64, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    96,    64,   8,   8,   16,   16,     2,     3,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,            1,            1,               S<1, 64, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>,
//     DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, DsLayout,  ELayout,    F8,    F8,     F32,      F32,     Tuple<F32>,    F8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,   128,   128,    32,   8,   8,   16,   16,     8,     2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,            1,            1,               S<1, 16, 1, 8>,               8, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1, true, F8>
// #endif
#endif
    // clang-format on
    >;

#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
