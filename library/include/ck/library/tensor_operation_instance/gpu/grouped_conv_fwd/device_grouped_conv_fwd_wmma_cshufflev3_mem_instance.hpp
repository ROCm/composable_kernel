// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef CK_ENABLE_FP8
using F8 = ck::f8_t;
#endif

#ifdef CK_ENABLE_BF8
using BF8 = ck::bf8_t;
#endif

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using I8   = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddClamp    = ck::tensor_operation::element_wise::AddClamp;
using Clamp       = ck::tensor_operation::element_wise::Clamp;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto ConvFwd1x1P0   = ConvolutionForwardSpecialization::Filter1x1Pad0;
static constexpr auto ConvFwd1x1S1P0 = ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;
static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_wmma_cshufflev3_int8_mem_instances = std::tuple<
    // clang-format off
          //########################################|     NumDim|       A|       B|          Ds|       E| AData| BData| AccData| CShuffle|             Ds| EData|           A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MWmma| NWmma|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|     CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|                   Pipeline scheduler |            Pipeline version |
          //########################################|    Spatial|  Layout|  Layout|      Layout|  Layout|  Type|  Type|    Type| DataType|       DataType|  Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|    |    | WMMA| WMMA|   Per|   Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MWmmaPerWave| NWmmaPerWave|        _MBlock_MWaveMPerWmma| ScalarPerVector|                                      |                             |
          //########################################|           |        |        |            |        |      |      |        |         |               |      |   Operation|   Operation|    Operation|               |               |      |      |      |      |    |    |     |     |  Wave|  Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|   PerShuffle|        _NBlock_NWaveNPerWmma|  _NWaveNPerWmma|                                      |                             |
          //########################################|           |        |        |            |        |      |      |        |         |               |      |            |            |             |               |               |      |      |      |      |    |    |     |     |      |      |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |             |                             |                |                                      |                             |
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,    32,    32,    64,   8,   8,   16,   16,     1,     1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,            1,            1,               S<1, 16, 1, 8>,               2,                      BlkGemmPipeSched, BlockGemmPipelineVersion::v1>
#ifndef ONE_INSTANCE_PER_LIST
    ,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    16,   128,   8,   8,   16,   16,     1,     1,     S<16, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<16, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,            1,            1,               S<1, 16, 1, 4>,               4,                      BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    16,    64,   8,   8,   16,   16,     1,     1,     S<8,  8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8,  8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,            1,            1,               S<1, 16, 1, 4>,               4,                      BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   128,    32,    32,    64,   8,   8,   16,   16,     1,     1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,            1,            1,               S<1, 16, 1, 8>,               4,                      BlkGemmPipeSched, BlockGemmPipelineVersion::v1>
#endif
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
