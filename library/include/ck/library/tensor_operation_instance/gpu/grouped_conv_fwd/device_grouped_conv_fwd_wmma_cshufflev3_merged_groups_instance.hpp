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

using I8 = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddClamp    = ck::tensor_operation::element_wise::AddClamp;
using Clamp       = ck::tensor_operation::element_wise::Clamp;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd3x3 = ConvolutionForwardSpecialization::Filter3x3;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_wmma_cshufflev3_merged_groups_int8_instances = std::tuple<
    // clang-format off
          //########################################|     NumDim|       A|       B|          Ds|       E| AData| BData| AccData| CShuffle|             Ds| EData|           A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MWmma| NWmma|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|     CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|                   Pipeline scheduler |            Pipeline version | AComp | BComp | Merge  |
          //########################################|    Spatial|  Layout|  Layout|      Layout|  Layout|  Type|  Type|    Type| DataType|       DataType|  Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|    |    | WMMA| WMMA|   Per|   Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MWmmaPerWave| NWmmaPerWave|        _MBlock_MWaveMPerWmma| ScalarPerVector|                                      |                             |  Type |  Type | Groups |
          //########################################|           |        |        |            |        |      |      |        |         |               |      |   Operation|   Operation|    Operation|               |               |      |      |      |      |    |    |     |     |  Wave|  Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|   PerShuffle|        _NBlock_NWaveNPerWmma|  _NWaveNPerWmma|                                      |                             |       |       |        |
          //########################################|           |        |        |            |        |      |      |        |         |               |      |            |            |             |               |               |      |      |      |      |    |    |     |     |      |      |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |             |                             |                |                                      |                             |       |       |        |
    // Instances with NumGroupsPerBatch > 1
    // TODO: I had to change A and B srcScalarPerVector from 8 to 1 in order to get these instances to be compatible with the device implementation. I am pretty sure they will not work for XDL either.
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,    32,   8,   8,   16,   16,     2,     2,    S<4, 16,  1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,            1,            1,               S<1, 16, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1,     I8,     I8,      8>
#ifndef ONE_INSTANCE_PER_LIST
    ,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,    32,   8,   8,   16,   16,     2,     2,    S<4, 16,  1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,    S<4, 16,  1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,            1,            1,               S<1, 16, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1,     I8,     I8,     16>,
    DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,    I8,    I8, int32_t,       I8,    DsDataTypes,    I8, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,    32,   8,   8,   16,   16,     2,     2,    S<4, 16,  1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,    S<4, 16,  1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,            1,            1,               S<1, 16, 1, 4>,               1, BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1,     I8,     I8,     32>
#endif
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
