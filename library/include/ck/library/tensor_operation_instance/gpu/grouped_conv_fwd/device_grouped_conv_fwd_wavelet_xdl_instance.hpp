// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wavelet_xdl_c_shuffle_f16_instances = std::tuple<
    // clang-format off
        //#########################################################|      NumDim|       A|       B|          Ds|       E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|  TileLoad|  TileMath|  MPer|  NPer|  KPer| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#########################################################|     Spatial|  Layout|  Layout|      Layout|  Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization|    Thread|    Thread| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //#########################################################|            |        |        |            |        |      |      |        |         |            |      |   Operation|   Operation|   Operation|               | GroupSize| GroupSize|      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //#########################################################|            |        |        |            |        |      |      |        |         |            |      |            |            |            |               |          |          |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,   128,    64,    64,  8,   32,   32,    2,    1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,   128,    64,    64,  8,   32,   32,    1,    2,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       256,       256,   128,    64,    64,  8,   32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       256,       256,    64,   128,    64,  8,   32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 16, 1, 16>,               8>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,   256,    64,    64,  8,   32,   32,    2,    2,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       256,       256,   256,    64,    64,  8,   32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,   128,    64,   128,  8,   32,   32,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,    64,    64,   128,  8,   32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       256,       256,    64,    64,   128,  8,   32,   32,    1,    1,    S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,    S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       128,       256,    64,   128,   128,  8,   32,   32,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 16, 1, 16>,               8>,
        DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3<NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThrough, PassThrough, PassThrough,       ConvSpec,       256,       256,    64,   128,   128,  8,   32,   32,    2,    1,    S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,    S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 16, 1, 16>,               8>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
