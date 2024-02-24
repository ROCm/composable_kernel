// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough          = ck::tensor_operation::element_wise::PassThrough;
using ScaleAddScaleAddRelu = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 = ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 = ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_bf16_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|             CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise|     Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|       Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |                |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32, BF16, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32, BF16, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    32,   8,   8,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32, BF16, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,

        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32, BF16, ck::Tuple<BF16, BF16>, BF16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_f16_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|             CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise|     Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|       Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |                |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32, F16, ck::Tuple<F16, F16>, F16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32, F16, ck::Tuple<F16, F16>, F16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    32,   8,   8,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32, F16, ck::Tuple<F16, F16>, F16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,

        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32, F16, ck::Tuple<F16, F16>, F16, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_f32_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|             CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise|     Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|       Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |                |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32, F32, ck::Tuple<F32, F32>, F32, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    16,   4,   4,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,         1,           1,           1,              S<1,  8, 1,  8>,              1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32, F32, ck::Tuple<F32, F32>, F32, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    16,   4,   4,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1,  8, 1,  8>,              1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32, F32, ck::Tuple<F32, F32>, F32, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    16,   4,   4,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,         1,           1,           1,              S<1, 16, 1, 16>,              4>,

        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32, F32, ck::Tuple<F32, F32>, F32, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,              4>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_int8_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|                 Ds| EData|           A|           B|             CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|           DataType|  Type| Elementwise| Elementwise|     Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |                   |      |   Operation|   Operation|       Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |                   |      |            |            |                |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout, int8_t, int8_t, int32_t, int8_t, ck::Tuple<F32, F32>, int8_t, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout, int8_t, int8_t, int32_t, int8_t, ck::Tuple<F32, F32>, int8_t, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    32,   8,   8,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout, int8_t, int8_t, int32_t, int8_t, ck::Tuple<F32, F32>, int8_t, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,

        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout, int8_t, int8_t, int32_t, int8_t, ck::Tuple<F32, F32>, int8_t, PassThrough, PassThrough, ScaleAddScaleAddRelu,  ConvSpec, GemmMNKPadding,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
