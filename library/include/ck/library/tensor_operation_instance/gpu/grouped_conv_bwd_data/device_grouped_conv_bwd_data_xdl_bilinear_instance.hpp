// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using BF8  = ck::bf8_t;
using F8   = ck::f8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

static constexpr auto ConvBwdDataDefault = ConvolutionBackwardDataSpecialization::Default;

static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
    ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;

// f16_f16_f32_f16
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_xdl_bilinear_f16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|           DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |                   |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F16,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,    64,    64,    64,    32,   8,   8,   32,   32,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,                1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F16,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,    64,   128,    32,   8,   8,   32,   32,       1,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            1,     S<1, 32, 1, 8>,                8>,
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F16,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   128,   128,    32,    32,   8,   8,   32,   32,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,        S<4, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 32, 1, 4>,                1>,

        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F16,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,   128,   256,    32,   8,   2,   32,   32,       2,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                8>
    // clang-format on
    >;

// bf16_bf16_f32_bf16
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_xdl_bilinear_bf16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|           DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |                   |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,  BF16,  BF16,     F32,     BF16,  Tuple<BF16>,  BF16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,    64,    64,    64,    32,   8,   8,   32,   32,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,    S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,                1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,  BF16,  BF16,     F32,     BF16,  Tuple<BF16>,  BF16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,    64,   128,    32,   8,   8,   32,   32,       1,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,   S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            1,     S<1, 32, 1, 8>,                8>,
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,  BF16,  BF16,     F32,     BF16,  Tuple<BF16>,  BF16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   128,   128,    32,    32,   8,   8,   32,   32,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,    S<4, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 32, 1, 4>,                1>,

        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,  BF16,  BF16,     F32,     BF16,  Tuple<BF16>,  BF16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,   128,   256,    32,   8,   2,   32,   32,       2,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,   S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                8>
    // clang-format on
    >;

// f32_f32_f32_f32
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_xdl_bilinear_f32_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|           DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |                   |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F32,   F32,     F32,      F32,  Tuple<F32>,   F32,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,    64,    64,    64,    32,   8,   8,   32,   32,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,         1,            1,            1,     S<1, 16, 1, 4>,                1>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F32,   F32,     F32,      F32,  Tuple<F32>,   F32,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,    64,   128,    32,   8,   8,   32,   32,       1,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              4,         1,            1,            1,     S<1, 32, 1, 8>,                4>,
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F32,   F32,     F32,      F32,  Tuple<F32>,   F32,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   128,   128,    32,    32,   8,   8,   32,   32,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,         1,            1,            1,     S<1, 32, 1, 4>,                1>,

        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F32,   F32,     F32,      F32,  Tuple<F32>,   F32,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,   128,   256,    32,   8,   2,   32,   32,       2,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                4>
    // clang-format on
    >;

// f16_f16_f16_comp_f8
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_xdl_bilinear_input_fp16_comp_bf8f8_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|           DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |                   |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
        // ##############################################|           |        |        |                   |        |      |      |        |         |            |      |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F32,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,    64,    64,    64,    32,   8,   8,   32,   32,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,         1,            1,            1,     S<1, 16, 1, 4>,                1,  LoopScheduler::Default, BF8, F8>,
        // instances for small conv.K and conv.C
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F32,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,    64,   128,    32,   8,   8,   32,   32,       1,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              4,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              4,         1,            1,            1,     S<1, 32, 1, 8>,                4,  LoopScheduler::Default, BF8, F8>,
        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F32,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   128,   128,    32,    32,   8,   8,   32,   32,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,         1,            1,            1,     S<1, 32, 1, 4>,                1,  LoopScheduler::Default, BF8, F8>,

        DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1< NDimSpatial, ALayout, BLayout, ck::Tuple<ELayout>, ELayout,   F16,   F16,     F32,      F32,  Tuple<F16>,   F16,  PassThrough,  PassThrough,       Bilinear,            ConvSpec,  true,  true,             1,   256,   128,   256,    32,   8,   2,   32,   32,       2,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                4,  LoopScheduler::Default, BF8, F8>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
