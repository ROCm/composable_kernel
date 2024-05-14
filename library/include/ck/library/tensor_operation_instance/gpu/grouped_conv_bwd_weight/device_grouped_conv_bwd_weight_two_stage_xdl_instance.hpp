// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using F16 = ck::half_t;
using F32 = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_two_stage_xdl_c_shuffle_f16_instances = std::tuple<
    // clang-format off
        //#########################################|              Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer|
        //#########################################|              Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|
        //#########################################|          Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl|
        //#########################################|                 |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |
        DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    16,    16,     4,  8,   16,   16,    1,    1,  S<1, 4, 8,  1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,              2,              1,              4,      true,  S<1, 4, 8,  1>,  S<0, 3, 1, 2>,  S<0, 2, 1, 3>,             2,              1,              4,      true,           1,           1,   S<1, 8, 1, 8>,               1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
