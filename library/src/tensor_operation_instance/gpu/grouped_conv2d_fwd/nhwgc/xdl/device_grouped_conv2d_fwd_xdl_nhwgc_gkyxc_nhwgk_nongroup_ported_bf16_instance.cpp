// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using NHWGC = ck::tensor_layout::convolution::NHWGC;
using GKYXC = ck::tensor_layout::convolution::GKYXC;
using NHWGK = ck::tensor_layout::convolution::NHWGK;

using EmptyTuple = Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_nongroup_ported_instances = std::tuple<
    // clang-format off
        //##########################################################################| InData| WeiData| OutData| AccData|          In|         Wei|         Out|    ConvForward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //##########################################################################|   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //##########################################################################|       |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //##########################################################################|       |        |        |        |            |            |            |               |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   256,   128,     32,  8, 8,   32,   32,    4,    2,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   128,   256,     32,  8, 8,   32,   32,    2,    4,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,   128,   128,     32,  8, 8,   32,   32,    4,    2,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   128,   128,     32,  8, 8,   32,   32,    2,    2,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,   128,    64,     32,  8, 8,   32,   32,    2,    2,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,    64,   128,     32,  8, 8,   32,   32,    2,    2,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,    64,    64,    64,     32,  8, 8,   32,   32,    2,    2,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   128,    64,     32,  8, 8,   32,   32,    2,    1,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,    64,   128,     32,  8, 8,   32,   32,    1,    2,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  8, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,   128,    32,     32,  8, 8,   32,   32,    2,    1,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,    32,   128,     32,  8, 8,   32,   32,    1,    2,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  4, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,    64,    64,    32,     32,  8, 8,   32,   32,    2,    1,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,    64,    32,    64,     32,  8, 8,   32,   32,    1,    2,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<4,  2, 8>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   128,    64,     16,  4,  4,   32,   32,    2,    1,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   128,    64,     8,  4,  4,   32,   32,    2,    1,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   256,   256,    64,     8,  4,  4,   32,   32,    4,    1,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<2, 32, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,   128,    64,     8,  4,   4,  32,   32,    2,    2,     S<2, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<2, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle< 2 ,NHWGC, GKYXC, EmptyTuple, NHWGK,                   BF16,   BF16,     F32,      BF16,    EmptyTuple,        BF16, PassThrough, PassThrough, PassThrough,        ConvFwdDefault,GemmMNKPadding, 1,   128,    64,    64,     8,  4,   4,  32,   32,    1,    2,     S<2, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,      true,     S<2, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,      true,           1,           1,             S<1, 16, 1, 4>,               8>
    // clang-format on
    >;

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_nongroup_ported_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                EmptyTuple,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                EmptyTuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_nongroup_ported_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
