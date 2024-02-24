// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for a[k, m] * b[n, k] = c[m, n]
using Instances =
    std::tuple<
// clang-format off
#if CK_EXPERIMENTAL_INTER_WAVE_INSTANCES        
        // pipeline v1, 2 waves
        //##########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer| NumPrefetch|          LoopScheduler|                     Pipeline|
        //##########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|            |                       |                             |
        //##########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|            |                       |                             |
        //##########|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |            |                       |                             |
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,   256,   128,     4,  8,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,   128,   256,     4,  8,   32,   32,    2,    4,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   128,   128,   128,     4,  8,   32,   32,    4,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,   128,   128,     4,  8,   32,   32,    2,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   128,   128,    64,     4,  8,   32,   32,    2,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   128,    64,   128,     4,  8,   32,   32,    2,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,   128,    64,     4,  8,   32,   32,    2,    1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>,
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,    64,   128,     4,  8,   32,   32,    1,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>
#endif
        // clang-format on
        >;

void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_interwave_pipeline_v1_instances(
    OwnerList<InstanceNN>& instances)
{
    add_device_operation_instances(instances, Instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
