// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using Instances = std::tuple<
// clang-format off
        // pipeline v2, 1 wave
        //#####################| ALayout| BLayout| CLayout| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|          LoopScheduler|                    Pipeline|
        //#####################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|                       |                            |
        //#####################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|                       |                            |
        //#####################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                       |                            |
#if CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES        
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    32,   8,   2,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,   256,    32,   8,   2,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,   128,   128,    32,   8,   2,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 16, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 16, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,   128,    32,   8,   2,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,   128,    64,    32,   8,   2,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 4>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 4>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,    64,   128,    32,   8,   2,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 16, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 16, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,    64,    32,   8,   2,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,    64,   128,    32,   8,   2,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>,
        DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,   F16,   F16,   F16,     F32,      F16, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,              8,  LoopScheduler::Default,        PipelineVersion::v2>
#endif
    // clang-format on
    >;

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_1_stage_default_pipeline_v2_instances(
    OwnerList<InstanceTT>& instances)
{
    add_device_operation_instances(instances, Instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
