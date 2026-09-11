// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault      = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto BlkGemmPipeSched = ck::BlockGemmPipelineScheduler::Intrawave;
static constexpr auto BlkGemmPipeVer   = ck::BlockGemmPipelineVersion::v3;

// A[m, k] * B[n, k] = C[m, n] with data cache prefetch support
template <bool UseDataCachePrefetch>
using device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_v3_instances = std::tuple<
    // clang-format off
    //#########################|ALayout|BLayout| CLayout| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| Block|  MPer|  NPer| KPer | AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|   BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|                 |               | Compute | Compute | Permute | Minimum  |         Use         |
    //#########################|       |       |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|    PipeScheduler|    PipelineVer|  TypeA  |  TypeB  |    A/B  | Occupancy|  DataCachePrefetch  |
    //#########################|       |       |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|                 |               |         |         |         |          |                     |
    //#########################|       |       |        |      |      |      |        |         |            |            |            |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                 |               |               |              |               |               |          |            |            |                             |                |                 |               |         |         |         |          |                     |
    // 128x128x64
    DeviceGemm_Xdl_CShuffleV3<     Row,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough,    GemmDefault,   256,   128,   128,    64,   8,   8,   16,   16,    4,    4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,      S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,           1,           2,               S<1, 32, 1, 8>,               8, BlkGemmPipeSched, BlkGemmPipeVer,     BF16,     BF16,    false,        0, UseDataCachePrefetch>,
    // 256x128x64
    DeviceGemm_Xdl_CShuffleV3<     Row,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough,    GemmDefault,   256,   256,   128,    64,   8,   8,   16,   16,    8,    4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,      S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,           1,           2,               S<1, 32, 1, 8>,               8, BlkGemmPipeSched, BlkGemmPipeVer,     BF16,     BF16,    false,        0, UseDataCachePrefetch>,
    // 128x256x64
    DeviceGemm_Xdl_CShuffleV3<     Row,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough,    GemmDefault,   256,   128,   256,    64,   8,   8,   16,   16,    4,    8,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,      S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,           1,           2,               S<1, 32, 1, 8>,               8, BlkGemmPipeSched, BlkGemmPipeVer,     BF16,     BF16,    false,        0, UseDataCachePrefetch>,
    // 256x256x64
    DeviceGemm_Xdl_CShuffleV3<     Row,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough,    GemmDefault,   256,   256,   256,    64,   8,   8,   16,   16,    8,    8,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,      S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         0,           1,           2,               S<1, 32, 1, 8>,               8, BlkGemmPipeSched, BlkGemmPipeVer,     BF16,     BF16,    false,        0, UseDataCachePrefetch>
    // clang-format on
    >;

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_v3_prefetch_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    if(ck::is_gfx125_supported())
    {
        add_device_operation_instances(
            instances, device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_v3_instances<true>{});
    }
}

void add_device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_v3_no_prefetch_instances(
    std::vector<std::unique_ptr<
        DeviceGemmV2<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    if(ck::is_gfx125_supported())
    {
        add_device_operation_instances(
            instances, device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_v3_instances<false>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
