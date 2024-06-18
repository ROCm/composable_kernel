// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmKPadding   = ck::tensor_operation::device::GemmSpecialization::KPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
template <ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::PipelineVersion PipVer,
          ck::LoopScheduler LoopSche>
using device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances = std::tuple<
    // clang-format off
        //#########################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#########################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#########################|     |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#########################|     |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,   128,     4,  8,   16,   16,    1,    4,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,   256,     4,  8,   16,   16,    1,    8,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,    16,   256,     4,  8,   16,   16,    1,    4,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 16>,              4,  F16, PipVer, LoopSche>,        
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,    16,   512,     4,  8,   16,   16,    1,    8,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 16>,              4,  F16, PipVer, LoopSche>,

        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   128,    16,     4,  8,   16,   16,    4,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   256,    16,     4,  8,   16,   16,    8,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   256,    16,     4,  8,   16,   16,    4,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 64, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   512,    16,     4,  8,   16,   16,    8,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 64, 1, 4>,               4,  F16, PipVer, LoopSche>,

        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,    64,    16,    16,     8,  8,   16,   16,    1,    1,  S<1, 8,  8, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8,  8, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,    64,    16,    16,    16,  8,   16,   16,    1,    1,  S<1, 16, 4, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 16, 4, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               4,  F16, PipVer, LoopSche>,
        
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,    32,     8,  8,   16,   16,    1,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,    64,     8,  8,   16,   16,    1,    2,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,   128,     8,  8,   16,   16,    1,    4,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,   256,     8,  8,   16,   16,    1,    8,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,    16,   256,     8,  8,   16,   16,    1,    4,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 16>,              4,  F16, PipVer, LoopSche>,        

        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    32,    16,     8,  8,   16,   16,    1,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    64,    16,     8,  8,   16,   16,    2,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   128,    16,     8,  8,   16,   16,    4,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   256,    16,     8,  8,   16,   16,    8,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               4,  F16, PipVer, LoopSche>,
        DeviceGemmXdlSplitKCShuffle<  F16,   F16,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   256,    16,     8,  8,   16,   16,    4,    1,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,      true,           1,           1,                   S<1, 64, 1, 4>,               4,  F16, PipVer, LoopSche>
    // clang-format on
    >;

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances<
                                       GemmDefault,
                                       ck::PipelineVersion::v2,
                                       ck::LoopScheduler::Default>{});
    add_device_operation_instances(instances,
                                   device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances<
                                       GemmKPadding,
                                       ck::PipelineVersion::v2,
                                       ck::LoopScheduler::Default>{});
    add_device_operation_instances(instances,
                                   device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instances<
                                       GemmMNKPadding,
                                       ck::PipelineVersion::v2,
                                       ck::LoopScheduler::Default>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
