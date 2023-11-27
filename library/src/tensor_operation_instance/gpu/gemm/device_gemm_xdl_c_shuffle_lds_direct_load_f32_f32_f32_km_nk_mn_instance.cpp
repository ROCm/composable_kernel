// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_lds_direct_load.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instances = std::tuple<
    // clang-format off
    // ##################################| ALayout| BLayout| CLayout| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
    // ##################################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster| SrcAccessOrder|   SrcVectorDim|         Scalar| AddExtraM|   ThreadCluster| SrcAccessOrder|  SrcVectorDim|         Scalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
    // ##################################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|               |               |      PerVector|          | Lengths_K0_N_K1|               |              |      PerVector|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
    // ##################################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |          |                |               |              |               |          |            |            |                             |                |
    DeviceGemm_Xdl_CShuffle_LdsDirectLoad<     Col,     Col,     Row,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,    64,    64,    32,   8,   8,   32,   32,    1,    1,      S<4, 8, 8>,     S<0, 2, 1>,              1,              1,         1,      S<4, 8, 8>,     S<1, 0, 2>,             2,              1,         1,           1,           1,                S<1, 8, 1, 8>,               4>
    // clang-format on
    >;

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
