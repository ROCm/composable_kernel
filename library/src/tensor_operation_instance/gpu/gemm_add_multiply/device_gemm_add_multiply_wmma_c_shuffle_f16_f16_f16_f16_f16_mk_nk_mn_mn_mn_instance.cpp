// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16       = ck::half_t;
using F32       = float;
using F16_Tuple = ck::Tuple<F16, F16>;

using Row       = ck::tensor_layout::gemm::RowMajor;
using Col       = ck::tensor_layout::gemm::ColumnMajor;
using Row_Tuple = ck::Tuple<Row, Row>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddMultiply = ck::tensor_operation::element_wise::AddMultiply;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using device_gemm_add_multiply_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances =
    std::tuple<
        // clang-format off
        // no padding
        //##############################|      A|      B|        Ds|      E| AData| BData| AccData| CShuffle|    DsData| EData|           A|           B|         CDE|           GEMM| Prefetch| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CDEShuffleBlockTransferClusterLengths|         CDEShuffleBlock|
        //##############################| Layout| Layout|    Layout| Layout|  Type|  Type|    Type| DataType|      Type|  Type| Elementwise| Elementwise| Elementwise| Specialization|    Stage|  Size| Block| Block| Block|    | WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|                     _MBlock_MPerBlock| TransferScalarPerVector|
        //##############################|       |       |          |       |      |      |        |         |          |      |   Operation|   Operation|   Operation|               |         |      |      |      |      |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                     _NBlock_NPerBlock|              _NPerBlock|
        //##############################|       |       |          |       |      |      |        |         |          |      |            |            |            |               |         |      |      |      |      |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                      |                        |         
        DeviceGemmMultipleD_Wmma_CShuffle<    Row,    Col, Row_Tuple,    Row,   F16,   F16,     F32,      F16, F16_Tuple,   F16, PassThrough, PassThrough, AddMultiply,    GemmDefault,        1,   256,   128,   128,    64,  8,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,           1,           1,                        S<1, 32, 1,  8>,                      8>,

        // M/N/K Padding
        //##############################|      A|      B|        Ds|      E| AData| BData| AccData| CShuffle|    DsData| EData|           A|           B|         CDE|           GEMM| Prefetch| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CDEShuffleBlockTransferClusterLengths|         CDEShuffleBlock|
        //##############################| Layout| Layout|    Layout| Layout|  Type|  Type|    Type| DataType|      Type|  Type| Elementwise| Elementwise| Elementwise| Specialization|    Stage|  Size| Block| Block| Block|    | WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|                     _MBlock_MPerBlock| TransferScalarPerVector|
        //##############################|       |       |          |       |      |      |        |         |          |      |   Operation|   Operation|   Operation|               |         |      |      |      |      |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                     _NBlock_NPerBlock|              _NPerBlock|
        //##############################|       |       |          |       |      |      |        |         |          |      |            |            |            |               |         |      |      |      |      |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                      |                        |         
        DeviceGemmMultipleD_Wmma_CShuffle<    Row,    Col, Row_Tuple,    Row,   F16,   F16,     F32,      F16, F16_Tuple,   F16, PassThrough, PassThrough, AddMultiply, GemmMNKPadding,        1,   256,   128,   128,    64,  8,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,           1,           1,                        S<1, 32, 1,  8>,                      8>
        // clang-format on
        >;

void add_device_gemm_add_multiply_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_add_multiply_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
