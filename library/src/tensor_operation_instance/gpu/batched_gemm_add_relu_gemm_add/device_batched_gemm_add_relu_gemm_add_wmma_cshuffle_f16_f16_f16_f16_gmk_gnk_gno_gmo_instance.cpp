// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_wmma_cshuffle_v3.hpp"
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

using PassThrough   = ck::tensor_operation::element_wise::PassThrough;
using CDE0ElementOp = ck::tensor_operation::element_wise::AddRelu;
using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

// c[g, m, n] = a[g, m, k] * b[g, n, k]
using device_batched_gemm_add_relu_gemm_add_wmma_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances =
    std::tuple<
        // clang-format off
        //#####################################################| A0Layout| B0Layout|       D0Layout| B1Layout|      D1sLayout| E1Layout| A0Data| B0Data|     D0DataType| B1Data|     D1DataType| E1Data| AccData| CShuffle|           A0|          B0|          CDE0|          B1|          CDE1|              GemmSpecialization| Block|  Gemm0| Gemm0| Gemm0| Gemm1| Gemm1|A0K1|B0K1| B1K1| MPer| NPer| MRepeat| LRepeat| NRepeat|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockTransfer|A0BlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds| CDE0BlockTransfer|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|   C1Shuffle|   C1Shuffle| CDE1BlockTransferClusterLengths| CDE1BlockTransfer|
        //#####################################################|         |         |               |         |               |         |   Type|   Type|               |   Type|               |   Type|    Type| DataType|  Elementwise| Elementwise|   Elementwise| Elementwise|   Elementwise|                                |  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     | WMMA| WMMA|        |        |        |  ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|         SrcScalar|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|     MRepeat|     NRepeat|                 _MBlock_MRepeat|   ScalarPerVector|
        //#####################################################|         |         |               |         |               |         |       |       |               |       |               |       |        |         |    Operation|   Operation|     Operation|   Operation|     Operation|                                |      |  Block| Block| Block| Block| Block|    |    |     |     |     |        |        |        |Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |         PerVector|  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|                 _NBlock_NRepeat|          _NRepeat|
        //#####################################################|         |         |               |         |               |         |       |       |               |       |               |       |        |         |             |            |              |            |              |                                |      |       |      |      |      |      |    |    |     |     |     |        |        |        |               |               |               |               |               |               |          |                 |                |                |                |                |                |           |                  |                 |                |                |                |                |                |           |            |            |                                |                  |
        // No padding
        DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3<      Row,      Col, ck::Tuple<Row>,      Row, ck::Tuple<Row>,      Row,    F16,    F16, ck::Tuple<F16>,    F16, ck::Tuple<F16>,    F16,     F32,      F32,  PassThrough, PassThrough, CDE0ElementOp, PassThrough, CDE1ElementOp,     GemmSpecialization::Default,    32,     16,    64,    64,    64,    64,   8,   8,    8,   16,   16,       1,       4,       4,    S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,                 4,      S<2, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,       true,           1,           2,                  S<1, 16, 1, 2>,                8>,
        // Fallback with padding
        DeviceBatchedGemmMultipleDGemmMultipleD_Wmma_CShuffleV3<      Row,      Col, ck::Tuple<Row>,      Row, ck::Tuple<Row>,      Row,    F16,    F16, ck::Tuple<F16>,    F16, ck::Tuple<F16>,    F16,     F32,      F32,  PassThrough, PassThrough, CDE0ElementOp, PassThrough, CDE1ElementOp, GemmSpecialization::MNKOPadding,    32,     16,    64,    64,    64,    64,   8,   8,    8,   16,   16,       1,       4,       4,    S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,     false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               1,               8,      false,                 1,      S<2, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               1,               1,       true,           1,           2,                  S<1, 16, 1, 2>,                1>
        // clang-format on
        >;

void add_device_batched_gemm_add_relu_gemm_add_wmma_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultipleDGemmMultipleD<Row,
                                                                        Col,
                                                                        ck::Tuple<Row>,
                                                                        Row,
                                                                        ck::Tuple<Row>,
                                                                        Row,
                                                                        F16,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        CDE0ElementOp,
                                                                        PassThrough,
                                                                        CDE1ElementOp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_add_relu_gemm_add_wmma_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
