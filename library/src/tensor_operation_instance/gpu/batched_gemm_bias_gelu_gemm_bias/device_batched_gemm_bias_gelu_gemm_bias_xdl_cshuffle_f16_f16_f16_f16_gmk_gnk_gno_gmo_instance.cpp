// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle.hpp"
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

static constexpr bool PadGemm0M = false;
static constexpr bool PadGemm0N = false;
static constexpr bool PadGemm0K = false;
static constexpr bool PadGemm1N = false;
static constexpr bool PadGemm1K = false;

// c[g, m, n] = a[g, m, k] * b[g, n, k]
using device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances =
    std::tuple<
        // clang-format off
        //#############################################| A0Layout| B0Layout|          D0Layout| B1Layout| C1Layout|       D1sLayout| A0Data| B0Data| Acc0DataType|        D0DataType| B1Data| Acc1CData| CShuffle| C1Data|        D1sData|          A0|          B0|          CDE0|          A1|          B1|           CDE1|    PadGemm0M|  PadGemm0N|  PadGemm0K|  PadGemm1N|  PadGemm1K| NumGemm0K| Block|  Gemm0| Gemm0| Gemm0| Gemm1| Gemm1| A0K1| B0K1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|  A0BlockTransfer| A0BlockTransfer| A0BlockTransfer| A0BlockTransfer| A0BlockTransfer| A0BlockTransfer| A0BlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    C1Shuffle|    C1Shuffle| C1BlockTransferClusterLengths|  C1BlockTransfer|
        //#############################################|         |         |                  |         |         |                |   Type|   Type|         Type|              Type|   Type|      Type| DataType|   Type|           Type| Elementwise| Elementwise|   Elementwise| Elementwise| Elementwise|    Elementwise|             |           |           |           |           |  Prefetch|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|     |     |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|  MXdlPerWave|  NXdlPerWave|          _MBlock_MWaveMPerXdl|  ScalarPerVector|
        //#############################################|         |         |                  |         |         |                |       |       |             |                  |       |          |         |       |               |   Operation|   Operation|     Operation|   Operation|   Operation|      Operation|             |           |           |           |           |     Stage|      |  Block| Block| Block| Block| Block|     |     |     |     |     |   Per|   Per|   Per|  Lengths_K0_M_K1|    ArrangeOrder|                |                |       PerVector|   PerVector_AK1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |   PerShuffle|   PerShuffle|          _NBlock_NWaveNPerXdl|    _NWaveNPerXdl|
        //#############################################|         |         |                  |         |         |                |       |       |             |                  |       |          |         |       |               |            |            |              |            |            |               |             |           |           |           |           |          |      |       |      |      |      |      |     |     |     |     |     |  Wave|  Wave|  Wave|                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |             |             |                              |                 |
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    PadGemm0M,  PadGemm0N,  PadGemm0K,  PadGemm1N,  PadGemm1K,    1,        256,    256,   128,    32,   128,    32,   8,    8,    2,   32,   32,     2,     4,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    PadGemm0M,  PadGemm0N,  PadGemm0K,  PadGemm1N,  PadGemm1K,    1,        256,    128,   128,    32,   128,    32,   8,    8,    2,   32,   32,     1,     4,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    PadGemm0M,  PadGemm0N,  PadGemm0K,  PadGemm1N,  PadGemm1K,    1,        256,    128,   128,    32,    64,    32,   8,    8,    2,   32,   32,     1,     4,     2,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    PadGemm0M,  PadGemm0N,  PadGemm0K,  PadGemm1N,  PadGemm1K,    1,        256,    128,    64,    32,   128,    32,   8,    8,    2,   32,   32,     1,     2,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        // Padded fallback kernel
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    true,        true,          true,       true,     true,       1,        256,    256,   128,    32,   128,    32,   8,    8,    2,   32,   32,     2,     4,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    true,        true,          true,       true,     true,       1,        256,    128,   128,    32,   128,    32,   8,    8,    2,   32,   32,     1,     4,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    true,        true,          true,       true,     true,       1,        256,    128,   128,    32,    64,    32,   8,    8,    2,   32,   32,     1,     4,     2,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<     Row,      Col,   ck::Tuple<Row>,      Row,       Row,   ck::Tuple<Row>,   F16,    F16,      F32,       ck::Tuple<F16>,     F16,     F32,      F32,      F16,   ck::Tuple<F16>, PassThrough, PassThrough, CDE0ElementOp, PassThrough, PassThrough, CDE1ElementOp,    true,        true,          true,       true,     true,       1,        256,    128,    64,    32,   128,    32,   8,    8,    2,   32,   32,     1,     2,     4,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,          true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8>
        // clang-format on
        >;

void add_device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmBiasGeluGemmBias<Row,
                                                                  Col,
                                                                  ck::Tuple<Row>,
                                                                  Row,
                                                                  Row,
                                                                  ck::Tuple<Row>,
                                                                  F16,
                                                                  F16,
                                                                  ck::Tuple<F16>,
                                                                  F16,
                                                                  F16,
                                                                  ck::Tuple<F16>,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  CDE0ElementOp,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  CDE1ElementOp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
