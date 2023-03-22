// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_dl.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

namespace {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using EmptyTuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault   = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using device_grouped_gemm_multiple_d_dl_f16_f16_f16_mk_nk_mn_instances = std::tuple<
    // clang-format off
        //     ##################| ALayout| BLayout|   DsLayout| ELayout| AData| BData| AccData|     DsData| EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|       BBlockTransfer|     CThreadTransfer|  CThreadTransfer|    CThreadTransfer|
        //     ##################|        |        |           |        |  Type|  Type|    Type|       Type|  Type| Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor|        SrcDstAccess|  SrcDstVectorDim| DstScalarPerVector|
        //     ##################|        |        |           |        |      |      |        |           |      |   Operation|   Operation|    Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder|  Lengths_K0_N0_N1_K1|               Order|                 |                   |
        //     ##################|        |        |           |        |      |      |        |           |      |            |            |             |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                     |                    |                 |                   |
    DeviceGroupedGemmMultipleD_Dl<     Row,     Col, EmptyTuple,     Row,   F16,   F16,     F32, EmptyTuple,   F16, PassThrough, PassThrough,  PassThrough,    GemmDefault,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>,      S<8, 1, 1, 2>,     S<2, 1,  128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  4>
    // clang-format on
    >;

using device_grouped_gemm_multiple_d_dl_f16_f16_f16_mk_nk_mn_irregular_instances = std::tuple<
    // clang-format off
        //     ##################| ALayout| BLayout|   DsLayout| ELayout| AData| BData| AccData|     DsData| EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|       BBlockTransfer|     CThreadTransfer|  CThreadTransfer|    CThreadTransfer|
        //     ##################|        |        |           |        |  Type|  Type|    Type|       Type|  Type| Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor|        SrcDstAccess|  SrcDstVectorDim| DstScalarPerVector|
        //     ##################|        |        |           |        |      |      |        |           |      |   Operation|   Operation|    Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder|  Lengths_K0_N0_N1_K1|               Order|                 |                   |
        //     ##################|        |        |           |        |      |      |        |           |      |            |            |             |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                     |                    |                 |                   |
    DeviceGroupedGemmMultipleD_Dl<     Row,     Col, EmptyTuple,     Row,   F16,   F16,     F32, EmptyTuple,   F16, PassThrough, PassThrough,  PassThrough,  GemmMNPadding,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  4>,
    DeviceGroupedGemmMultipleD_Dl<     Row,     Col, EmptyTuple,     Row,   F16,   F16,     F32, EmptyTuple,   F16, PassThrough, PassThrough,  PassThrough,  GemmMNPadding,    64,     8,   128,    16,  2,          1,          4,      1,       S<2, 2>,       S<8, 2>,      S<2, 1, 1, 2>,      S<8, 1,   8, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<2, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>,     S<16, 1, 2, 2>,      S<1, 1,  64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  4>,
    DeviceGroupedGemmMultipleD_Dl<     Row,     Col, EmptyTuple,     Row,   F16,   F16,     F32, EmptyTuple,   F16, PassThrough, PassThrough,  PassThrough,  GemmMNPadding,    64,     4,   128,    16,  2,          1,          2,      1,       S<2, 1>,       S<8, 4>,      S<1, 1, 1, 2>,     S<16, 1,   4, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>,     S<16, 1, 2, 2>,      S<1, 1,  64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  2>
    // clang-format on
    >;

} // anonymous namespace

void add_device_grouped_gemm_multiple_d_dl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Col,
                                                  EmptyTuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  EmptyTuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances, device_grouped_gemm_multiple_d_dl_f16_f16_f16_mk_nk_mn_instances{});
    add_device_operation_instances(
        instances, device_grouped_gemm_multiple_d_dl_f16_f16_f16_mk_nk_mn_irregular_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
