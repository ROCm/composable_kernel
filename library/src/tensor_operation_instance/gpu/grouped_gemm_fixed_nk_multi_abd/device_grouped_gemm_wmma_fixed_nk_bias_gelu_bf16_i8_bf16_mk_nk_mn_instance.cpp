// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_wmma_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <index_t... Is>
using S = Sequence<Is...>;

using BF16 = bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

using Multiply    = element_wise::Multiply;
using PassThrough = element_wise::PassThrough;
using AddFastGelu = element_wise::AddFastGelu;
using Add         = element_wise::Add;
using FastGelu    = element_wise::FastGelu;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

template <typename DsLayout,
          typename DsDataType,
          typename CDEElementOp,
          GemmSpecialization GemmSpec>
using device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances = std::tuple<
    // clang-format off
        //######################################|    AsLayout|        BsLayout| DsLayout| ELayout|      AsData|          BsData| AccData| CShuffle|     DsData| EData|           A|           B|          CDE|          GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|       CBlockTransferClusterLengths|  CBlockTransfer|
        //######################################|            |                |         |        |        Type|            Type|    Type| DataType|       Type|  Type| Elementwise| Elementwise|  Elementwise|Spacialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat| _MBlock_MPerBlock_NBlock_NPerBlock| ScalarPerVector|
        //######################################|            |                |         |        |            |                |        |         |           |      |   Operation|   Operation|    Operation|              |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                                   |                |
        //######################################|            |                |         |        |            |                |        |         |           |      |            |            |             |              |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                   |                |
        DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK< Tuple<Row>, Tuple<Col, Col>, DsLayout,     Row, Tuple<BF16>, Tuple<I8, BF16>,     F32,     BF16, DsDataType,  BF16, PassThrough,    Multiply, CDEElementOp,      GemmSpec,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,                     S<1, 64, 1, 4>,               8>,
        DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK< Tuple<Row>, Tuple<Col, Col>, DsLayout,     Row, Tuple<BF16>, Tuple<I8, BF16>,     F32,     BF16, DsDataType,  BF16, PassThrough,    Multiply, CDEElementOp,      GemmSpec,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              2,              2,         1,           1,           1,                     S<1, 64, 1, 4>,               8>,
        DeviceGroupedGemm_Wmma_Multi_ABD_Fixed_NK< Tuple<Row>, Tuple<Col, Col>, DsLayout,     Row, Tuple<BF16>, Tuple<I8, BF16>,     F32,     BF16, DsDataType,  BF16, PassThrough,    Multiply, CDEElementOp,      GemmSpec,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,                     S<1, 64, 1, 4>,               8>
    // clang-format on
    >;

void add_device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<Tuple<Row>,
                                                                 Tuple<Col, Col>,
                                                                 Tuple<Row>,
                                                                 Row,
                                                                 Tuple<BF16>,
                                                                 Tuple<I8, BF16>,
                                                                 Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 AddFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances<
            Tuple<Row>,
            Tuple<BF16>,
            AddFastGelu,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<Tuple<Row>,
                                                                 Tuple<Col, Col>,
                                                                 Tuple<Row>,
                                                                 Row,
                                                                 Tuple<BF16>,
                                                                 Tuple<I8, BF16>,
                                                                 Tuple<BF16>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 Add>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances<
            Tuple<Row>,
            Tuple<BF16>,
            Add,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<Tuple<Row>,
                                                                 Tuple<Col, Col>,
                                                                 Tuple<>,
                                                                 Row,
                                                                 Tuple<BF16>,
                                                                 Tuple<I8, BF16>,
                                                                 Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances<
            Tuple<>,
            Tuple<>,
            PassThrough,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<Tuple<Row>,
                                                                 Tuple<Col, Col>,
                                                                 Tuple<>,
                                                                 Row,
                                                                 Tuple<BF16>,
                                                                 Tuple<I8, BF16>,
                                                                 Tuple<>,
                                                                 BF16,
                                                                 PassThrough,
                                                                 Multiply,
                                                                 FastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_wmma_fixed_nk_multi_abd_bf16_i8_bf16_mk_nk_mn_instances<
            Tuple<>,
            Tuple<>,
            FastGelu,
            GemmMNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
