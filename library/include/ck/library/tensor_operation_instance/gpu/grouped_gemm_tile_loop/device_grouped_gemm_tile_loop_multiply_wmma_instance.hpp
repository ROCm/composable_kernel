// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstdlib>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/utility/loop_scheduler.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I8   = int8_t;
using F8   = ck::f8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

using CShuffleDataType = F32;
using AccDataType      = F32;
using ELayout          = Row;

static constexpr auto PipelineV1         = BlockGemmPipelineVersion::v1;
static constexpr auto PipelineV3         = BlockGemmPipelineVersion::v3;
static constexpr auto IntrawaveScheduler = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto InterwaveScheduler = BlockGemmPipelineScheduler::Interwave;
static constexpr auto GemmKPadding       = device::GemmSpecialization::KPadding;
static constexpr auto GemmMNPadding      = device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding     = device::GemmSpecialization::MNKPadding;
static constexpr auto GemmDefault        = device::GemmSpecialization::Default;

// Instances for 2 byte * 1 byte datatypes in RRR layout, with EDataType = ADataType
// HACK: CBlockTransfer_ScalarPerVector_NRepeat elements should depend on the amount and data types
// in the D tensors. In practice, D tensors are 2 bytes and there's never more than two. So this
// works, but isn't very robust.
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename DsLayout,
          device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          enable_if_t<sizeof(ADataType) == 2, bool> = false,
          enable_if_t<sizeof(BDataType) == 1, bool> = false>
using device_grouped_gemm_tile_loop_multiply_wmma_mk_kn_mn_instances = std::tuple<
    // clang-format off
        //#################################################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#################################################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //#################################################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //#################################################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
     // DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType,        ADataType, DsDataType, ADataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,      S<8, 8, 8>, BlkGemmPipeSched, BlkGemmPipelineVer>,
     // DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType,        ADataType, DsDataType, ADataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              2,         1,           1,           1,               S<1, 64, 1, 4>,      S<8, 8, 8>, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemmMultipleD_Wmma_CShuffle_TileLoop_V3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, ADataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              1,         1,           1,           1,               S<1, 64, 1, 4>,      S<8, 8, 8>, BlkGemmPipeSched, BlkGemmPipelineVer>
    // clang-format on
    >;

static constexpr device::GemmSpecialization GemmSpecVariants[] = {GemmDefault, GemmMNKPadding};

// Helper function to add a list of layout instances for instances with matching A/B/E data types
// for all supported padding/scheduler/pipeline version combinations
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          template <typename ADataType_inner,
                    typename BDataType_inner,
                    typename DsDataTyper_inner,
                    typename DsLayout_inner,
                    device::GemmSpecialization GemmSpec,
                    BlockGemmPipelineScheduler BlkGemmPipeSched,
                    BlockGemmPipelineVersion BlkGemmPipelineVer,
                    typename AElementOp,
                    typename BElementOp,
                    typename CDEElementOp>
          typename LayoutInstances,
          typename AElementOp, // NOTE: element-wise op parameters as last so that they can be
          typename BElementOp, // inferred from the vector argument
          typename CDEElementOp>
void add_device_grouped_gemm_tile_loop_multiply_wmma_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<ALayout,
                                                          BLayout,
                                                          DsLayout,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          DsDataType,
                                                          ADataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CDEElementOp>>>& instances)
{
    static_for<0, std::size(GemmSpecVariants), 1>{}([&](auto i) {
        constexpr auto GemmSpec = GemmSpecVariants[i];

        add_device_operation_instances(instances,
                                       LayoutInstances<ADataType,
                                                       BDataType,
                                                       DsDataType,
                                                       DsLayout,
                                                       GemmSpec,
                                                       IntrawaveScheduler,
                                                       PipelineV1,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>{});
        add_device_operation_instances(instances,
                                       LayoutInstances<ADataType,
                                                       BDataType,
                                                       DsDataType,
                                                       DsLayout,
                                                       GemmSpec,
                                                       InterwaveScheduler,
                                                       PipelineV1,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>{});
        add_device_operation_instances(instances,
                                       LayoutInstances<ADataType,
                                                       BDataType,
                                                       DsDataType,
                                                       DsLayout,
                                                       GemmSpec,
                                                       IntrawaveScheduler,
                                                       PipelineV3,
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>{});
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
