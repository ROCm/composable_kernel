// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough         = ck::tensor_operation::element_wise::PassThrough;
using Multiply            = ck::tensor_operation::element_wise::Multiply;
using MultiplyAddFastGelu = ck::tensor_operation::element_wise::MultiplyAddFastGelu;
using MultiplyFastGelu    = ck::tensor_operation::element_wise::MultiplyFastGelu;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;

static constexpr auto GemmDefault    = GemmSpecialization::Default;
static constexpr auto GemmKPadding   = GemmSpecialization::KPadding;
static constexpr auto GemmMNPadding  = GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

template <typename DsLayout,
          typename DsDataType,
          typename CDEElementwiseOp,
          GemmSpecialization GemmSpec = GemmMNKPadding>
using device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_comp_instances = std::tuple<
    // clang-format off
        //###########################################|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|      DsData| EData|           A|           B|                C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //###########################################| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|        Type|  Type| Elementwise| Elementwise|      Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //###########################################|       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //###########################################|       |       |            |       |      |      |        |         |            |      |            |            |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |   S<C,D0...,D_N| 
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   256,   256,    32,   8,   4,   32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   4,   32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   256,   256,    32,   8,   4,   32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   256,   256,    32,   8,   4,   32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   224,   256,    64,   8,   4,   16,   16,    7,    8,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,         0,           1,           2,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   4,   32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   128,   256,    32,   8,   4,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,1>,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   4,   32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              4,         0,           1,           1,               S<1, 32, 1, 8>,        S<8,8,8>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <typename DsLayout,
          typename DsDataType,
          typename CDEElementwiseOp,
          GemmSpecialization GemmSpec                 = GemmMNKPadding,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave>
using device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_mem_instances =
    std::tuple<
        // clang-format off
        //###########################################|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|      DsData| EData|           A|           B|                C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //###########################################| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|        Type|  Type| Elementwise| Elementwise|      Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //###########################################|       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //###########################################|       |       |            |       |      |      |        |         |            |      |            |            |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |   S<C,D0...,D_N| 
        // Latency friendly
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,    64,    16,    16,   256,   8,   4,   16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<64, 1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 4>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    16,    32,   256,   8,   4,   16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<64, 2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
        // Memory friendly
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,    64,    16,    16,   256,   8,   4,   16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<64, 1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 4>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    16,    32,   256,   8,   4,   16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<64, 2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    16,    64,   128,   8,   4,   16,   16,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<32, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    32,    64,   128,   8,   4,   32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<32, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<8,8,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    16,   128,    64,   8,   4,   16,   16,    1,    4,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<16, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   128,    32,   128,    64,   8,   4,   32,   32,    1,    2,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<16, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,               S<1, 16, 1, 8>,        S<8,8,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,    16,   256,    64,   8,   4,   16,   16,    1,    4,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,              S<1, 16, 1, 16>,        S<4,4,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
        // DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<   Row,     Row,    DsLayout,    Row,  BF16,    I8,     F32,      F32,  DsDataType,  BF16, PassThrough, PassThrough, CDEElementwiseOp,       GemmSpec,        1,   256,    32,   256,    64,   8,   4,   32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,             16,              4,          0,          1,           1,              S<1, 16, 1, 16>,        S<8,8,1>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>
        // clang-format on
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
