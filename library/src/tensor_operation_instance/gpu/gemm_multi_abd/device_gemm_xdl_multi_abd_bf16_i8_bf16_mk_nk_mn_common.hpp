// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"

#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = BF16;
using AsDataType       = ck::Tuple<A0DataType>;
using B0DataType       = I8;
using B1DataType       = BF16;
using BsDataType       = ck::Tuple<B0DataType, B1DataType>;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using D0DataType       = BF16;
// using DsDataType       = ck::Tuple<D0DataType>;
using EDataType = BF16;

using A0Layout = Row;
using AsLayout = ck::Tuple<A0Layout>;
using B0Layout = Col;
using B1Layout = B0Layout;
using BsLayout = ck::Tuple<B0Layout, B1Layout>;
using D0Layout = Row;
// using DsLayout = ck::Tuple<D0Layout>;
using ELayout = Row;

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;
using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp = PassThrough;
using BElementOp = Multiply;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

// using CDEElementOp = AddFastGelu;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
template <typename DsLayout,
          typename DsDataType,
          typename CDEElementOp,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched>
using device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_comp_instances = std::tuple<
    // clang-format off
        //###############################|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer| K0Per| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //###############################|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //###############################|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //###############################|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
	//Compute-bound
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,  16,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,            16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,                8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   224,   256,    64,   8,  16,  16,   16,    7,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,            16,             16,          0,          1,           2,                   S<1, 32, 1, 8>,                8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,  16,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,            16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,                8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,  16,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,            16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,                8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,  16,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,            16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,                8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;
template <typename DsLayout,
          typename DsDataType,
          typename CDEElementOp,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched>
using device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_mem_instances = std::tuple<
    // clang-format off
        //###############################|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer| K0Per| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //###############################|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //###############################|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //###############################|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        // Latency friendly 
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    32,   16,   128,   8,   16,  16,   16,    1,    1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,    64,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,    64,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    16,   32,   128,   8,   16,  16,   16,    1,    1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
        // Memory friendly
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,   128,   32,   128,   8,   16,  32,   32,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,   128,   16,   128,   8,   16,  16,   16,    4,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    64,   32,   128,   8,   16,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    64,   16,   128,   8,   16,  16,   16,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    32,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,    64,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,    64,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    16,   32,   128,   8,   16,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    16,   64,   128,   8,   16,  16,   16,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    32,   64,   128,   8,   16,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    16,  128,   128,   8,   16,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    32,  128,   128,   8,   16,  32,   32,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,    16,  256,   128,   8,   16,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 16>,              4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>,
       DeviceGemmMultipleABD_Xdl_CShuffle< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,    32,  256,   128,   8,   16,  32,   32,    1,    2,     S<16, 16, 1>,    S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 16>,              8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v2>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
