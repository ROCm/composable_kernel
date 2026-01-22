// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/numeric.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::make_ParallelTensorFunctor;
using ::ck::Tensor;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F16;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 1;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr auto ABSpec = ck::tensor_operation::device::TensorSpecialization::Packed;
static constexpr auto DESpec = ck::tensor_operation::device::TensorSpecialization::Default;

// clang-format off
using DeviceOpInstanceKKNN = ck::tensor_operation::device::
        //############################################| NumDimG| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           Gemm|              A|              B|             DE| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################################|        |        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Spacialization| Spacialization| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################################|        |        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |               |               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################################|        |        |        |        |      |      |        |         |           |      |             |            |             |               |               |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceBatchedContractionMultipleD_Xdl_CShuffle< NumDimG, NumDimM, NumDimN, NumDimK,   F16,   F16,     F32,      F16, DsDataType,   F16,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,         ABSpec,         ABSpec,         DESpec,        1,   256,   256,   128,    32,   8,   8,   16,   16,    8,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 32, 1, 4>,               4>;
// clang-format on

using DeviceOpInstance = DeviceOpInstanceKKNN;

#include "run_batched_gemm_bias_e_permute_example.inc"
int main(int argc, char* argv[]) { return !run_batched_gemm_bias_e_permute_example(argc, argv); }
