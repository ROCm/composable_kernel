// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8   = f8_t;
using BF16 = bhalf_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using PassThrough      = element_wise::PassThrough;
using MultiplyMultiply = element_wise::MultiplyMultiply;

static constexpr auto GemmDefault = GemmSpecialization::Default;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;

static constexpr auto v1 = BlockGemmPipelineVersion::v1;

template <GemmSpecialization GemmSpec>
using device_gemm_multiply_multiply_weight_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_instances =
    std::tuple<
        // clang-format off
        //###########################################| ALayout| BLayout|         DsLayout| ELayout|AData| BData|          DsData| EData| AccData| Cshuffle|           A|           B|                C|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|   CShuffle|   CShuffle|   CDEBlockTransferClusterLengths|  CDEBlockTransfer|                         BlkGemmPipeSched| Block-wiseGemm|
        //###########################################|        |        |                 |        | Type|  Type|            Type|  Type|    Type|     Type| Elementwise| Elementwise|      Elementwise| Specialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|    MRepeat|    NRepeat|                _MBlock_MPerBlock|   ScalarPerVector|                                         |       Pipeline|
        //###########################################|        |        |                 |        |     |      |                |      |        |         |   Operation|   Operation|        Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          | PerShuffle| PerShuffle|                _NBlock_NPerBlock|     _NWaveNPerXdl|                                         |       Verision|
        //###########################################|        |        |                 |        |     |      |                |      |        |         |            |            |                 |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |           |           |                                 |                  |                                         |               |
        DeviceGemmMultiD_Wmma_CShuffle_V3_BPreshuffle<     Row,     Col,  Tuple<Row, Col>,     Row,   F8,    F8, Tuple<F32, F32>,  BF16,     F32,      F32, PassThrough, PassThrough, MultiplyMultiply,       GemmSpec,   256,    32,   128,   128,  16,  16,   16,   16,       2,       1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         0,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         0,          1,          1,                   S<1, 8, 1, 32>,        S<4, 4, 1>,    BlockGemmPipelineScheduler::Intrawave,             v1, F8>
        // clang-format on
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
