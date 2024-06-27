// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_tile_loop.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/utility/loop_scheduler.hpp"

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

using Empty_Tuple = ck::Tuple<>;
using ck::tensor_operation::device::GemmSpecialization;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::PipelineVersion Pipeline,
          LoopScheduler Scheduler = LoopScheduler::Default>
using device_ggemm_md_splitk_xdl_cshuffle_f16_f16_f16_mk_kn_mn_memory_instances = std::tuple<
    // clang-format off
        //#########################################|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|      DsData| EData|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|     PipelineVersion| LoopScheduler|
        //#########################################| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|        Type|  Type| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|                    |              |
        //#########################################|       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|                    |              |
        //#########################################|       |       |            |       |      |      |        |         |            |      |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                    |              |
        // Memory friendly
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   256,   256,    32,    64,   8,   8,   32,   32,    2,    1,   S<8, 32, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 32, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   256,   256,    16,    64,   8,   8,   16,   16,    4,    1,   S<8, 32, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 32, 1, 8>,               2,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,   128,    32,    64,   8,   8,   32,   32,    2,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,   128,    16,    64,   8,   8,   16,   16,    4,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               2,  Pipeline, Scheduler>,
        
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    64,    32,    64,   8,   8,   32,   32,    1,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    64,    16,    64,   8,   8,   16,   16,    2,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               2,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    32,    16,    64,   8,   8,   16,   16,    1,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               2,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,    64,    16,    16,   128,   8,   8,   16,   16,    1,    1,   S<16, 4, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<16, 4, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 4>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,    64,    16,    16,    64,   8,   8,   16,   16,    1,    1,   S<8,  8, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8,  8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 4>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    16,    32,    64,   8,   8,   16,   16,    1,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    16,    64,    64,   8,   8,   16,   16,    1,    2,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    32,    64,    64,   8,   8,   32,   32,    1,    1,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               8,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    16,   128,    64,   8,   8,   16,   16,    1,    4,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   128,    32,   128,    64,   8,   8,   32,   32,    1,    2,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 8>,               8,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   256,    16,   256,    64,   8,   8,   16,   16,    1,    4,   S<8, 16, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 16>,              4,  Pipeline, Scheduler>,
        DeviceGroupedGemmMultipleDSplitKXdlCShuffle<  Row,     Col,  Empty_Tuple,     Row,     F16,   F16,  F32,   F32,  Empty_Tuple,   F16,    PassThrough, PassThrough, PassThrough,  GemmSpec, NumPrefetch,   256,    32,   256,    64,   8,   8,   32,   32,    1,    2,   S<8, 32, 1>,  S<1, 0, 2>, S<1, 0, 2>,              2,              8,              8,         1,  S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,             2,              8,              8,          1,          1,           1,               S<1, 16, 1, 16>,              8,  Pipeline, Scheduler>
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
