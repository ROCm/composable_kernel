// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp"

using BF16 = ck::bhalf_t;
using F32  = float;

using ADataType   = BF16;
using BDataType   = BF16;
using CDataType   = BF16;
using AccDataType = F32;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
    <ALayout,                    // typename ALayout
     BLayout,                    // typename BLayout
     CLayout,                    // typename CLayout
     ADataType,                  // typename ADataType
     BDataType,                  // typename BDataType
     CDataType,                  // typename CDataType
     AccDataType,                // typename GemmAccDataType
     CDataType,                  // typename CShuffleDataType
     PassThrough,                // typename AElementwiseOperation
     PassThrough,                // typename BElementwiseOperation
     PassThrough,                // typename CElementwiseOperation
     GemmDefault,                // GemmSpecialization GemmSpec
     1,                          // index_t NumGemmKPrefetchStage
     256,                        // index_t BlockSize
     256,                        // index_t MPerBlock
     128,                        // index_t NPerBlock
     32,                         // index_t KPerBlock
     8,                          // index_t AK1
     8,                          // index_t BK1
     32,                         // index_t MPerXDL
     32,                         // index_t NPerXDL
     4,                          // index_t MXdlPerWave
     2,                          // index_t NXdlPerWave
     S<4, 64, 1>,                // typename ABlockTransferThreadClusterLengths_AK0_M_AK1
     S<1, 0, 2>,                 // typename ABlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename ABlockTransferSrcAccessOrder
     2,                          // index_t ABlockTransferSrcVectorDim
     8,                          // index_t ABlockTransferSrcScalarPerVector
     8,                          // index_t ABlockTransferDstScalarPerVector_AK1
     1,                          // index_t ABlockLdsExtraM
     S<4, 64, 1>,                // typename BBlockTransferThreadClusterLengths_BK0_N_BK1
     S<1, 0, 2>,                 // typename BBlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename BBlockTransferSrcAccessOrder
     2,                          // index_t BBlockTransferSrcVectorDim
     8,                          // index_t BBlockTransferSrcScalarPerVector
     8,                          // index_t BBlockTransferDstScalarPerVector_BK1
     1,                          // index_t BBlockLdsExtraN
     1,                          // index_t CShuffleMXdlPerWavePerShuffle
     1,                          // index_t CShuffleNXdlPerWavePerShuffle
     S<1, 32, 1, 8>,             // typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
     8>;                         // index_t CShuffleBlockTransferScalarPerVector_NPerBlock
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
