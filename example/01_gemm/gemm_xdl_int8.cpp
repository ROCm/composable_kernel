// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp"

using ADataType        = int8_t;
using BDataType        = int8_t;
using CDataType        = int8_t;
using AccDataType      = int32_t;
using CShuffleDataType = int8_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
     ALayout,                    // typename ALayout
     BLayout,                    // typename BLayout
     CLayout,                    // typename CLayout
     ADataType,                  // typename ADataType
     BDataType,                  // typename BDataType
     CDataType,                  // typename CDataType
     AccDataType,                // typename GemmAccDataType
     CShuffleDataType,           // typename CShuffleDataType
     PassThrough,                // typename AElementwiseOperation
     PassThrough,                // typename BElementwiseOperation
     PassThrough,                // typename CElementwiseOperation
     GemmDefault,                // GemmSpecialization GemmSpec
     1,                          // index_t NumGemmKPrefetchStage
     256,                        // index_t BlockSize
     256,                        // index_t MPerBlock
     128,                        // index_t NPerBlock
     64,                         // index_t KPerBlock
     16,                         // index_t AK1
     16,                         // index_t BK1
     32,                         // index_t MPerXDL
     32,                         // index_t NPerXDL
     4,                          // index_t MXdlPerWave
     2,                          // index_t NXdlPerWave
     S<4, 64, 1>,                // typename ABlockTransferThreadClusterLengths_AK0_M_AK1
     S<1, 0, 2>,                 // typename ABlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename ABlockTransferSrcAccessOrder
     2,                          // index_t ABlockTransferSrcVectorDim
     16,                         // index_t ABlockTransferSrcScalarPerVector
     16,                         // index_t ABlockTransferDstScalarPerVector_AK1
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
     S<1, 64, 1, 4>,             // typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
     16>;                        // index_t CShuffleBlockTransferScalarPerVector_NPerBlock
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
