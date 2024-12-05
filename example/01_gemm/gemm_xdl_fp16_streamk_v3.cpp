// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        64,  // Block Size
        16, // MPer Block
        16, // NPer Block
        64,  // KPer Block
        8,   // AK1
        8,   // BK1
        16,  // MPer XDL
        16,  // NPer XDL
        1,   // Mxdl Per Wave
        1,   // Nxdl Per Wave
        S<8, 8, 1>,  // AblockTransfer ThreadCluster Lenghts_K0_M_kK1
        S<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,   // ABlockTransfer SrcAccessOrder
        2,            // ABlockTransfer SrcVectorDim
        8,            // ABlockTransfer SrcScalar PerVector
        8,            // ABlockTransfer DstScalar PerVector_K1
        0,            // ABlockLds AddExtraM
        S<8, 8, 1>,  // BBlockTransfer ThreadCluster Lengths_K0_N_K1
        S<1, 0, 2>,   // BBlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,   // BlockTransfer  SrcAccessOrder
        2,            // BBlockTransfer SrcVectorDim
        8,            // BBlockTransfer SrcScalar PerVector
        8,            // BBlockTransfer DstScalar PerVector_K1
        0,            // BBlocksLds AddExtraN
        1,            // CShuffle MXdlPerWave PerShuffle
        1,            // CShuffle NXdlPerWave PerShuffle
        S<1, 16, 1, 4>, // CBlockTransferClusterLenghts _MBlock_MXdlPerWave_MWaveMPerXdl _NBlock_NXdlPerWave_NWaveNPerXdl
        4,              // CBlockTransfer ScalarPerVector _NWaveNPerXdl
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on


// // clang-format off
// using DeviceGemmV2_Streamk_Instance = 
//     ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
//         ALayout,   BLayout,  CLayout,   
//         ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
//         PassThrough, PassThrough, PassThrough, GemmDefault, 
//         64,  // Block Size
//         16, // MPer Block
//         16, // NPer Block
//         128,  // KPer Block
//         8,   // AK1
//         8,   // BK1
//         16,  // MPer XDL
//         16,  // NPer XDL
//         1,   // Mxdl Per Wave
//         1,   // Nxdl Per Wave
//         S<16, 4, 1>,  // AblockTransfer ThreadCluster Lenghts_K0_M_kK1
//         S<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder
//         S<1, 0, 2>,   // ABlockTransfer SrcAccessOrder
//         2,            // ABlockTransfer SrcVectorDim
//         8,            // ABlockTransfer SrcScalar PerVector
//         8,            // ABlockTransfer DstScalar PerVector_K1
//         0,            // ABlockLds AddExtraM
//         S<16, 4, 1>,  // BBlockTransfer ThreadCluster Lengths_K0_N_K1
//         S<1, 0, 2>,   // BBlockTransfer ThreadCluster ArrangeOrder
//         S<1, 0, 2>,   // BlockTransfer  SrcAccessOrder
//         2,            // BBlockTransfer SrcVectorDim
//         8,            // BBlockTransfer SrcScalar PerVector
//         8,            // BBlockTransfer DstScalar PerVector_K1
//         0,            // BBlocksLds AddExtraN
//         1,            // CShuffle MXdlPerWave PerShuffle
//         1,            // CShuffle NXdlPerWave PerShuffle
//         S<1, 16, 1, 4>, // CBlockTransferClusterLenghts _MBlock_MXdlPerWave_MWaveMPerXdl _NBlock_NXdlPerWave_NWaveNPerXdl
//         4,              // CBlockTransfer ScalarPerVector _NWaveNPerXdl
//         ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// // clang-format on

#if 0
// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256,
        224, 256, 
        64, 8, 2,
        16,   16,
        7,    8,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<0, 2, 1>,  S<0, 2, 1>, 
        1, 8, 2, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

#endif

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_streamk_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_universal_streamk_example(argc, argv); }
