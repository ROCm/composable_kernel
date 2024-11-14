// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp"

using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// // clang-format off
// using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
// // ######| ALayout| BLayout| CLayout|     AData|     BData|     CData|     AccData|         CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
// // ######|        |        |        |      Type|      Type|      Type|        Type|         DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
// // ######|        |        |        |          |          |          |            |                 |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
// // ######|        |        |        |          |          |          |            |                 |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
//          < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>;
// // clang-format on

#if 0
// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256, // Block Size
        128, // MPer Block
        128, // NPer Block
        64,  // KPer Block
        8,   // AK1
        8,   // BK1  float4 float8
        16,  // MPer XDL
        16,  // NPer XDL
        4,   // MXdl Per Wave 
        4,   // NXdl Per Wave
        S<8, 32, 1>,  // ABlockTransfer ThreadCluster Lengths_K0_M_K1
        S<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,   // ABlockTransfer SrcAccessOrder
        2, // ABlockTransfer SrcVectorDim
        8, // ABlockTransfer SrcScalar PerVector
        8, // ABlockTransfer DstScalar PerVector_K1
        0,  // ABlockLds AddExtraM
        S<8, 32, 1>, // BBlockTransfer ThreadCluster Lengths_K0_N_K1
        S<1, 0, 2>,  // BBlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,  // BBlockTransfer SrcAccessOrder
        2,  // BBlockTransfer SrcVectorDim
        8,  // BBlockTransfer SrcScalar PerVector 
        8,  // BlockTransfer DstScalar PerVector_k1
        0,  // B BlockLdsAddExtraN
        1,  // CShuffle MXdlPerWave PerShuffle
        2,  // CShuffle NXdlPerWave
        S<1, 32, 1, 8>, // CBlockTransferClusterLengths _MBlock_MWaveMPerXdl  _NBlock_NWaveNPerXdl
        8,  // CBlockTransfer ScalarPerVector _NWaveNPerXdl
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

#endif

#if 1
// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        64, // Block Size
        16, // MPer Block
        16, // NPer Block
        64,  // KPer Block
        8,   // AK1
        8,   // BK1  float4 float8
        16,  // MPer XDL
        16,  // NPer XDL
        1,   // MXdl Per Wave 
        1,   // NXdl Per Wave
        S<8, 8, 1>,  // ABlockTransfer ThreadCluster Lengths_K0_M_K1
        S<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,   // ABlockTransfer SrcAccessOrder
        2, // ABlockTransfer SrcVectorDim
        8, // ABlockTransfer SrcScalar PerVector
        8, // ABlockTransfer DstScalar PerVector_K1
        0,  // ABlockLds AddExtraM
        S<8, 8, 1>, // BBlockTransfer ThreadCluster Lengths_K0_N_K1
        S<1, 0, 2>,  // BBlockTransfer ThreadCluster ArrangeOrder
        S<1, 0, 2>,  // BBlockTransfer SrcAccessOrder
        2,  // BBlockTransfer SrcVectorDim
        8,  // BBlockTransfer SrcScalar PerVector 
        8,  // BlockTransfer DstScalar PerVector_k1
        0,  // B BlockLdsAddExtraN
        1,  // CShuffle MXdlPerWave PerShuffle
        1,  // CShuffle NXdlPerWave
        S<1, 16, 1, 4>, // CBlockTransferClusterLengths _MBlock_MWaveMPerXdl  _NBlock_NWaveNPerXdl
        4,  // CBlockTransfer ScalarPerVector _NWaveNPerXdl
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

#endif

#if 0
// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 

        // Tiling Parameters - How to partition 'Block tiling - wave tiling' 
        64, // Block Size
        16, // MPer Block
        16, // NPer Block
        64,  // KPer Block
        8,   // AK1  :: 
        8,   // BK1  float4 float8
        16,  // MPer XDL
        16,  // NPer XDL
        1,   // MXdl Per Wave 
        1,   // NXdl Per Wave

        // For Tensor A these define how to copy data from Global to Shared Mem
        S<8,8,1> ,  //S<8, 32, 1>,  // ABlockTransfer ThreadCluster Lengths_K0_M_K1

        S<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder  !!!!! Determined by Layout
        S<1, 0, 2>,   // ABlockTransfer SrcAccessOrder  !!!!!! Determined by Layout  , Always 1-0-2 If A is row major  ,

        // ABlockTransfer ThreadCluster Lengths_K0_M_K1 S<8,32,1> :: Calculation : First Number 8 =  (KPerBlock) / ABlockTransfer SrcScalar PerVector (row-col-row) ! A Tensor is row major
        // Calculation Second Number 32 = ( BlockSize ) / ( FirstNumber (8)  ) !!! = 8
        // Caldulation Third Number = 1 

        2, // ABlockTransfer SrcVectorDim  !! If A is row major this is always 2
        8, // ABlockTransfer SrcScalar PerVector     // How you read 'A tensor' data from global memory 
        8, // ABlockTransfer DstScalar PerVector_K1  // How you write 'A tensor'  data to shared memory
        0,  // ABlockLds AddExtraM
        // Tensor A


        // For Tensor B these define how to copy data from Global to Shared Mem
        S<8, 32, 1>, // BBlockTransfer ThreadCluster Lengths_K0_N_K1
        S<1, 0, 2>,  // BBlockTransfer ThreadCluster ArrangeOrder   Always 1-0-2 If B is col major 
        S<1, 0, 2>,  // BBlockTransfer SrcAccessOrder   Always 1-0-2 If B is col major

        2,  // BBlockTransfer SrcVectorDim !! If B is column major this is always 2 
        8,  // BBlockTransfer SrcScalar PerVector 
        8,  // BlockTransfer DstScalar PerVector_k1
        0,  // B BlockLdsAddExtraN
        // Tensor B

        // How we write final results from registers (vgpr) to GLobal for C Tensor ,  vgpr to global mem

        // Partila Tile size M = MPerblock/(MXdlPerWave/Cshuffle_ MXdlPerWave_PerShuffle)
        // Partila Tile size N = NPerblock/(NXdlPerWave/Cshuffle_ NXdlPerWave_PerShuffle) 16/

        // How many
        // Determine partial tile size for writing results
        1,  // CShuffle MXdlPerWave PerShuffle  :: 
        2,  // CShuffle NXdlPerWave  2 OR 1 it depens on kernel sometimes only 1

        S<1, 32, 1 , 8> , //S<1, 32, 1, 8>, // CBlockTransferClusterLengths _MBlock_MWaveMPerXdl  _NBlock_NWaveNPerXdl
        // First Number = Third Number = 1
        // Fourth Number Line130: fourth number: Partial Tile size N/Line134,    PartialTileSize_N / (CBlockTransfer ScalarPerVector _NWaveNPerXdl)
        // Second Number second number: min(BlockSize/ fourth number, Partial tile size M)
        
        8,  // CBlockTransfer ScalarPerVector _NWaveNPerXdl :: 16/sizeof(CDataType)   , 16 byte "largest data per instruction read/write"

        // Which Optimization for Kernel Software Pipeline 
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

#endif

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_streamk_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_universal_streamk_example(argc, argv); }
