// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_mx_common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"

using F4 = ck::f4x2_pk_t;
using F8 = ck::f8_t;

#if defined(A_DATATYPE)
using ADataType = A_DATATYPE;
#else
using ADataType = F4;
#endif
#if defined(B_DATATYPE)
using BDataType = B_DATATYPE;
#else
using BDataType = F4;
#endif

using XDataType       = ck::e8m0_bexp_t;
using XPackedDataType = int32_t;

using CDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = CDataType;

using ALayout = Row;
using BLayout = MFMA;
using CLayout = Row;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t DataPackedSize =
    ck::packed_size_v<ADataType>;                            // Packed representation of data
constexpr ck::index_t ScaleBlockSize = 32;                   // scaling block size
constexpr ck::index_t KPerBlock      = 256 / DataPackedSize; // 256 f4 = 128 fp4x2

constexpr auto GemmSpec      = ck::tensor_operation::device::GemmSpecialization::Default;
constexpr auto BlkGemmPSched = ck::BlockGemmPipelineScheduler::Intrawave;
constexpr auto BlkGemmPVer   = ck::BlockGemmPipelineVersion::v3;

// AB DataType: f4x2_pk_t
// Mathematically, all numbers are represented as f4x2.
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    ALayout,          // ALayout
    BLayout,          // BLayout
    CLayout,          // CLayout
    ADataType,        // ADataType
    XPackedDataType,  // AScaleDataType
    BDataType,        // BDataType
    XPackedDataType,  // BScaleDataType
    CDataType,        // CDataType
    AccDataType,      // GemmAccDataType
    CShuffleDataType, // CShuffleDataType
    AElementOp,       // AElementwiseOperation
    BElementOp,       // BElementwiseOperation
    CElementOp,       // CElementwiseOperation
    GemmSpec,         // GemmSpec
    ScaleBlockSize,   // ScaleBlockSize: Scaling block size
    128,              // BlockSize: Thread block size
    64,               // MPerBlock
    64,               // NPerBlock
    KPerBlock,        // KPerBlock
    16,               // AK1
    16,               // BK1
    16,               // MPerXDL
    16,               // NPerXDL
    2,                // MXdlPerWave
    2,                // NXdlPerWave
    S<8, 16, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
    2,                // ABlockTransferSrcVectorDim
    16,               // ABlockTransferSrcScalarPerVector
    16,               // ABlockTransferDstScalarPerVector_AK1
    true,             // ABlockLdsExtraM
    S<8, 16, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
    2,                // BBlockTransferSrcVectorDim
    16,               // BBlockTransferSrcScalarPerVector
    16,               // BBlockTransferDstScalarPerVector_BK1
    true,             // BBlockLdsExtraN
    2,                // CShuffleMXdlPerWavePerShuffle
    2,                // CShuffleNXdlPerWavePerShuffle
    S<1, 8, 1, 8>,    // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    2,                // CShuffleBlockTransferScalarPerVector_NPerBlock
    BlkGemmPSched,    // BlkGemmPipeSched
    BlkGemmPVer,      // BlkGemmPipelineVer
    ADataType,        // ComputeTypeA
    BDataType         // ComputeTypeB
    >;

int main(int argc, char* argv[])
{
    return run_mx_gemm_example<DeviceOpInstance,
                               ADataType,
                               BDataType,
                               XDataType,
                               XPackedDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               CLayout,
                               AElementOp,
                               BElementOp,
                               CElementOp,
                               AccDataType,
                               CShuffleDataType,
                               ScaleBlockSize>(argc, argv)
               ? 0
               : -1;
}
