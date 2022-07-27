// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_reduce_xdl_common.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/reduction_operator.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType         = BF16;
using BDataType         = BF16;
using CDataType         = BF16;
using GemmAccDataType   = F32;
using ReduceAccDataType = F32;
using ReduceDataType    = F32;
using ReducePtrsGlobal  = ck::Tuple<ReduceDataType*, ReduceDataType*>;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;
using ReduceOp0  = ck::reduce::Add;
using ReduceOp1  = ck::reduce::Add;
using ReduceOps  = ck::Tuple<ReduceOp0, ReduceOp1>;

using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
using UnaryDivElementOp     = ck::tensor_operation::element_wise::UnaryDivide;
using UnarySquareElementOp  = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps    = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using ReduceOutElementOps   = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;

using ReduceGlobalMemOps =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmReduceInstance = ck::tensor_operation::device::DeviceGemmReduce_Xdl_CShuffle
        <Row,                       // ALayout
         Col,                       // BLayout
         Row,                       // CLayout
         ADataType,                 // ADataType
         BDataType,                 // BDataType
         CDataType,                 // CDataType
         GemmAccDataType,           // GemmAccDataType
         F32,                       // CShuffleDataType
         ReduceAccDataType,         // ReduceAccDataType
         ReducePtrsGlobal,          // ReduceData Tuple
         AElementOp,                // AElementwiseOperation
         BElementOp,                // BElementwiseOperation
         CElementOp,                // CElementwiseOperation
         ReduceOps,                 // ReduceOperation
         ReduceInElementOps,        // ReduceInEleOp
         ReduceOutElementOps,       // ReduceOutEleOp
         ReduceGlobalMemOps,        // ReduceMemoryDataOperation
         GemmSpecialization,        // GEMM Specialization
         1,                         // NumGemmKPrefetchStage
         256,                       // BlockSize
         256,                       // MPerBlock
         128,                       // NPerBlock
         32,                        // KPerBlock
         8,                         // AK1
         8,                         // BK1
         32,                        // MPerXdl
         32,                        // NPerXdl
         4,                         // MXdlPerWave
         2,                         // NXdlPerWave
         S<4, 64, 1>,               // ABlockTransfer ThreadCluster Lengths_K0_M_K1
         S<1, 0, 2>,                // ABlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // ABlockTransfer SrcAccessOrder
         2,                         // ABlockTransfer SrcVectorDim
         8,                         // ABlockTransfer SrcScalarPerVector
         8,                         // ABlockTransfer DstScalarPerVector_K1
         1,                         // ABlockLdsExtraM
         S<4, 64, 1>,               // BBlockTransfer ThreadCluster Lengths_K0_N_K1
         S<1, 0, 2>,                // BBlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // BBlockTransfer SrcAccessOrder
         2,                         // BBlockTransfer SrcVectorDim
         8,                         // BBlockTransfer SrcScalarPerVector
         8,                         // BBlockTransfer DstScalarPerVector_K1
         1,                         // BBlockLdsExtraN
         1,                         // CShuffleMXdlPerWavePerShuffle
         1,                         // CShuffleNXdlPerWavePerShuffle
         S<1, 32, 1, 8>,            // CBlockTransferClusterLengths _MBlock_MPerBlock_NBlock_NPerBlock
         8,                         // CBlockTransferScalarPerVector_NPerBlock
         S<64, 4>,                  // CReduceThread ClusterLengths _MPerBlock_NPerBlock
         4,                         // CReduceThread Lds2VGprCopy SrcDstScalarPerVector _NPerBlock
         1>;                        // CReduceThread Vgpr2GlobalCopy SrcDstScalarPerVector _MPerBlock
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;
int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    return run_gemm_reduce_mean_squaremean_xdl<ADataType,
                                               BDataType,
                                               CDataType,
                                               ReduceDataType,
                                               ReduceAccDataType,
                                               ALayout,
                                               BLayout,
                                               CLayout,
                                               AElementOp,
                                               BElementOp,
                                               CElementOp,
                                               UnaryIdenticElementOp,
                                               UnarySquareElementOp,
                                               UnaryDivElementOp,
                                               ReduceOp0,
                                               ReduceOp1,
                                               DeviceGemmReduceInstance,
                                               ReferenceGemmInstance>(
        M, N, K, StrideA, StrideB, StrideC, do_verification, init_method, time_kernel);
}
