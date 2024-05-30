// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_tall_and_skinny_gemm_splitk.hpp"

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = Row;
using BLayout = Row; // Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// #define K1 2
// #define K0 4
// #define N1 2
// #define B 64 // block-size:64
// #define M1 16

#define K1 2
#define K0 4
#define N1 2
#define B 64 // block-size:64
#define M1 16

// clang-format off
using DeviceTSMMInstance = ck::tensor_operation::device::deviceTsmmDl/*
// ######|     AData|     BData|     CData|     AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer  |     ABlockTransfer|      ABlockTransfer  | BBlockTransfer|  BThreadTransfer|    BThreadTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
// ######|      Type|      Type|      Type|        Type|        |        |        | Elementwise| Elementwise| Elementwise| Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|SrcVectorTensorLengths|    SrcVectorTensor|DstVectorTensorLengths|      SrcAccess|     SrcVectorDim| SrcScalarPerVector|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
// ######|          |          |          |            |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |           |           |       | KBatch_K0_M0_M1_K1|   KBatch_K0_M0_M1_K1|   ArrangeOrder|          Order| KBatch_K0_M0_M1_K1   | ContiguousDimOrder| KBatch_K0_M0_M1_K1   |          Order|                 |                   |               Order|                |                   |
// ######|          |          |          |            |        |        |        |            |            |            |               |      |      |      |      |   |           |           |       |                   |                     |               |               |                      |                   |                      |               |                 |                   |                    |                |                   |
       //< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,  GemmMNPadding,    64,     1,    64,    32,  2,          1,          1,      1,      S<1, 1, 1, 2>,      S<32, 1,  1, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,    S<1, 2, 0, 3>,             3,               2,         S<0, 1, 2, 3, 4, 5>,               5,                  1>;*/
         < ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,  GemmMNPadding,    B,     M1,   B*N1,   K0,  K1,         M1,         N1,     1,   S<1,1, 1, 1, K1>,   S<1,K0, 1,M1, 1>,  S<0,1,2,3,4>,   S<0,1,2,3,4>,      S<1,1, 1, 1, K1>,     S<0,1,2,3,4>,     S<1,1, 1, 1, K1>,      S<0,1,2,3,4>,                3,               N1,        S<0, 1, 2, 3, 4, 5>,             5,                  N1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_tall_and_skinny_gemm_splitk_example.inc"

int main(int argc, char* argv[]) { return !run_tall_and_skinny_gemm_example(argc, argv); }

