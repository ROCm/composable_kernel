// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_splitK_gemv.hpp"

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

// static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;



#define K1 8 //2,4,8
#define K0 4 //1,2,3,4...32 --> 423ms vs 282ms (k0=4)
#define N1 2 //2,4,8
#define B 64 //64

// clang-format off
using DeviceGemvInstance = ck::tensor_operation::device::deviceGemvDl/*
// ######|     AData|     BData|     CData|     AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer  |     ABlockTransfer|      ABlockTransfer  | BBlockTransfer|  BThreadTransfer|    BThreadTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
// ######|      Type|      Type|      Type|        Type|        |        |        | Elementwise| Elementwise| Elementwise| Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|SrcVectorTensorLengths|    SrcVectorTensor|DstVectorTensorLengths|      SrcAccess|     SrcVectorDim| SrcScalarPerVector|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
// ######|          |          |          |            |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |           |           |       | KBatch_K0_M0_M1_K1|   KBatch_K0_M0_M1_K1|   ArrangeOrder|          Order| KBatch_K0_M0_M1_K1   | ContiguousDimOrder| KBatch_K0_M0_M1_K1   |          Order|                 |                   |               Order|                |                   |
// ######|          |          |          |            |        |        |        |            |            |            |               |      |      |      |      |   |           |           |       |                   |                     |               |               |                      |                   |                      |               |                 |                   |                    |                |                   |
       //< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,  GemmMNPadding,    64,     1,    64,    32,  2,          1,          1,      1,      S<1, 1, 1, 2>,      S<32, 1,  1, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,    S<1, 2, 0, 3>,             3,               2,         S<0, 1, 2, 3, 4, 5>,               5,                  1>;*/
      // < ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,  GemmMNPadding,    64,     1,    128,    8,  8,          1,          2,      1,     S<1,1, 1, 1, 8>,    S<1,8, 1,  1, 1>, S<0,1,2,3,4>,  S<0,1,2,3,4>,      S<1,1, 1, 1, 8>,      S<0,1,2,3,4>,     S<1,1, 1, 1, 2>,    S<0,1,2,3,4>,                4,               8,        S<0, 1, 2, 3, 4, 5>,              5,                  2>;
         < ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,  GemmMNPadding,    B,     1,    B*N1,   K0,  K1,         1,          N1,      1,    S<1,1, 1, 1, K1>,   S<1,K0, 1,  1, 1>,S<0,1,2,3,4>,  S<0,1,2,3,4>,      S<1,1, 1, 1, K1>,     S<0,1,2,3,4>,     S<1,1, 1, 1, 2>,    S<0,1,2,3,4>,                4,               K1,        S<0, 1, 2, 3, 4, 5>,             5,                  N1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_splitK_gemv_example.inc"

int main(int argc, char* argv[]) { return !run_gemv_example(argc, argv); }
