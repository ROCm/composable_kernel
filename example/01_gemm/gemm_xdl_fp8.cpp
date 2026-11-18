// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/library/utility/run_fp8_example.hpp"

template <typename Fp8, typename Bf8>
struct Example
{

    using ADataType        = Fp8;
    using BDataType        = Fp8;
    using CDataType        = Fp8;
    using AccDataType      = float;
    using CShuffleDataType = float;

    using ALayout = Row;
    using BLayout = Col;
    using CLayout = Row;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = PassThrough;

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr auto LoopSched   = ck::make_default_loop_scheduler();
    static constexpr auto PipelineVer = ck::PipelineVersion::v1;
    using ComputeTypeA                = Fp8;
    using ComputeTypeB                = Fp8;

    // clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
// ######| ALayout| BLayout| CLayout|     AData|     BData|     CData|     AccData|         CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|       Loop|    Pipeline|      Compute|      Compute|
// ######|        |        |        |      Type|      Type|      Type|        Type|         DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|  Scheduler|     Version|        TypeA|        TypeB|
// ######|        |        |        |          |          |          |            |                 |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|           |            |             |             |
// ######|        |        |        |          |          |          |            |                 |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |           |            |             |             |
         < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,        1,   256,   256,   128,    64,  16,  16,   16,   16,    8,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               4,  LoopSched, PipelineVer, ComputeTypeA, ComputeTypeB>;
    // this instance has been tested working on gfx950     
    //     < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  CElementOp,    GemmDefault,        1,   256,   256,   128,    128,  32,  32,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8,  LoopSched, PipelineVer, ComputeTypeA, ComputeTypeB>;
    // clang-format on

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                            BDataType,
                                                                            CDataType,
                                                                            AccDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CElementOp>;

    using ReferenceComputeType = float;
    using ReferenceGemmInstanceGPU =
        ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    ADataType,
                                                    BDataType,
                                                    CDataType,
                                                    AccDataType,
                                                    AElementOp,
                                                    BElementOp,
                                                    CElementOp,
                                                    ReferenceComputeType,
                                                    ReferenceComputeType>;

#include "run_gemm_example.inc"

    int Run(int argc, char* argv[]);
};

template <typename Fp8, typename Bf8>
int Example<Fp8, Bf8>::Run(int argc, char* argv[])
{
    if(ck::is_gfx11_supported() || ck::is_gfx117_supported())
    {
        return 0;
    }

    return !run_gemm_example(argc, argv);
}

int main(int argc, char* argv[]) { return ck::utils::run_fp8_example<Example>(argc, argv); }
