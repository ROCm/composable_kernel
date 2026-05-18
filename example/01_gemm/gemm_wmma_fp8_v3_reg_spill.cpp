// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * \brief Example of GEMM using WMMA that illustrates register spilling on gfx1200 architecture but
 * no register spilling on gfx1250.
 *
 * This example demonstrates how more registers available on the gfx1250 architecture can help avoid
 * register spilling that occurs on gfx1200 when using a specific GEMM configuration.
 *
 * This example must be compiled with the following flag to see the resource allocations:
 *  "-Rpass-analysis=kernel-resource-usage"
 *
 * On gfx1200, the kernel will show register spilling due to limited VGPRs:
 * \verbatim
 * TotalSGPRs: 105
 * VGPRs: 256
 * ScratchSize [bytes/lane]: 56
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 5
 * SGPRs Spill: 0
 * VGPRs Spill: 15
 * LDS Size [bytes/block]: 32768
 *
 * gfx1201 - AMD Radeon RX 9070 XT
 * Problem {M:3840, N:4096, K:4096, SA:4096, SB:4096, SC:4096, MP:3840, NP:4096, KRead:4096,
 * KP:4096, AK0:512, BK0:512, MBlock: 30, NBlock: 32}
 *
 * Perf: 0.882764 ms, 145.961 TFlops, 72.4578 GB/s, DeviceGemm_Wmma_CShuffleV3<Default, RCR>
 * BlkSize: 128, BlkTile: 128x128x128, WaveTile: 16x16, WaveMap: 4x4, VmemReadVec: 8x8,
 * BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, BlkGemmPipelinePrefetchStages:
 * 1, KPack: 16
 * \endverbatim
 *
 * On gfx1250, the same kernel will not show register spilling due to increased VGPRs:
 * \verbatim
 * TotalSGPRs: 32
 * VGPRs: 318
 * ScratchSize [bytes/lane]: 0
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 3
 * SGPRs Spill: 0
 * VGPRs Spill: 0
 * LDS Size [bytes/block]: 32768
 * \endverbatim
 *
 * \note The register allocations above can be influenced by compiler version and code
 * changes/optimizations.
 */

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"

using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using ComputeTypeA     = ck::f8_t;
using ComputeTypeB     = ck::f8_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    PassThrough, PassThrough, PassThrough, GemmDefault,
    128, //blocksize
    128, 128, 128, // M/N/KPerBlock
    8, 8, // AK1, BK1
    16, 16, //MPerWmma, NPerWmma
    4, 4, //MRepeat, NRepeat
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>,//
    2, 8, 8, 0,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 0,
    1, 1, S<1, 32, 1, 4>, 8,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1,
    ComputeTypeA, ComputeTypeB>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp,
                                                                        ComputeTypeA,
                                                                        ComputeTypeB>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This kernel support gfx12 only" << std::endl;

        return 0;
    }
    return !run_gemm_splitk_example(argc, argv);
}
