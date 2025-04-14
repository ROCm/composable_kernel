// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "moe_gemm.hpp"
#include "ck_tile/host/reference/reference_fused_single_moe_gemm.hpp"

namespace {

struct MoeGemmKernelParam
{
    static const bool kPadM = false;
    static const bool kPadN = false;
    static const bool kPadK = false;

    static const int kBlockPerCu         = 1;
    static const ck_tile::index_t M_Tile = 128;
    static const ck_tile::index_t N_Tile = 128;
    static const ck_tile::index_t K_Tile = 32; // need to ensure the M_per_thread = 1

    static const ck_tile::index_t M_Warp = 2;
    static const ck_tile::index_t N_Warp = 2;
    static const ck_tile::index_t K_Warp = 1;

    static const ck_tile::index_t M_Warp_Tile = 32;
    static const ck_tile::index_t N_Warp_Tile = 32;
    static const ck_tile::index_t K_Warp_Tile = 8;
};

using CodegenGemmShape = ck_tile::TileGemmShape<ck_tile::sequence<MoeGemmKernelParam::M_Tile,
                                                                  MoeGemmKernelParam::N_Tile,
                                                                  MoeGemmKernelParam::K_Tile>,
                                                ck_tile::sequence<MoeGemmKernelParam::M_Warp,
                                                                  MoeGemmKernelParam::N_Warp,
                                                                  MoeGemmKernelParam::K_Warp>,
                                                ck_tile::sequence<MoeGemmKernelParam::M_Warp_Tile,
                                                                  MoeGemmKernelParam::N_Warp_Tile,
                                                                  MoeGemmKernelParam::K_Warp_Tile>>;

using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

template <typename ALayout, typename BLayout, typename CLayout>
using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<MoeGemmKernelParam::kPadM,
                                                           MoeGemmKernelParam::kPadN,
                                                           MoeGemmKernelParam::kPadK,
                                                           true,
                                                           ALayout,
                                                           BLayout,
                                                           CLayout>;

template <typename ALayout, typename BLayout, typename CLayout>
using CodegenPipelineProblem =
    ck_tile::UniversalGemmPipelineProblem<ADataType,
                                          BDataType,
                                          AccDataType,
                                          CodegenGemmShape,
                                          CodegenGemmTraits<ALayout, BLayout, CLayout>>;

template <typename ALayout, typename BLayout, typename CLayout>
using CodegenGemmPipeline =
    ck_tile::MoeGemmPipelineAgBgCrImpl<CodegenPipelineProblem<ALayout, BLayout, CLayout>>;

template <typename ALayout, typename BLayout, typename CLayout>
using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
    ADataType,
    BDataType,
    AccDataType,
    CDataType,
    CLayout,
    CodegenPipelineProblem<ALayout, BLayout, CLayout>::kBlockSize,
    TilePartitioner::MPerBlock,
    TilePartitioner::NPerBlock,
    MoeGemmKernelParam::M_Warp,
    MoeGemmKernelParam::N_Warp,
    MoeGemmKernelParam::M_Warp_Tile,
    MoeGemmKernelParam::N_Warp_Tile,
    MoeGemmKernelParam::K_Warp_Tile,
    CodegenPipelineProblem<ALayout, BLayout, CLayout>::TransposeC>>;

// template <typename ALayout, typename BLayout, typename CLayout>
// using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<ck_tile::DefaultGemm2DEpilogueProblem<
//     AccDataType,
//     CDataType,
//     CLayout,
//     MoeGemmKernelParam::kPadM,
//     MoeGemmKernelParam::kPadN,
//     MoeGemmKernelParam::M_Warp_Tile,
//     MoeGemmKernelParam::N_Warp_Tile,
//     MoeGemmKernelParam::K_Warp_Tile,
//     CodegenPipelineProblem<ALayout, BLayout, CLayout>::TransposeC
// >>;

template <typename ALayout, typename BLayout, typename CLayout>
using Kernel = ck_tile::MoeGemmKernel<TilePartitioner,
                                      CodegenGemmPipeline<ALayout, BLayout, CLayout>,
                                      GemmEpilogue<ALayout, BLayout, CLayout>>;
}; // namespace

template <typename ALayout, typename BLayout, typename CLayout>
float moe_gemm(const moe_gemm_kargs& gemm_desc, const ck_tile::stream_config& s)
{
    using MoeGemmKernel = ::Kernel<ALayout, BLayout, CLayout>;

    // TODO: malloc sorted_tokend_ids buffer
    const auto arguments = MoeGemmKernel::MoeGemmKernelArgs::MakeKernelArgs(gemm_desc);

    const dim3 grids      = MoeGemmKernel::GridSize(gemm_desc.M, gemm_desc.N, 1);
    constexpr dim3 blocks = MoeGemmKernel::BlockSize();

    // ck_tile::hip_check_error(hipMemcpyWithStream(
    //     arguments.data(),
    //     arguments.size() * sizeof(typename MoeGemmKernel::MoeGemmKernelArg),
    //     hipMemcpyHostToDevice,
    //     s.stream_id_));

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << MoeGemmKernel::GetName() << " with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time =
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<blocks.x, MoeGemmKernelParam::kBlockPerCu>(
                                   MoeGemmKernel{}, grids, blocks, 0, arguments));
    return ave_time;
}

#include "run_moe_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_moe_gemm_example(argc, argv); }
