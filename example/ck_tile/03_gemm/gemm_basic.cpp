// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_basic.hpp"

#if !defined(EXAMPLE_USE_UNIVERSAL_GEMM_PIPELINE_POLICY)
#define EXAMPLE_USE_UNIVERSAL_GEMM_PIPELINE_POLICY 1
#endif

template <typename ALayout, typename BLayout, typename CLayout>
std::pair<float, std::string> gemm_calc(const gemm_basic_args& args,
                                        const ck_tile::stream_config& s)
{
    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA        = true;
    constexpr bool kPadB        = true;
    constexpr bool kPadC        = true;
    constexpr bool kTilePermute = false;
    // The rank and permutation will also be generate out by the CodeGen part.
    constexpr ck_tile::index_t kOutputRank = 2;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    // Whether doing the CShuffle (transpose before the global memory), depending on the output
    // layout.
    constexpr bool CShuffleEpilogue =
        std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTilePartitioner<CodegenGemmShape>;

    using GemmEpilogue = std::conditional_t<
        CShuffleEpilogue,
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AccDataType,
                                                                   CDataType,
                                                                   kPadA,
                                                                   kPadB,
                                                                   kTilePermute,
                                                                   kOutputRank,
                                                                   1,
                                                                   0,
                                                                   TilePartitioner::kM,
                                                                   TilePartitioner::kN>>,
        ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadA, kPadB>>>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadA, kPadB, kPadC, ALayout, BLayout, CLayout>;
    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;
    /// NOTICE: There are 3 limitations for using UniversalGemmPipelineAgBgCrPolicy<>
    ///     1. Both of M_Tile and N_Tile should be no less than get_warp_size()
    ///     2. K_Warp should be 1
    ///     3. (K_Tile * N_Warp_Tile / K_Warp_Tile) should be no less than 64
#if EXAMPLE_USE_UNIVERSAL_GEMM_PIPELINE_POLICY
    using CodegenGemmPolicy = ck_tile::UniversalGemmPipelineAgBgCrPolicy<ALayout, BLayout, CLayout>;
#endif
    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem
#if EXAMPLE_USE_UNIVERSAL_GEMM_PIPELINE_POLICY
                                                                      ,
                                                                      CodegenGemmPolicy
#endif
                                                                      >;
    // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
    // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
    using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKargs(args.p_a,
                                   args.p_b,
                                   args.p_c,
                                   args.M,
                                   args.N,
                                   args.K,
                                   args.stride_A,
                                   args.stride_B,
                                   args.stride_C);

    const dim3 grids      = Kernel::GridSize(args.M, args.N, args.kbatch);
    constexpr dim3 blocks = Kernel::BlockSize();

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return std::make_pair(ave_time, "GemmPipelineAGmemBGmemCRegV1");
}

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
