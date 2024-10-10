
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_basic.hpp"

template <typename ALayout, typename BLayout, typename CLayout>
float gemm_calc(const gemm_basic_args& args, const ck_tile::stream_config& s)
{
    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA = true;
    constexpr bool kPadB = true;
    constexpr bool kPadC = true;

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

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using CodegenPipelineProblem = ck_tile::BlockGemmPipelineProblem<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     kPadA,
                                                                     kPadB,
                                                                     kPadC>;

    using CodegenGemmPipeline = ck_tile::BlockGemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    using TilePartitioner = ck_tile::GemmTilePartitioner<CodegenGemmShape>;
    using GemmEpilogue    = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadA, kPadB>>;
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
        std::cout << "Lunching kernel with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
