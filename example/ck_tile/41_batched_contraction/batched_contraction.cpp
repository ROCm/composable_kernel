// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"

#include "ck_tile/ops/batched_contraction.hpp"
#include "contraction_utils.hpp"

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>

float batched_contraction_impl(const ck_tile::BatchedContractionHostArgs<DsDataType::size()>& args,
                               const ck_tile::stream_config& s)
{
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    constexpr bool DoubleSmemBuffer = false;

    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                         = 1;
    constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    constexpr ck_tile::index_t TileParitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, ELayout>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                 kPadN,
                                                                 kPadK,
                                                                 DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 TransposeC>;

    using Problem = ck_tile::BatchedContractionProblem<ADataType,
                                                       BDataType,
                                                       DsDataType,
                                                       EDataType,
                                                       NumDimG,           // NumDimG
                                                       NumDimM,           // NumDimM
                                                       NumDimN,           // NumDimN
                                                       NumDimK,           // NumDimK
                                                       DsDataType::size() // NumDTensor
                                                       >;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = UNIVERSAL_GEMM_PIPELINE<GemmPipelineProblem>;

    ck_tile::index_t K_total = 1;
    for(ck_tile::index_t i = NumDimG + NumDimM; i < NumDimG + NumDimM + NumDimK; ++i)
    {
        K_total *= args.A_dims[i];
    }

    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_total);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = GEMM_PIPELINE_SCHEDULER;
        constexpr auto memory_operation =
            ck_tile::memory_operation_enum::set; // Always set (no atomic_add)

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v>;

        using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;

        using Kernel =
            ck_tile::BatchedContractionKernel<Problem, TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::GetBlockSize();

        if(!Kernel::IsSupportedArguments(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping contraction!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetKernelName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << GemmPipelineProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        auto kernel = ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs);

        ave_time = ck_tile::launch_kernel(s, kernel);

        return ave_time;
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);

    return ave_time;
}

#define HANDLE_CASE(G, M, N, K)                                                  \
    if(num_g_dims == G && num_m_dims == M && num_n_dims == N && num_k_dims == K) \
    {                                                                            \
        return batched_contraction_impl<ADataType,                               \
                                        BDataType,                               \
                                        DsDataType,                              \
                                        AccDataType,                             \
                                        EDataType,                               \
                                        ALayout,                                 \
                                        BLayout,                                 \
                                        DsLayout,                                \
                                        ELayout,                                 \
                                        G,                                       \
                                        M,                                       \
                                        N,                                       \
                                        K,                                       \
                                        CDEElementWise>(args, s);                \
    }

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float batched_contraction(const ck_tile::BatchedContractionHostArgs<DsDataType::size()>& args,
                          const ck_tile::stream_config& s,
                          ck_tile::index_t num_g_dims,
                          ck_tile::index_t num_m_dims,
                          ck_tile::index_t num_n_dims,
                          ck_tile::index_t num_k_dims)
{
    std::cout << "Dimensions: G=" << num_g_dims << ", M=" << num_m_dims << ", N=" << num_n_dims
              << ", K=" << num_k_dims << std::endl;

    HANDLE_CASE(1, 1, 1, 1);
    HANDLE_CASE(2, 1, 1, 1);
    HANDLE_CASE(2, 2, 2, 1);
    HANDLE_CASE(1, 2, 1, 1);
    HANDLE_CASE(1, 1, 1, 2);
    HANDLE_CASE(2, 2, 2, 2);
    HANDLE_CASE(4, 4, 4, 4);

    throw std::runtime_error(
        "Unsupported dimension combination: G=" + std::to_string(num_g_dims) +
        ", M=" + std::to_string(num_m_dims) + ", N=" + std::to_string(num_n_dims) +
        ", K=" + std::to_string(num_k_dims) + ". Please add this combination to the kernel.");
}

#include "run_batched_contraction_example.inc"

int main(int argc, char* argv[])
{
    try
    {
        return !run_batched_contraction_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
