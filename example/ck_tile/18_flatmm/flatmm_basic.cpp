// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "flatmm_basic.hpp"

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
float flatmm_calc(const ck_tile::FlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 2;

    // This part comes from the Codegen
#if defined(USING_MFMA_16x16x32) || defined(ENABLE_FP16)
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 128;

    constexpr ck_tile::index_t M_Warp = 1;
    constexpr ck_tile::index_t N_Warp = 4;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = is_8bit_type<ADataType>::value ? 16 : 32;
    constexpr ck_tile::index_t N_Warp_Tile = is_8bit_type<ADataType>::value ? 16 : 32;
    constexpr ck_tile::index_t K_Warp_Tile = is_8bit_type<ADataType>::value ? 64 : 16;

#elif defined(USING_MFMA_32x32x16) && defined(ENABLE_FP8)
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 128;

    constexpr ck_tile::index_t M_Warp = 1;
    constexpr ck_tile::index_t N_Warp = 8;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = is_8bit_type<ADataType>::value ? 32 : 32;
    constexpr ck_tile::index_t N_Warp_Tile = is_8bit_type<ADataType>::value ? 32 : 32;
    constexpr ck_tile::index_t K_Warp_Tile = is_8bit_type<ADataType>::value ? 32 : 16;
#endif
    using CodegenFlatmmShape =
        ck_tile::TileFlatmmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                 ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                 ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenFlatmmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                BDataType,
                                                                AccDataType,
                                                                CodegenFlatmmShape,
                                                                CodegenGemmTraits>;
    const auto Run               = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             AccDataType,
                                             CDataType,
                                             CLayout,
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation>>;

        using CodegenFlatmmPolicy = ck_tile::UniversalFlatmmPipelineAgBgCrPolicy;
        using CodegenFlatmmPipeline =
            ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenFlatmmPolicy>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::FlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };
    if(args.k_batch == 1)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
}

#include "run_flatmm_example.inc"

int run_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "C")
    {
        if(data_type == "fp16")
        {
            run_flatmm_example_with_layouts<ck_tile::half_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            run_flatmm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            run_flatmm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf8")
        {
            run_flatmm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_flatmm_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
