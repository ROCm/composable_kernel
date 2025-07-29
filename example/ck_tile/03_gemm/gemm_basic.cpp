// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

{
    if constexpr(Persistent)
        std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;

    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
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

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
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

#include "run_gemm_example.inc"

template <typename APrecType, typename BPrecType = APrecType, typename CPrecType = APrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices when "
                                     "BPrecType is ck_tile::pk_int4_t!");
        }
    }
    else
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(a_layout == "R" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfigBase, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
}

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<ck_tile::half_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i8")
    {
        return run_gemm_example_prec_type<ck_tile::int8_t, ck_tile::int8_t, int32_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "pk_int4_t")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfigBase::Pipeline == CK_TILE_PIPELINE_COMPUTE_V3)
        {
            return run_gemm_example_prec_type<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for this operation !!!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_gemm_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
