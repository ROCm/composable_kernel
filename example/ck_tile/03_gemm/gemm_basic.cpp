// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "gemm_basic.hpp"

template <typename ALayout, typename BLayout, typename CLayout, typename DataTypeConfig>
float gemm_(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{
    using Types = GemmBasicTypeConfig<DataTypeConfig>;

    // Specific type aliases for easy access
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;

    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

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
                                                                   kPadM,
                                                                   kPadN,
                                                                   kTilePermute,
                                                                   kOutputRank,
                                                                   1,
                                                                   0,
                                                                   TilePartitioner::kM,
                                                                   TilePartitioner::kN>>,
        ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;
    using CodegenGemmPolicy = ck_tile::UniversalGemmPipelineAgBgCrPolicy;
    using CodegenGemmPipeline =
        ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenGemmPolicy>;
    // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
    // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
    using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

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
}

template <typename DataType>
float gemm_type_(const gemm_traits& t,
                 const ck_tile::GemmHostArgs& args,
                 const ck_tile::stream_config& s)
{
    if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
    {
        return gemm_<Row, Row, Row, DataType>(args, s);
    }
    else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
    {
        return gemm_<Row, Col, Row, DataType>(args, s);
    }
    else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
    {
        return gemm_<Col, Row, Row, DataType>(args, s);
    }
    else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
    {
        return gemm_<Col, Col, Row, DataType>(args, s);
    }
    else
    {
        throw std::runtime_error("Wrong! Layouts not supported!\n");
    }
}

float gemm(const gemm_traits& t, const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{
    if(t.data_type == "fp16")
    {
        return gemm_type_<GemmFp16>(t, args, s);
    }
    else if(t.data_type == "bf16")
    {
        return gemm_type_<GemmBf16>(t, args, s);
    }
    else
    {
        throw std::runtime_error("Wrong! Data type not supported!\n");
    }
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "R", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
