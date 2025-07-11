// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel_streamk.hpp"
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
float gemm_streamk(const ck_tile::StreamKGemmHostArgs</*NumDTensor = 0*/>& args, 
                   const ck_tile::stream_config& s)
{
    // Stream-K configuration parameters
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // Tile configuration - same as basic GEMM for consistency
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

    // Use the same pipeline as basic GEMM
    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    // Epilogue configuration with atomic operations support for Stream-K
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
                                         ck_tile::memory_operation_enum::atomic_add>>; // Always use atomic for Stream-K

    // Use Stream-K kernel instead of regular kernel
    using Kernel = ck_tile::StreamKGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
    
    // Get device properties for Stream-K configuration
    hipDeviceProp_t device_prop;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&device_prop, 0));
    const int num_sms = device_prop.multiProcessorCount;
    
    // Create Stream-K specific arguments
    auto streamk_args = args;
    
    // Calculate optimal Stream-K configuration if not provided
    if (streamk_args.sk_blocks == 0) {
        // Calculate work decomposition
        const auto [total_tiles, work_per_block, remainder_work, sk_blocks] = 
            ck_tile::StreamKWorkDecomposition::ComputeStreamKDecomposition(
                args.M, args.N, args.K,
                TilePartitioner::MPerBlock,
                TilePartitioner::NPerBlock,
                TilePartitioner::KPerBlock,
                num_sms,
                args.grid_size_multiplier);
        
        streamk_args.sk_blocks = sk_blocks;
        streamk_args.sk_big_blocks = remainder_work;
        streamk_args.num_sms = num_sms;
    }
    
    auto kargs = Kernel::MakeKernelArgs(streamk_args);

    // Grid size is determined by Stream-K blocks
    const dim3 grids = Kernel::GridSize(args.M, args.N, args.K, streamk_args.sk_blocks);
    constexpr dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping Stream-K gemm!\n");
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Launching Stream-K kernel with args: " << Kernel::GetName() << '\n'
                  << "shape: " << CodegenGemmShape::GetName() << '\n'
                  << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                  << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                  << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << '\n'
                  << "Stream-K blocks: " << streamk_args.sk_blocks << '\n'
                  << "Big blocks: " << streamk_args.sk_big_blocks << '\n'
                  << "Work per block: " << kargs.work_per_block << '\n'
                  << "Total tiles: " << kargs.total_tiles << '\n'
                  << "Grid size multiplier: " << args.grid_size_multiplier << std::endl;
    }

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

#include "run_gemm_example.inc"

template <typename APrecType, typename BPrecType = APrecType, typename CPrecType = APrecType>
int run_streamk_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    // Add Stream-K specific arguments
    arg_parser.insert("streamk_mode", 1, "Stream-K mode: 0=disabled, 1=stream-k, 2=split-k");
    arg_parser.insert("grid_multiplier", 1.0f, "Grid size multiplier for Stream-K");
    arg_parser.insert("sk_blocks", 0, "Number of Stream-K blocks (0=auto)");
    
    if(!arg_parser.parse(argc, argv))
        return -1;

    // Define the Stream-K GEMM configuration
    struct StreamKGemmConfig : GemmConfigBase
    {
        static constexpr bool Persistent = true;  // Stream-K uses persistent kernels
    };

    if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{}, gemm_streamk<StreamKGemmConfig, 
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType, 
                                                               Row, Col, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Col{}, Row{}, gemm_streamk<StreamKGemmConfig,
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType,
                                                               Col, Col, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
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
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{}, gemm_streamk<StreamKGemmConfig,
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType,
                                                               Row, Col, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
        }
        else if(a_layout == "R" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Row{}, Row{}, gemm_streamk<StreamKGemmConfig,
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType,
                                                               Row, Row, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
        }
        else if(a_layout == "C" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Row{}, Row{}, gemm_streamk<StreamKGemmConfig,
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType,
                                                               Col, Row, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<StreamKGemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Col{}, Row{}, gemm_streamk<StreamKGemmConfig,
                                                               APrecType, BPrecType, ck_tile::tuple<>,
                                                               CPrecType, CPrecType,
                                                               Col, Col, ck_tile::tuple<>, Row,
                                                               true, ck_tile::element_wise::PassThrough>);
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
}

int run_streamk_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    std::cout << "Running Stream-K GEMM example..." << std::endl;

    if(data_type == "fp16")
    {
        return run_streamk_example_prec_type<ck_tile::half_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_streamk_example_prec_type<ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8")
    {
        return run_streamk_example_prec_type<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        return run_streamk_example_prec_type<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i8")
    {
        return run_streamk_example_prec_type<ck_tile::int8_t, ck_tile::int8_t, int32_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "pk_int4_t")
    {
        if constexpr(GemmConfigBase::Pipeline == CK_TILE_PIPELINE_COMPUTE_V3)
        {
            return run_streamk_example_prec_type<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for Stream-K with this pipeline!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for Stream-K operation!");
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_streamk_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}