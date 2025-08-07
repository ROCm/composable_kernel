// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"
#include "run_gemm_example.inc"

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool Persistent,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                           GemmConfig::kPadN,
                                           GemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           GemmConfig::NumWaveGroups>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             UniversalGemmProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation,
                                             GemmConfig::NumWaveGroups>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        dim3 grids;
        if constexpr(Persistent)
        {
            grids = Kernel::MaxOccupancyGridSize(s);
        }
        else
        {
            grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        }
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                    Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time =
                ck_tile::launch_kernel(s,
                                       ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                           Kernel{}, grids, blocks, 0, kargs));
        }
        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}

template <typename GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row                 = ck_tile::tensor_layout::gemm::RowMajor;
    using Col                 = ck_tile::tensor_layout::gemm::ColumnMajor;
    auto [result, arg_parser] = create_args(argc, argv);
    bool preshuffle           = GemmConfig::Preshuffle;

    if(preshuffle && std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        throw std::runtime_error("Preshuffle is not supported for this int4 datatype!");
    }

    if(preshuffle && a_layout != "R" && b_layout != "C")
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
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
        if(a_layout == "R" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Row{}, Row{});
        }
        else if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "R")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig, APrecType, BPrecType, CPrecType>(
                argc, argv, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
}

template <template <typename PreType> typename GemmConfig>
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
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::bf16_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          ck_tile::bf8_t,
                                          ck_tile::bf8_t,
                                          ck_tile::half_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "int8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::int8_t>,
                                          ck_tile::int8_t,
                                          ck_tile::int8_t,
                                          ck_tile::int32_t>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "pk_int4_t")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == CK_TILE_PIPELINE_COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>,
                                              ck_tile::half_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
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
        return !run_gemm_example<GemmConfigComputeV3>(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
