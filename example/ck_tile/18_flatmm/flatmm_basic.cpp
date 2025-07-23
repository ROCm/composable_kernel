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
#include "run_flatmm_example.inc"

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::FlatmmHostArgs<>& args, const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               persistent,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = FlatmmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using CodegenPipelineProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                             BDataType,
                                                                             AccDataType,
                                                                             CodegenFlatmmShape,
                                                                             CodegenGemmTraits,
                                                                             scheduler,
                                                                             has_hot_loop_v,
                                                                             tail_number_v>;

        using CodegenFlatmmPipeline =
            ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation,
                                             FlatmmConfig::NumWaveGroups>>;

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
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
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
            ave_time = ck_tile::launch_kernel_preprocess(
                s,
                run_flush_cache,
                ck_tile::make_kernel<blocks.x, FlatmmConfig::kBlockPerCu>(
                    Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time =
                ck_tile::launch_kernel(s,
                                       ck_tile::make_kernel<blocks.x, FlatmmConfig::kBlockPerCu>(
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

template <template <typename PreType> typename FlatmmConfig>
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
            run_flatmm_example_with_layouts<ck_tile::half_t, FlatmmConfig<ck_tile::half_t>>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            run_flatmm_example_with_layouts<ck_tile::bf16_t, FlatmmConfig<ck_tile::bf16_t>>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            run_flatmm_example_with_layouts<ck_tile::fp8_t, FlatmmConfig<ck_tile::fp8_t>>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf8")
        {
            run_flatmm_example_with_layouts<ck_tile::bf8_t, FlatmmConfig<ck_tile::bf8_t>>(
                argc, argv, Row{}, Col{}, Row{});
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
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    try
    {
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return !run_flatmm_example<FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            return !run_flatmm_example<FlatmmConfig32>(argc, argv);
        }
        else if(warp_tile == 2)
        {
            return !run_flatmm_example<FlatmmConfig16_950>(argc, argv);
        }
        else
        {
            return !run_flatmm_example<FlatmmConfig32_950>(argc, argv);
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
