// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core/config.hpp"
#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"

template <typename GemmConfig,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          uint32_t QuantGroupSize,
          typename CDEElementWise>
float gemm_calc_aquant(const ck_tile::AQuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    static_assert(std::is_same_v<ELayout, ck_tile::tensor_layout::gemm::RowMajor>);

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using Traits = ck_tile::TileGemmAQuantTraits<GemmConfig::kPadM,
                                                 GemmConfig::kPadN,
                                                 GemmConfig::kPadK,
                                                 ALayout,
                                                 BLayout,
                                                 ELayout>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                 BDataType,
                                                                 AccDataType,
                                                                 GemmShape,
                                                                 Traits,
                                                                 ComputeDataType>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t K_split =
        (args.K + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile * GemmConfig::K_Tile;
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

        using UniversalGemmProblem = ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                                        AQDataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        GemmShape,
                                                                        Traits,
                                                                        QuantGroupSize,
                                                                        ComputeDataType,
                                                                        scheduler,
                                                                        has_hot_loop_v,
                                                                        tail_number_v>;
        using GemmPipeline         = typename PipelineTypeTraits<
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
                                             memory_operation>>;
        using Kernel = ck_tile::AQuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
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

        ave_time = ck_tile::launch_kernel(s,
                                          ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                              Kernel{}, grids, blocks, 0, kargs));

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
            throw std::runtime_error("split-k is not supported yet!");
        }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}

#include "run_gemm_aquant_example.inc"

template <typename GemmConfig, typename TypeConfig, uint32_t QuantGroupSize>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if constexpr(std::is_same_v<typename TypeConfig::ADataType, ck_tile::pk_int4_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::fp8_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::bf8_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig, TypeConfig, QuantGroupSize>(
                argc, argv, Row{}, Row{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for A.");
    }

    return 0;
}

template <typename GemmConfig>
int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, float>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4fp8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::fp8_t,
                                                        float,
                                                        ck_tile::fp8_t>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::bf8_t,
                                                        float,
                                                        ck_tile::bf8_t>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4f32fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::fp8_t, float, float>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4f32bf8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::bf8_t, float, float>{});
        return run_gemm_example_prec_type<GemmConfig, TypeConfig, 128>(
            a_layout, b_layout, argc, argv);
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
