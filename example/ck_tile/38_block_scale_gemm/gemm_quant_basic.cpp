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
          typename TypeConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize,
          ck_tile::QuantType QuantMode>
float gemm_calc_quant(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    constexpr bool kPadM = GemmConfig::kPadM;
    constexpr bool kPadN = GemmConfig::kPadN;
    constexpr bool kPadK = GemmConfig::kPadK;

    constexpr int kBlockPerCu = GemmConfig::kBlockPerCu;

    static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);

    constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
    constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
    constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

    constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    // B datatype is safe to use as compute type as it should be at least fp8
    using ComputeDataType = typename TypeConfig::BDataType;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                           kPadN,
                                                           kPadK,
                                                           GemmConfig::PreshuffleQuant,
                                                           ALayout,
                                                           BLayout,
                                                           CLayout,
                                                           QuantMode>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<typename TypeConfig::ADataType,
                                                                 typename TypeConfig::BDataType,
                                                                 typename TypeConfig::AccDataType,
                                                                 CodegenGemmShape,
                                                                 CodegenGemmTraits,
                                                                 ComputeDataType>;

    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

    const ck_tile::index_t K_split      = (args.K + K_Tile - 1) / K_Tile * K_Tile;
    const ck_tile::index_t num_loop     = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop             = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num  = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    constexpr bool transposed_warp_gemm = false;

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr bool transpose_c    = false;

        using CodegenPipelineProblem = std::conditional_t<
            QuantMode == ck_tile::QuantType::AQuantGrouped,
            ck_tile::GemmAQuantPipelineProblem<typename TypeConfig::ADataType,
                                               typename TypeConfig::QDataType,
                                               typename TypeConfig::BDataType,
                                               typename TypeConfig::AccDataType,
                                               CodegenGemmShape,
                                               CodegenGemmTraits,
                                               QuantGroupSize,
                                               transpose_c,
                                               ComputeDataType,
                                               ck_tile::GemmPipelineScheduler::Intrawave,
                                               has_hot_loop_v,
                                               tail_number_v>,
            ck_tile::GemmRowColQuantPipelineProblem<typename TypeConfig::ADataType,
                                                    typename TypeConfig::BDataType,
                                                    typename TypeConfig::AccDataType,
                                                    typename TypeConfig::AccDataType,
                                                    CodegenGemmShape,
                                                    CodegenGemmTraits,
                                                    transpose_c,
                                                    ComputeDataType,
                                                    ck_tile::GemmPipelineScheduler::Intrawave,
                                                    has_hot_loop_v,
                                                    tail_number_v>>;

        using CodegenGemmPipeline =
            std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                               ck_tile::AQuantGemmPipelineAgBgCrCompV3<CodegenPipelineProblem>,
                               ck_tile::GemmPipelineAgBgCrCompV3<CodegenPipelineProblem>>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<typename TypeConfig::ADataType,
                                             typename TypeConfig::BDataType,
                                             ck_tile::tuple<>,
                                             typename TypeConfig::AccDataType,
                                             typename TypeConfig::CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             transposed_warp_gemm,
                                             ck_tile::memory_operation_enum::set>>;
        using Kernel =
            ck_tile::QuantGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue, QuantMode>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(args.k_batch != 1)
        {
            throw std::runtime_error("split-k is not supported yet!");
        }

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
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };
    return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

#include "run_gemm_quant_example.inc"

template <typename GemmConfig,
          typename TypeConfig,
          uint32_t QuantGroupSize,
          ck_tile::QuantType QuantMode>
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
            return run_gemm_example_with_layouts<GemmConfig, TypeConfig, QuantGroupSize, QuantMode>(
                argc, argv, Row{}, Row{}, Col{}, Col{}, Row{});
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

void print_help(const char* program_name)
{
    std::cout
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "\n Parameters:\n"
        << "  -quant_mode=MODE     aquant (A quantization) or rowcol (row/column quantization)\n"
        << "  -v=LEVEL              0=No validation, 1=CPU validation, 2=GPU validation\n"
        << "  -prec=TYPE           Data types: fp8, bf8, i4fp8, i4bf8, i4f32fp8, i4f32bf8\n"
        << "  -m, -n, -k=SIZE       Matrix dimensions (M×K) @ (K×N) = (M×N)\n"
        << "\nExample:\n"
        << "  " << program_name << " -quant_mode=rowcol -v=1 -prec=fp8\n"
        << std::endl;
}

template <template <typename PreType> typename GemmConfig>
int run_gemm_example(int argc, char* argv[])
{

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    std::string quant_mode = arg_parser.get_str("quant_mode");

    if(data_type == "fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else if(quant_mode == "rowcol")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::RowColQuant>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported quantization mode! Use 'aquant' or 'rowcol'");
        }
    }
    else if(data_type == "bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, float>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else if(quant_mode == "rowcol")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::RowColQuant>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported quantization mode! Use 'aquant' or 'rowcol'");
        }
    }
    else if(data_type == "i4fp8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::fp8_t,
                                                        float,
                                                        ck_tile::fp8_t>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::pk_int4_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported quantization mode for this datatype! Use 'aquant'.");
        }
    }
    else if(data_type == "i4bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::bf8_t,
                                                        float,
                                                        ck_tile::bf8_t>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::pk_int4_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported quantization mode for this datatype! Use 'aquant'.");
        }
    }
    else if(data_type == "i4f32fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::fp8_t, float, float>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::pk_int4_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported quantization mode for this datatype! Use 'aquant'.");
        }
    }
    else if(data_type == "i4f32bf8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::bf8_t, float, float>{});

        if(quant_mode == "aquant")
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::pk_int4_t>,
                                              TypeConfig,
                                              128,
                                              ck_tile::QuantType::AQuantGrouped>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported quantization mode for this datatype! Use 'aquant'.");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[]) { return !run_gemm_example<GemmConfigQuant>(argc, argv); }
