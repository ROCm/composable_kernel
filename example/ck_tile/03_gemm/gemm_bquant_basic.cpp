// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/core/config.hpp"
#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"

template <typename T>
struct is_8bit_type
    : std::bool_constant<std::is_same_v<T, ck_tile::fp8_t> || std::is_same_v<T, ck_tile::bf8_t>>
{
};

template <typename ADataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize,
          bool as_br_cr>
float gemm_calc_bquant(const ck_tile::BQuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    if constexpr(!as_br_cr)
    {
        constexpr bool kPadM = false;
        constexpr bool kPadN = false;
        constexpr bool kPadK = false;

        constexpr int kBlockPerCu = 1;

        static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);

        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 64;
        constexpr ck_tile::index_t K_Tile = 256;

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
            ck_tile::TileGemmBQuantTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
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

            using CodegenPipelineProblem =
                ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                   BDataType,
                                                   BQDataType,
                                                   AccDataType,
                                                   CodegenGemmShape,
                                                   CodegenGemmTraits,
                                                   QuantGroupSize,
                                                   transposed_warp_gemm,
                                                   ComputeDataType,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   has_hot_loop_v,
                                                   tail_number_v>;
            using CodegenGemmPipeline =
                ck_tile::BQuantGemmPipelineAgBgCrCompV3<CodegenPipelineProblem>;
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
                                                 transposed_warp_gemm,
                                                 ck_tile::memory_operation_enum::set>>;
            using Kernel =
                ck_tile::BQuantGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

            auto kargs = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            assert(args.k_batch == 1 && "split-k is not supported yet!");

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
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            float ave_time = ck_tile::launch_kernel(
                s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

            return ave_time;
        };
        if(has_hot_loop)
        {
            if(tail_num == ck_tile::TailNumber::Full)
            {
                return Run(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                return Run(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                return Run(
                    ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("Unsupported tail number for this operation !!!");
            }
        }
        else
        {
            if(tail_num == ck_tile::TailNumber::Full)
            {
                return Run(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                return Run(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                return Run(
                    ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("Unsupported tail number for this operation !!!");
            }
        }
    }
    else
    {
        // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
        constexpr bool kPadM = false;
        constexpr bool kPadN = false;
        constexpr bool kPadK = false;

        constexpr int kBlockPerCu = 2;

        constexpr bool transposed_warp_gemm = false;

        std::cout << "flatmm kernel from host..." << std::endl;

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
        using FlatmmCodegenGemmShape =
            ck_tile::TileFlatmmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                     ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                     ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using FlatmmTilePartitioner = ck_tile::GemmTile1DPartitioner<FlatmmCodegenGemmShape>;

        using FlatmmCodegenGemmTraits =
            ck_tile::TileGemmBQuantTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        /*using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits>;*/

        using FlatmmCodegenPipelineProblem =
            ck_tile::GemmBQuantPipelineProblem<ADataType,
                                               BDataType,
                                               BQDataType,
                                               AccDataType,
                                               FlatmmCodegenGemmShape,
                                               FlatmmCodegenGemmTraits,
                                               QuantGroupSize,
                                               transposed_warp_gemm,
                                               ComputeDataType,
                                               ck_tile::GemmPipelineScheduler::Intrawave,
                                               false,
                                               ck_tile::TailNumber::Full>;
        const auto Run = [&](const auto memory_operation_) {
            constexpr auto memory_operation = memory_operation_.value;

            using FlatmmGemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 FlatmmCodegenPipelineProblem::kBlockSize,
                                                 FlatmmTilePartitioner::MPerBlock,
                                                 FlatmmTilePartitioner::NPerBlock,
                                                 M_Warp,
                                                 N_Warp,
                                                 M_Warp_Tile,
                                                 N_Warp_Tile,
                                                 K_Warp_Tile,
                                                 FlatmmCodegenPipelineProblem::TransposeC,
                                                 memory_operation>>;

            using FlatmmCodegenPolicy = ck_tile::GemmBQuantFlatmmPipelineAgBgCrDefaultPolicyV1;
            using FlatmmCodegenPipeline =
                ck_tile::GemmBQuantFlatmmPipelineAgBgCrV1<FlatmmCodegenPipelineProblem,
                                                          FlatmmCodegenPolicy>;

            // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
            // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
            using Kernel = ck_tile::
                BQuantGemmKernel<FlatmmTilePartitioner, FlatmmCodegenPipeline, FlatmmGemmEpilogue>;

            auto kargs = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args:" << " grid: {" << grids.x << ", "
                          << grids.y << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", "
                          << blocks.y << ", " << blocks.z << "}" << std::endl;
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

    return 0.f;
}

#include "run_gemm_bquant_example.inc"

template <typename TypeConfig, uint32_t QuantGroupSize, bool as_br_cr>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if constexpr(std::is_same_v<typename TypeConfig::BDataType, ck_tile::pk_int4_t> ||
                 std::is_same_v<typename TypeConfig::BDataType, ck_tile::fp8_t> ||
                 std::is_same_v<typename TypeConfig::BDataType, ck_tile::bf8_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<TypeConfig, QuantGroupSize, as_br_cr>(
                argc, argv, Row{}, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for B.");
    }

    return 0;
}

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    bool as_br_cr         = arg_parser.get_bool("as_br_cr");

    if(as_br_cr)
    {
        // TODO: (Add support for fp8i4, bf8i4 once we tune perf for fp8/bf8).
        if(data_type == "fp8")
        {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                            ck_tile::fp8_t,
                                                            float,
                                                            ck_tile::fp8_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, true>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "bf8")
        {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                            ck_tile::bf8_t,
                                                            float,
                                                            ck_tile::bf8_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, true>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for this operation !!!");
        }
    }
    else
    {
        if(data_type == "fp8")
        {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                            ck_tile::fp8_t,
                                                            float,
                                                            ck_tile::fp8_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "bf8")
        {
            using TypeConfig =
                decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "fp8i4")
        {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                            ck_tile::pk_int4_t,
                                                            float,
                                                            ck_tile::fp8_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "bf8i4")
        {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                            ck_tile::pk_int4_t,
                                                            float,
                                                            ck_tile::bf8_t>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "fp8i4f32")
        {
            using TypeConfig =
                decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::pk_int4_t, float, float>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else if(data_type == "bf8i4f32")
        {
            using TypeConfig =
                decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::pk_int4_t, float, float>{});
            return run_gemm_example_prec_type<TypeConfig, 128, false>(
                a_layout, b_layout, argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for this operation !!!");
        }
    }
}

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
