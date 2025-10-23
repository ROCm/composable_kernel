// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "ck_tile/ops/common.hpp"

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
          typename CDEElementWise,
          ck_tile::StreamKReductionStrategy ReductionStrategy>
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::StreamKHostArgs& args,
                                         const ck_tile::stream_config& s)

{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;

    using TilePartitioner = ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 GemmConfig::Persistent,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;

    const auto Run = [&](const auto memory_operation) -> std::tuple<float, ck_tile::index_t> {
        // We create the GEMM pipeline without specifying has_hot_loop or tail_num.
        // This is because num_loop can vary (a) per WG and (b) per iteration of the Stream-K
        // while loop. Instead, has_hot_loop and tail_num are determined in the Stream-K
        // Kernel's RunGemm function. This is a similar pattern used by grouped GEMM.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           GemmConfig::Scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation.value,
                                             GemmConfig::NumWaveGroups>>;

        using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        dim3 grids  = Kernel::GridSize(kargs.tile_partitioner);
        dim3 blocks = Kernel::BlockSize();

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

        // Function to clear the output C tensor results after each repetition of the kernel
        auto clear_gemm_output = [&]() {
            if(ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic)
                hipGetErrorString(hipMemsetAsync(
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
        };

        std::function<void()> preprocess = clear_gemm_output;

        float ave_time = ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        ck_tile::index_t num_wgs_per_tile = ck_tile::estimate_num_wgs_per_tile<ReductionStrategy>(
            kargs.tile_partitioner.sk_num_blocks,
            // k_iters_per_big_block could be 1, which indicates that all Stream-K workgroups are
            // big and each does one iteration. Thus, we ensure the value passed in is at least 1 to
            // avoid division by zero errors.
            ck_tile::max(kargs.tile_partitioner.k_iters_per_big_block - 1, 1u),
            kargs.tile_partitioner.k_iters_per_tile.get());

        return std::tuple{ave_time, num_wgs_per_tile};
    };

    if constexpr(ck_tile::StreamKReductionStrategy::Atomic == ReductionStrategy)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              // Since we are doing stream K, in the case of
                                              // atomics, multiple workgroups may write to the same
                                              // output tile in the C tensor, so we must atomic add
                                              // the results (not set)
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
    else // We are using ck_tile::StreamKReductionStrategy::Reduction
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              // In this case, there is only ever 1 WG writing final
                                              // results to each macro tile in the C tensor, so we
                                              // can do a set.
                                              ck_tile::memory_operation_enum::set>{});
    }
}

template <typename GemmConfig, typename TypeConfig>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_gemm_example_with_layouts<GemmConfig, TypeConfig>(
            argc, argv, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported layouts.");
    }

    return 0;
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

    if(data_type == "bf16")
    {
        using TypeConfig = StreamKGemmTypeConfig<ck_tile::bf16_t>;
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf16_t>, TypeConfig>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp16")
    {
        using TypeConfig = StreamKGemmTypeConfig<ck_tile::half_t>;
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, TypeConfig>(
            a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }

    return false;
}

int main(int argc, char* argv[])
{
    return !run_gemm_example<GemmConfigMemoryInterwave>(argc, argv);
}
