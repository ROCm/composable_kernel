// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// This example demonstrates 2D block scale quantization (N×K) for BQuant
// using non-preshuffled configuration.
// NOTE: Once more 2d support is ready, we can migrate all 2d quant types to this example
// This is currently done separately to avoid too verbose dispatching.

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
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename CDEElementWise>
float gemm_calc_quant(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);
    using ComputeDataType = std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped ||
                                                   QuantMode == ck_tile::QuantType::RowColQuant,
                                               typename TypeConfig::BDataType,
                                               typename TypeConfig::ADataType>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
                                                    GemmConfig::kPadN,
                                                    GemmConfig::kPadK,
                                                    GemmConfig::PreshuffleQuant,
                                                    GemmConfig::PreshuffleB,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    QuantMode,
                                                    ALayout, // for AQLayout
                                                    BLayout, // for BQLayout
                                                    false,
                                                    GemmConfig::DoubleSmemBuffer>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<typename TypeConfig::ADataType,
                                                                 typename TypeConfig::BDataType,
                                                                 typename TypeConfig::AccDataType,
                                                                 GemmShape,
                                                                 GemmTraits,
                                                                 ComputeDataType>;

    // This example only supports BQuant (no AQuant)
    // For non-preshuffled BQuant, use BaseBQuantGemmPipelineAgBgCrCompV3
    using BaseGemmPipeline = std::conditional_t<
        GemmConfig::PreshuffleB == true,
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>,
        ck_tile::BaseBQuantGemmPipelineAgBgCrCompV3<GemmPipelineProblem>>;

    const ck_tile::index_t K_split =
        (args.K + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr bool transpose_c    = false;

        // row-col and tensor quants use the regular pipeline, A/B quants use their own
        using PipelineProblem = std::conditional_t<
            QuantMode == ck_tile::QuantType::RowColQuant ||
                QuantMode == ck_tile::QuantType::TensorQuant,
            ck_tile::GemmRowColTensorQuantPipelineProblem<typename TypeConfig::ADataType,
                                                          typename TypeConfig::BDataType,
                                                          typename TypeConfig::AccDataType,
                                                          typename TypeConfig::AccDataType,
                                                          GemmShape,
                                                          GemmTraits,
                                                          transpose_c,
                                                          ComputeDataType,
                                                          GemmConfig::Scheduler,
                                                          has_hot_loop_v,
                                                          tail_number_v>,
            std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                               ck_tile::GemmAQuantPipelineProblem<typename TypeConfig::ADataType,
                                                                  typename TypeConfig::QDataType,
                                                                  typename TypeConfig::BDataType,
                                                                  typename TypeConfig::AccDataType,
                                                                  GemmShape,
                                                                  GemmTraits,
                                                                  QuantGroupSize,
                                                                  transpose_c,
                                                                  ComputeDataType,
                                                                  GemmConfig::Scheduler,
                                                                  has_hot_loop_v,
                                                                  tail_number_v>,
                               ck_tile::GemmBQuantPipelineProblem<typename TypeConfig::ADataType,
                                                                  typename TypeConfig::BDataType,
                                                                  typename TypeConfig::QDataType,
                                                                  typename TypeConfig::AccDataType,
                                                                  GemmShape,
                                                                  GemmTraits,
                                                                  QuantGroupSize,
                                                                  ComputeDataType,
                                                                  GemmConfig::Scheduler,
                                                                  has_hot_loop_v,
                                                                  tail_number_v>>>;

        using GemmPipeline = std::conditional_t<
            QuantMode == ck_tile::QuantType::RowColQuant ||
                QuantMode == ck_tile::QuantType::TensorQuant,
            ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>,
            std::conditional_t<
                QuantMode == ck_tile::QuantType::AQuantGrouped,
                ck_tile::AQuantGemmPipelineAgBgCrMem<PipelineProblem>, // memory pipeline hardcoded
                                                                       // for aquant
                std::conditional_t<GemmConfig::PreshuffleB == true,
                                   ck_tile::WPQuantBPipelineAgBgCrV2<PipelineProblem>,
                                   ck_tile::BQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<typename TypeConfig::ADataType,
                                             typename TypeConfig::BDataType,
                                             ck_tile::tuple<>,
                                             typename TypeConfig::AccDataType,
                                             typename TypeConfig::CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             transpose_c,
                                             ck_tile::memory_operation_enum::set,
                                             1,
                                             false,
                                             1,
                                             GemmConfig::TiledMMAPermuteN>>;
        using Kernel =
            ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

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
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << PipelineProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        float ave_time = 0;
        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<typename TypeConfig::ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<typename TypeConfig::BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            ck_tile::RotatingMemWrapper<typename TypeConfig::ADataType,
                                        typename TypeConfig::BDataType>
                rotating_mem(
                    kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(
                        hipMemsetAsync(args.c_ptr,
                                       0,
                                       args.M * args.N * sizeof(typename TypeConfig::CDataType),
                                       s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }

        return ave_time;
    };
    return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

#include "run_gemm_quant_example.inc"

template <typename GemmConfig,
          typename TypeConfig,
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if((QuantMode == ck_tile::QuantType::AQuantGrouped ||
        QuantMode == ck_tile::QuantType::RowColQuant) &&
       GemmConfig::PreshuffleB)
    {
        throw std::runtime_error(
            "Preshuffling weight matrix is not supported for AQuant or RowColQuant");
    }

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

// Forward declaration for dispatch function
template <template <typename PreType> typename GemmConfig, typename QuantGroupSize>
int dispatch_by_data_type(const std::string& data_type,
                          const std::string& quant_mode,
                          const std::string& a_layout,
                          const std::string& b_layout,
                          int argc,
                          char* argv[]);

// Helper function to parse group size string "MxNxK"
std::tuple<int, int, int> parse_group_size(const std::string& group_size_str)
{
    int m = 1, n = 1, k = 128;

    size_t first_x = group_size_str.find('x');
    if(first_x == std::string::npos)
    {
        // Single number provided, assume it's the K dimension
        k = std::stoi(group_size_str);
        return {1, 1, k};
    }

    size_t second_x = group_size_str.find('x', first_x + 1);
    if(second_x == std::string::npos)
    {
        throw std::runtime_error("Invalid group_size format! Expected MxNxK (e.g., 1x32x128)");
    }

    m = std::stoi(group_size_str.substr(0, first_x));
    n = std::stoi(group_size_str.substr(first_x + 1, second_x - first_x - 1));
    k = std::stoi(group_size_str.substr(second_x + 1));

    return {m, n, k};
}

template <template <typename PreType> typename GemmConfig>
int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type      = arg_parser.get_str("prec");
    std::string a_layout       = arg_parser.get_str("a_layout");
    std::string b_layout       = arg_parser.get_str("b_layout");
    std::string quant_mode     = arg_parser.get_str("quant_mode");
    std::string group_size_str = arg_parser.get_str("group_size");

    auto [m_group, n_group, k_group] = parse_group_size(group_size_str);

    // Dispatch based on group size (M, N, K)
    return dispatch_group_size_ct<GemmConfig>(m_group, n_group, k_group, [&](auto QGS_) {
        using QuantGroupSize = decltype(QGS_);
        return dispatch_by_data_type<GemmConfig, QuantGroupSize>(
            data_type, quant_mode, a_layout, b_layout, argc, argv);
    });
}

template <template <typename PreType> typename GemmConfig, typename QuantGroupSize>
int dispatch_by_data_type(const std::string& data_type,
                          const std::string& quant_mode,
                          const std::string& a_layout,
                          const std::string& b_layout,
                          int argc,
                          char* argv[])
{
    // This example ONLY supports BQuant for 2D block scale quantization
    if(quant_mode != "bquant")
    {
        throw std::runtime_error("This example only supports BQuant! Use --quant_mode=bquant");
    }

    if(data_type == "fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});

        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::BQuantGrouped>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});

        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::BQuantGrouped>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8i4")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                        ck_tile::pk_int4_t,
                                                        ck_tile::half_t,
                                                        ck_tile::fp8_t>{});

        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::BQuantGrouped>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8i4")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                        ck_tile::pk_int4_t,
                                                        ck_tile::half_t,
                                                        ck_tile::bf8_t>{});

        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::BQuantGrouped>(
            a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

template <template <typename> typename GemmConfig, typename F>
int dispatch_group_size_ct(int m, int n, int k, F&& f)
{
    // This expands into a sequence of `if (m==M && n==N && k==K) { ... }`
#define DISPATCH_ONE(M, N, K)                                                        \
    if(m == M && n == N && k == K)                                                   \
    {                                                                                \
        using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<M, N, K>>; \
        return f(QuantGroupSize{});                                                  \
    }

    CK_TILE_SUPPORTED_QUANT_GROUPS(DISPATCH_ONE)

#undef DISPATCH_ONE

    throw std::runtime_error(
        "Unsupported group size! Please add it to CK_TILE_SUPPORTED_QUANT_GROUPS(X).");
}

int main(int argc, char* argv[])
{
    // Use non-preshuffled GemmConfig for 2D block scale support
    return !run_gemm_example<GemmConfigBQuantPrefill>(argc, argv);
}
