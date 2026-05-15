// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"

#include "gemm_weight_preshuffle_invoker.hpp"

template <template <typename, ck_tile::DataCachePrefetchKind, ck_tile::DataCachePrefetchKind>
          class GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
bool run_gemm_with_prefetch_comparison(ck_tile::ArgParser& arg_parser,
                                       bool compare_with_non_prefetch,
                                       ck_tile::DataCachePrefetchKind prefetch_kind_a,
                                       ck_tile::DataCachePrefetchKind prefetch_kind_b)
{
    using Row     = ck_tile::tensor_layout::gemm::RowMajor;
    using Col     = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Invoker = WeightPreshuffleInvoker;

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout != "R" || b_layout != "C")
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    std::cout << "\n=== Running with DataCache Prefetch ENABLED (TDM ";
    std::cout << (prefetch_kind_a == ck_tile::DataCachePrefetchKind::L1 ? "L1" : "L2")
              << " / Flat ";
    std::cout << (prefetch_kind_b == ck_tile::DataCachePrefetchKind::L1 ? "L1" : "L2") << ") ===\n"
              << std::endl;

    using Kind         = ck_tile::DataCachePrefetchKind;
    bool pass_prefetch = false;
    if(prefetch_kind_a == Kind::L1 && prefetch_kind_b == Kind::L1)
    {
        pass_prefetch = run_gemm_example_with_layouts<GemmConfig<APrecType, Kind::L1, Kind::L1>,
                                                      Invoker,
                                                      APrecType,
                                                      BPrecType,
                                                      CPrecType>(arg_parser, Row{}, Col{}, Row{});
    }
    else if(prefetch_kind_a == Kind::L1 && prefetch_kind_b == Kind::L2)
    {
        pass_prefetch = run_gemm_example_with_layouts<GemmConfig<APrecType, Kind::L1, Kind::L2>,
                                                      Invoker,
                                                      APrecType,
                                                      BPrecType,
                                                      CPrecType>(arg_parser, Row{}, Col{}, Row{});
    }
    else if(prefetch_kind_a == Kind::L2 && prefetch_kind_b == Kind::L1)
    {
        pass_prefetch = run_gemm_example_with_layouts<GemmConfig<APrecType, Kind::L2, Kind::L1>,
                                                      Invoker,
                                                      APrecType,
                                                      BPrecType,
                                                      CPrecType>(arg_parser, Row{}, Col{}, Row{});
    }
    else
    {
        pass_prefetch = run_gemm_example_with_layouts<GemmConfig<APrecType, Kind::L2, Kind::L2>,
                                                      Invoker,
                                                      APrecType,
                                                      BPrecType,
                                                      CPrecType>(arg_parser, Row{}, Col{}, Row{});
    }

    if(compare_with_non_prefetch)
    {
        std::cout << "\n=== Running with DataCache Prefetch DISABLED ===\n" << std::endl;
        bool pass_no_prefetch =
            run_gemm_example_with_layouts<GemmConfig<APrecType,
                                                     ck_tile::DataCachePrefetchKind::None,
                                                     ck_tile::DataCachePrefetchKind::None>,
                                          Invoker,
                                          APrecType,
                                          BPrecType,
                                          CPrecType>(arg_parser, Row{}, Col{}, Row{});

        std::cout << "\n=== Comparison Summary ===" << std::endl;
        std::cout << "Note: Check the timing results above to compare performance." << std::endl;
        std::cout << "With prefetch vs without prefetch - speedup can be observed in the "
                     "timing outputs."
                  << std::endl;

        return pass_prefetch && pass_no_prefetch;
    }

    return pass_prefetch;
}

template <template <typename, ck_tile::DataCachePrefetchKind, ck_tile::DataCachePrefetchKind>
          class GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    const std::string data_type = arg_parser.get_str("prec");

    const bool compare_with_non_prefetch = arg_parser.get_int("compare") == 1;
    const auto prefetch_kind_a           = arg_parser.get_int("prefetch_l1_a") == 1
                                               ? ck_tile::DataCachePrefetchKind::L1
                                               : ck_tile::DataCachePrefetchKind::L2;
    const auto prefetch_kind_b           = arg_parser.get_int("prefetch_l1_b") == 1
                                               ? ck_tile::DataCachePrefetchKind::L1
                                               : ck_tile::DataCachePrefetchKind::L2;

    if(data_type == "fp16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig, ck_tile::half_t>(
            arg_parser, compare_with_non_prefetch, prefetch_kind_a, prefetch_kind_b);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig, ck_tile::bf16_t>(
            arg_parser, compare_with_non_prefetch, prefetch_kind_a, prefetch_kind_b);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::fp8_t,
                                                 ck_tile::fp8_t,
                                                 ck_tile::half_t>(
            arg_parser, compare_with_non_prefetch, prefetch_kind_a, prefetch_kind_b);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::bf8_t,
                                                 ck_tile::bf8_t,
                                                 ck_tile::half_t>(
            arg_parser, compare_with_non_prefetch, prefetch_kind_a, prefetch_kind_b);
    }
    else if(data_type == "int4")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::fp8_t,
                                                 ck_tile::pk_int4_t,
                                                 ck_tile::half_t>(
            arg_parser, compare_with_non_prefetch, prefetch_kind_a, prefetch_kind_b);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for GEMM weight preshuffle TDM prefetch!");
    }
}

template <typename PrecType,
          ck_tile::DataCachePrefetchKind DataCachePrefetchA_ = ck_tile::DataCachePrefetchKind::None,
          ck_tile::DataCachePrefetchKind DataCachePrefetchB_ = DataCachePrefetchA_>
struct GemmConfigWeightPreshuffleTDMPrefetch : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr int kBlockPerCu                = 2;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::PRESHUFFLE_TDM;
    static constexpr bool Preshuffle                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA = DataCachePrefetchA_;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB = DataCachePrefetchB_;
    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    arg_parser.insert(
        "compare",
        "0",
        "0: Run with data cache prefetch only, 1: Compare with/without data cache prefetch");
    arg_parser.insert("prefetch_l1_a", "0", "0: Prefetch A to L2 cache, 1: Prefetch A to L1 cache");
    arg_parser.insert("prefetch_l1_b", "1", "0: Prefetch B to L2 cache, 1: Prefetch B to L1 cache");
    auto result = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example<GemmConfigWeightPreshuffleTDMPrefetch>(arg_parser);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
