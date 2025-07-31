// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <tuple>
#include <exception>

#include "gemm_profiler.hpp"
#include "benchmark_gemm.hpp"

void benchmark_gemm(const ck_tile::ArgParser& arg_parser)
{
    GemmProblem gemm_problem{arg_parser.get_int("split_k"),
                             arg_parser.get_int("m"),
                             arg_parser.get_int("n"),
                             arg_parser.get_int("k"),
                             arg_parser.get_int("stride_a"),
                             arg_parser.get_int("stride_b"),
                             arg_parser.get_int("stride_c"),
                             DataTypeTraits<ADataType>::name,
                             DataTypeTraits<BDataType>::name,
                             DataTypeTraits<AccDataType>::name,
                             DataTypeTraits<CDataType>::name,
                             ALayout::name,
                             BLayout::name,
                             CLayout::name,
                             arg_parser.get_bool("structured_sparsity")};

    Setting setting{arg_parser.get_int("warmup"),
                    arg_parser.get_int("repeat"),
                    arg_parser.get_bool("timer"),
                    arg_parser.get_int("verify"),
                    arg_parser.get_int("init"),
                    arg_parser.get_bool("log"),
                    arg_parser.get_str("csv_filename"),
                    arg_parser.get_bool("flush_cache"),
                    arg_parser.get_int("rotating_count"),
                    arg_parser.get_int("bench_time")};

    auto& profiler = GemmProfiler::instance(setting);

    try
    {
        auto kernel_func = get_kernel_func_by_trait(arg_parser);
        profiler.benchmark(gemm_problem, kernel_func);
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;
        benchmark_gemm(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
