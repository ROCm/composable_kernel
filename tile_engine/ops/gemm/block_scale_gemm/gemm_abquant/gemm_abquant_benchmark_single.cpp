// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <functional>
#include <tuple>
#include <exception>
#include <sstream>
#include <vector>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_abquant_profiler.hpp"
#include "gemm_abquant_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct, KERNEL_NAME, and type aliases:
//   ADataType, BDataType, AQDataType, BQDataType, AccDataType, CDataType
//   ALayout, BLayout, CLayout, AQLayout, BQLayout

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C. Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("group_size_k",
                std::to_string(SelectedKernel::GroupSizeK),
                "The quantization group size along K. Default matches kernel config.")
        .insert("group_size_n",
                std::to_string(SelectedKernel::GroupSizeN),
                "The quantization group size along N for BQ. Default matches kernel config.")
        .insert("verify",
                "1",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU. "
                "Default is 1, CPU validation.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Possible values are true or "
                "false. Default is false")
        .insert(
            "warmup", "50", "The number of iterations before benchmark the kernel. Default is 50.")
        .insert(
            "repeat", "100", "The number of iterations to benchmark the kernel. Default is 100.")
        .insert("timer",
                "true",
                "Whether if the timer is gpu timer or not. Possible values are false or true. "
                "Default is true.")
        .insert("init",
                "0",
                "The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 "
                "for constant(1). Default is 0, random.")
        .insert("flush_cache",
                "true",
                "To flush cache, possible values are true or false. Default is true.")
        .insert(
            "rotating_count", "1000", "number of iterations to rotate the cache. default is 1000.")
        .insert("metric",
                "0",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth. Default is 0, latency.")
        .insert("csv_filename",
                "",
                "The filename of benchmark result. Default is empty (no CSV output).")
        .insert("json_output",
                "false",
                "Whether to output results in JSON format only. Possible values are true or false. "
                "Default is false");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    std::string dtype_a   = DataTypeTraits<ADataType>::name;
    std::string dtype_b   = DataTypeTraits<BDataType>::name;
    std::string dtype_aq  = DataTypeTraits<AQDataType>::name;
    std::string dtype_bq  = DataTypeTraits<BQDataType>::name;
    std::string dtype_acc = DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = DataTypeTraits<CDataType>::name;

    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_c = CLayout::name;

    int M            = arg_parser.get_int("m");
    int N            = arg_parser.get_int("n");
    int K            = arg_parser.get_int("k");
    int group_size_k = arg_parser.get_int("group_size_k");
    int group_size_n = arg_parser.get_int("group_size_n");

    if(M <= 0 || N <= 0 || K <= 0)
    {
        throw std::invalid_argument("m, n, k must be positive integers");
    }
    if(group_size_k <= 0 || K % group_size_k != 0)
    {
        throw std::invalid_argument(
            "group_size_k must be positive and k must be divisible by group_size_k");
    }
    if(group_size_n > 1 && N % group_size_n != 0)
    {
        throw std::invalid_argument("n must be divisible by group_size_n when group_size_n > 1");
    }

    ABQuantGemmProblem problem{arg_parser.get_int("split_k"),
                               M,
                               N,
                               K,
                               arg_parser.get_int("stride_a"),
                               arg_parser.get_int("stride_b"),
                               arg_parser.get_int("stride_c"),
                               0, // stride_aq computed by profiler
                               0, // stride_bq computed by profiler
                               group_size_k,
                               group_size_n,
                               dtype_a,
                               dtype_b,
                               dtype_aq,
                               dtype_bq,
                               dtype_acc,
                               dtype_c,
                               layout_a,
                               layout_b,
                               layout_c};

    Setting setting{arg_parser.get_int("warmup"),
                    arg_parser.get_int("repeat"),
                    arg_parser.get_bool("timer"),
                    arg_parser.get_int("verify"),
                    arg_parser.get_int("init"),
                    arg_parser.get_bool("log"),
                    arg_parser.get_str("csv_filename"),
                    arg_parser.get_bool("flush_cache"),
                    arg_parser.get_int("rotating_count"),
                    arg_parser.get_bool("json_output")};

    auto& profiler = ABQuantGemmProfiler::instance(setting);

    try
    {
        auto kernel_func = [](const ck_tile::QuantGemmHostArgs& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        profiler.benchmark(problem, kernel_func);
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

        benchmark_single(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
