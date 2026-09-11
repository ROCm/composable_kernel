// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include "batched_gemm_profiler.hpp"
#include "batched_gemm_common.hpp"

#ifdef BATCHED_GEMM_SINGLE_INSTANCE_HPP
#include BATCHED_GEMM_SINGLE_INSTANCE_HPP
#endif

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C. Default is 0.")
        .insert("batch_stride_a", "0", "Batch stride for tensor A. Default is 0.")
        .insert("batch_stride_b", "0", "Batch stride for tensor B. Default is 0.")
        .insert("batch_stride_c", "0", "Batch stride for tensor C. Default is 0.")
        .insert("batch_count", "8", "Batch count. Default is 8.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "1",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 "
                "for validation on GPU. Default is 1, CPU validation.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Possible values are true or "
                "false. Default is false.")
        .insert("warmup",
                "50",
                "The number of iterations before benchmarking the kernel. Default is 50.")
        .insert(
            "repeat", "100", "The number of iterations to benchmark the kernel. Default is 100.")
        .insert("timer", "true", "Whether if the timer is GPU timer. Values: true or false.")
        .insert("init",
                "0",
                "Tensor initialization. 0: random, 1: linear, 2: constant(1). Default is 0.")
        .insert("flush_cache",
                "true",
                "To flush cache, possible values are true or false. Default is true.")
        .insert(
            "rotating_count", "1000", "Number of iterations to rotate the cache. Default is 1000.")
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
                "Default is false.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    std::string dtype_a   = DataTypeTraits<ADataType>::name;
    std::string dtype_b   = DataTypeTraits<BDataType>::name;
    std::string dtype_acc = DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = DataTypeTraits<CDataType>::name;

    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_c = CLayout::name;

    GemmProblem gemm_problem{arg_parser.get_int("split_k"),
                             arg_parser.get_int("m"),
                             arg_parser.get_int("n"),
                             arg_parser.get_int("k"),
                             arg_parser.get_int("stride_a"),
                             arg_parser.get_int("stride_b"),
                             arg_parser.get_int("stride_c"),
                             arg_parser.get_int("batch_stride_a"),
                             arg_parser.get_int("batch_stride_b"),
                             arg_parser.get_int("batch_stride_c"),
                             arg_parser.get_int("batch_count"),
                             dtype_a,
                             dtype_b,
                             dtype_acc,
                             dtype_c,
                             layout_a,
                             layout_b,
                             layout_c,
                             false};

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

    auto& profiler = GemmProfiler::instance(setting);

    try
    {
        auto kernel_func = [](const ck_tile::BatchedGemmHostArgs& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

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

        benchmark_single(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
