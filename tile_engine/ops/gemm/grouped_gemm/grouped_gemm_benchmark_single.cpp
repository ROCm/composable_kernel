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
#include "grouped_gemm_profiler.hpp"
#include "grouped_gemm_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME
// DataTypeTraits are now defined in grouped_gemm_common.hpp

// Create argument parser
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("m", "3840", "Default M for all groups (if Ms not specified). Default is 3840.")
        .insert("n", "4096", "Default N for all groups (if Ns not specified). Default is 4096.")
        .insert("k", "2048", "Default K for all groups (if Ks not specified). Default is 2048.")
        .insert("Ms", "", "Comma-separated M dimensions per group.")
        .insert("Ns", "", "Comma-separated N dimensions per group.")
        .insert("Ks", "", "Comma-separated K dimensions per group.")
        .insert("stride_As", "", "Comma-separated stride values for tensor A per group.")
        .insert("stride_Bs", "", "Comma-separated stride values for tensor B per group.")
        .insert("stride_Cs", "", "Comma-separated stride values for tensor C per group.")
        .insert("group_count", "8", "Number of groups. Default is 8.")
        .insert("kbatch", "1", "SplitK batch count. Default is 1.")
        .insert("verify",
                "2",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, "
                "2 for validation on GPU. Default is 2, GPU validation.")
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
                "To flush cache, possible values are true or false. "
                "Default is true.")
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
                "Default is "
                "false");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    // Use DataTypeTraits to get the actual type names from the generated header
    std::string dtype_a   = DataTypeTraits<ADataType>::name;
    std::string dtype_b   = DataTypeTraits<BDataType>::name;
    std::string dtype_acc = DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = DataTypeTraits<CDataType>::name;

    // Layout names from the layout types
    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_c = CLayout::name;

    const int group_count = arg_parser.get_int("group_count");
    const int kbatch      = arg_parser.get_int("kbatch");

    // Parse per-group dimensions
    std::vector<int> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<int> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<int> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<int> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<int> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<int> stride_Cs = arg_parser.get_int_vec("stride_Cs");

    // If Ms/Ns/Ks not provided or wrong size, use -m/-n/-k defaults for all groups
    const auto gc_size = static_cast<std::size_t>(group_count);

    if(group_count == 0 || Ms.size() != gc_size || Ns.size() != gc_size || Ks.size() != gc_size)
    {
        const int default_m = arg_parser.get_int("m");
        const int default_n = arg_parser.get_int("n");
        const int default_k = arg_parser.get_int("k");

        Ms.assign(group_count, default_m);
        Ns.assign(group_count, default_n);
        Ks.assign(group_count, default_k);
    }

    // Default stride vectors to 0 independently if missing or wrong size
    if(stride_As.size() != gc_size)
        stride_As.assign(group_count, 0);
    if(stride_Bs.size() != gc_size)
        stride_Bs.assign(group_count, 0);
    if(stride_Cs.size() != gc_size)
        stride_Cs.assign(group_count, 0);

    // Create GroupedGemmProblem struct
    GroupedGemmProblem problem{group_count,
                               kbatch,
                               Ms,
                               Ns,
                               Ks,
                               stride_As,
                               stride_Bs,
                               stride_Cs,
                               dtype_a,
                               dtype_b,
                               dtype_acc,
                               dtype_c,
                               layout_a,
                               layout_b,
                               layout_c};

    // Create Setting struct
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

    // Get the profiler instance
    auto& profiler = GroupedGemmProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const std::vector<ck_tile::GroupedGemmHostArgs<>>& descs,
                              const ck_tile::stream_config& stream,
                              void* kargs_ptr) {
            return SelectedKernel::launch(descs, stream, kargs_ptr);
        };

        // Benchmark the kernel
        profiler.benchmark(problem, kernel_func);

        // Select best instance based on metric
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
