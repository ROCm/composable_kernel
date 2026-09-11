// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <functional>
#include <tuple>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "batched_contraction_profiler.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME, NUM_D_TENSORS,
// NUM_DIM_G, NUM_DIM_M, NUM_DIM_N, NUM_DIM_K,
// ADataType, BDataType, DBaseDataType, AccDataType, EDataType,
// ALayout, BLayout, ELayout, DsDataType, DsLayout, CDEElementWise

// Create argument parser
inline auto batched_contraction_create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("g_dims", "2", "G dimensions separated by comma (e.g., '4,2' for 2D batch)")
        .insert("m_dims", "256", "M dimensions separated by comma (e.g., '16,32' for 2D M)")
        .insert("n_dims", "128", "N dimensions separated by comma (e.g., '32,32' for 2D N)")
        .insert("k_dims", "64", "K dimensions separated by comma (e.g., '64,32' for 2D K)")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify", "1", "for validation on CPU. Default is 1.")
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
                "Default is false");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    // Use ck_tile::DataTypeTraits to get the actual type names from the generated header
    std::string dtype_a   = ck_tile::DataTypeTraits<ADataType>::name;
    std::string dtype_b   = ck_tile::DataTypeTraits<BDataType>::name;
    std::string dtype_acc = ck_tile::DataTypeTraits<AccDataType>::name;
    std::string dtype_e   = ck_tile::DataTypeTraits<EDataType>::name;
    std::string dtype_d   = ck_tile::DataTypeTraits<DBaseDataType>::name;

    // Layout names from the layout types
    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_e = ELayout::name;

    // Parse dimension strings
    auto g_dims = parse_dims_string(arg_parser.get_str("g_dims"));
    auto m_dims = parse_dims_string(arg_parser.get_str("m_dims"));
    auto n_dims = parse_dims_string(arg_parser.get_str("n_dims"));
    auto k_dims = parse_dims_string(arg_parser.get_str("k_dims"));

    // Validate dimension counts match the compiled kernel
    auto validate_dims = [](const std::vector<int>& dims, int expected, const std::string& name) {
        if(static_cast<int>(dims.size()) != expected)
            throw std::runtime_error("Expected " + std::to_string(expected) + " " + name +
                                     " dimension(s), got " + std::to_string(dims.size()));
        for(int v : dims)
            if(v <= 0)
                throw std::runtime_error(name + " dimensions must be positive, got " +
                                         std::to_string(v));
    };
    validate_dims(g_dims, NUM_DIM_G, "G");
    validate_dims(m_dims, NUM_DIM_M, "M");
    validate_dims(n_dims, NUM_DIM_N, "N");
    validate_dims(k_dims, NUM_DIM_K, "K");

    // Create BatchedContractionProblem struct
    BatchedContractionProblem problem;
    problem.split_k_       = arg_parser.get_int("split_k");
    problem.g_dims_        = g_dims;
    problem.m_dims_        = m_dims;
    problem.n_dims_        = n_dims;
    problem.k_dims_        = k_dims;
    problem.num_d_tensors_ = NUM_D_TENSORS;
    problem.dtype_a_       = dtype_a;
    problem.dtype_b_       = dtype_b;
    problem.dtype_d_       = dtype_d;
    problem.dtype_acc_     = dtype_acc;
    problem.dtype_e_       = dtype_e;
    problem.layout_a_      = layout_a;
    problem.layout_b_      = layout_b;
    problem.layout_e_      = layout_e;

    // Create Settings struct
    Settings setting{arg_parser.get_int("warmup"),
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
    auto& profiler = BatchedContractionProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
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
        auto [result, parser] = batched_contraction_create_args(argc, argv);
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
