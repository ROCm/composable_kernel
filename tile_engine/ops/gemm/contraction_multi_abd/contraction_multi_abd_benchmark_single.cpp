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
#include "contraction_multi_abd_profiler.hpp"
#include "contraction_multi_abd_common.hpp"

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("g_dims", "2", "G dimensions separated by comma. Default is 2.")
        .insert("m_dims", "256", "M dimensions separated by comma. Default is 256.")
        .insert("n_dims", "128", "N dimensions separated by comma. Default is 128.")
        .insert("k_dims", "64", "K dimensions separated by comma. Default is 64.")
        .insert("verify", "1", "For validation. Default is 1, validation on CPU.")
        .insert("log", "false", "Whether to output kernel instance information. Default is false.")
        .insert("warmup", "50", "Number of warmup iterations. Default is 50.")
        .insert("repeat", "100", "Number of benchmark iterations. Default is 100.")
        .insert("timer", "true", "Whether to use GPU timer. Default is true.")
        .insert("init",
                "0",
                "Tensor initialization method. 0=random, 1=linear, 2=constant(1). Default is 0.")
        .insert(
            "metric", "0", "Performance metric. 0=latency, 1=tflops, 2=bandwidth. Default is 0.")
        .insert("csv_filename", "", "CSV output filename. Default is empty (no CSV).")
        .insert("json_output",
                "false",
                "Whether to output results in JSON format only. Default is false.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    auto g_dims = parse_dimensions(arg_parser.get_str("g_dims"));
    auto m_dims = parse_dimensions(arg_parser.get_str("m_dims"));
    auto n_dims = parse_dimensions(arg_parser.get_str("n_dims"));
    auto k_dims = parse_dimensions(arg_parser.get_str("k_dims"));

    ContractionMultiABDProblem problem;
    problem.g_dims_  = g_dims;
    problem.m_dims_  = m_dims;
    problem.n_dims_  = n_dims;
    problem.k_dims_  = k_dims;
    problem.g_total_ = calculate_total(g_dims);
    problem.m_total_ = calculate_total(m_dims);
    problem.n_total_ = calculate_total(n_dims);
    problem.k_total_ = calculate_total(k_dims);

    Settings setting{arg_parser.get_int("warmup"),
                     arg_parser.get_int("repeat"),
                     arg_parser.get_bool("timer"),
                     arg_parser.get_int("verify"),
                     arg_parser.get_int("init"),
                     arg_parser.get_bool("log"),
                     arg_parser.get_str("csv_filename"),
                     /*flush_cache=*/false,
                     /*rotating_count=*/0,
                     arg_parser.get_bool("json_output")};

    auto& profiler = ContractionMultiABDProfiler::instance(setting);

    try
    {
        auto kernel_func = [](const ck_tile::BatchedContractionMultiABDHostArgs<NumDimG,
                                                                                NumDimM,
                                                                                NumDimN,
                                                                                NumDimK,
                                                                                NumATensor,
                                                                                NumBTensor,
                                                                                NumDTensor>& args,
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
