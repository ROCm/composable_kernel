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
#include "gemm_multi_abd_profiler.hpp"
#include "gemm_multi_abd_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME

// Create argument parser for multi ABD
// Multi ABD uses stride_as/stride_bs/stride_e (vectors/different naming)
// instead of the shared create_args stride_a/stride_b/stride_c,
// so a custom parser is necessary.
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_as", "0", "The stride value for A tensors. Default is 0.")
        .insert("stride_bs", "0", "The stride value for B tensors. Default is 0.")
        .insert("stride_ds", "0", "The stride value for D tensors. Default is 0.")
        .insert("stride_e", "0", "The stride value for tensor E. Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "1",
                "Default is 1, validation on CPU, as validation on GPU is "
                "not supported. 0 for no validation.")
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
    // Build vectors for dtypes (all A tensors share ADataType, etc.)
    std::vector<std::string> dtype_as(NumATensors, ck_tile::DataTypeTraits<ADataType>::name);
    std::vector<std::string> dtype_bs(NumBTensors, ck_tile::DataTypeTraits<BDataType>::name);
    std::vector<std::string> dtype_ds(NumDTensors, ck_tile::DataTypeTraits<DBaseDataType>::name);

    // Build vectors for layouts
    std::vector<std::string> layout_as(NumATensors, std::string(ALayout::name));
    std::vector<std::string> layout_bs(NumBTensors, std::string(BLayout::name));
    std::vector<std::string> layout_ds(NumDTensors, std::string(DLayout::name));

    // Build vectors for strides (same default stride for all tensors in each group)
    std::vector<int> stride_as(NumATensors, arg_parser.get_int("stride_as"));
    std::vector<int> stride_bs(NumBTensors, arg_parser.get_int("stride_bs"));
    std::vector<int> stride_ds(NumDTensors, arg_parser.get_int("stride_ds"));

    GemmMultiABDProblem problem{GemmProblem{arg_parser.get_int("split_k"),
                                            arg_parser.get_int("m"),
                                            arg_parser.get_int("n"),
                                            arg_parser.get_int("k"),
                                            /*stride_a_=*/arg_parser.get_int("stride_as"),
                                            /*stride_b_=*/arg_parser.get_int("stride_bs"),
                                            /*stride_c_=*/arg_parser.get_int("stride_e"),
                                            std::string(ck_tile::DataTypeTraits<ADataType>::name),
                                            std::string(ck_tile::DataTypeTraits<BDataType>::name),
                                            std::string(ck_tile::DataTypeTraits<AccDataType>::name),
                                            std::string(ck_tile::DataTypeTraits<EDataType>::name),
                                            std::string(ALayout::name),
                                            std::string(BLayout::name),
                                            std::string(ELayout::name),
                                            /*structured_sparsity_=*/false},
                                stride_as,
                                stride_bs,
                                stride_ds,
                                arg_parser.get_int("stride_e"),
                                dtype_as,
                                dtype_bs,
                                dtype_ds,
                                std::string(ck_tile::DataTypeTraits<EDataType>::name),
                                layout_as,
                                layout_bs,
                                layout_ds,
                                std::string(ELayout::name),
                                std::string(AElementWiseFn::name),
                                std::string(BElementWiseFn::name),
                                std::string(CDEElementWiseFn::name)};

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

    auto& profiler = GemmMultiABDProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func =
            [](const ck_tile::GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors>& args,
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
