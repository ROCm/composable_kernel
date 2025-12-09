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
#include "pool_profiler.hpp"
#include "pool_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME
// DataTypeTraits are defined in pool_common.hpp

// Create argument parser
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("N", "2", "Batch size N dimension. Default is 2.")
        .insert("D", "30", "Depth D dimension (for 3D pooling). Default is 30.")
        .insert("H", "30", "Height H dimension. Default is 30.")
        .insert("W", "30", "Width W dimension. Default is 30.")
        .insert("C", "32", "Channel C dimension. Default is 32.")
        .insert("Z", "2", "Window depth Z dimension. Default is 2.")
        .insert("Y", "2", "Window height Y dimension. Default is 2.")
        .insert("X", "2", "Window width X dimension. Default is 2.")
        .insert("Sz", "2", "Window stride depth. Default is 2.")
        .insert("Sy", "2", "Window stride height. Default is 2.")
        .insert("Sx", "2", "Window stride width. Default is 2.")
        .insert("Dz", "1", "Window dilation depth. Default is 1.")
        .insert("Dy", "1", "Window dilation height. Default is 1.")
        .insert("Dx", "1", "Window dilation width. Default is 1.")
        .insert("LeftPz", "0", "Left padding depth. Default is 0.")
        .insert("LeftPy", "0", "Left padding height. Default is 0.")
        .insert("LeftPx", "0", "Left padding width. Default is 0.")
        .insert("RightPz", "0", "Right padding depth. Default is 0.")
        .insert("RightPy", "0", "Right padding height. Default is 0.")
        .insert("RightPx", "0", "Right padding width. Default is 0.")
        .insert("pool_dim",
                "3",
                "Pooling dimension (2 for 2D, 3 for 3D). Default is 3.")
        .insert("verify",
                "1",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU. "
                "Default is 1, CPU validation.")
        .insert("log",
                "false",
                "Whether output kernel instance information or not. Possible values are true or "
                "false. Default is false")
        .insert(
            "warmup", "20", "The number of iterations before benchmark the kernel. Default is 20.")
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
        .insert("rotating_count", "1000", "Number of iterations to rotate the cache. Default is 1000.")
        .insert("metric",
                "2",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth. Default is 2, bandwidth.")
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
    // Use DataTypeTraits to get the actual type names from the generated header
    // The generated header defines InDataType, OutDataType, ComputeDataType, IndexDataType
    std::string inDType      = DataTypeTraits<InDataType>::name;
    std::string outDType     = DataTypeTraits<OutDataType>::name;
    std::string computeDType = DataTypeTraits<ComputeDataType>::name;
    std::string indexDType   = DataTypeTraits<IndexDataType>::name;

    // Get block shape from the generated kernel
    std::string blockShape = BLOCK_SHAPE_NAME;

    // Get reduce op from the generated kernel
    std::string reduceOp = REDUCE_OP_NAME;

    // Create PoolProblem struct
    PoolProblem pool_problem{
        inDType,
        outDType,
        computeDType,
        indexDType,
        blockShape,
        reduceOp,
        arg_parser.get_int("pool_dim"),
        arg_parser.get_int("N"),
        arg_parser.get_int("D"),
        arg_parser.get_int("H"),
        arg_parser.get_int("W"),
        arg_parser.get_int("C"),
        arg_parser.get_int("Z"),
        arg_parser.get_int("Y"),
        arg_parser.get_int("X"),
        arg_parser.get_int("Sz"),
        arg_parser.get_int("Sy"),
        arg_parser.get_int("Sx"),
        arg_parser.get_int("Dz"),
        arg_parser.get_int("Dy"),
        arg_parser.get_int("Dx"),
        arg_parser.get_int("LeftPz"),
        arg_parser.get_int("LeftPy"),
        arg_parser.get_int("LeftPx"),
        arg_parser.get_int("RightPz"),
        arg_parser.get_int("RightPy"),
        arg_parser.get_int("RightPx"),
        OUTPUT_INDEX,
        PROPAGATE_NAN};

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
    auto& profiler = PoolProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [&](const auto& args, const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        // Benchmark the kernel using the templated version
        profiler.template benchmark<TensorShapeType, WindowShapeType>(pool_problem, kernel_func);

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
