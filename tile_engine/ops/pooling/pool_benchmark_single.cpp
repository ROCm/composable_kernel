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
// DataTypeTraits are now defined in gemm_common.hpp

// Create argument parser TODO
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    // TODO

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    // Use DataTypeTraits to get the actual type names from the generated header
    // The generated header defines InDataType, OutDataType, ComputeDataType, IndexDataType
    std::string inDType   = DataTypeTraits<InDataType>::name;
    std::string outDType   = DataTypeTraits<OutDataType>::name;
    std::string computeDType = DataTypeTraits<ComputeDataType>::name;
    std::string indexDType   = DataTypeTraits<IndexDataType>::name;

    PoolProblem pool_problem{inDType,
                             outDType,
                             computeDType,
                             indexDType,
                             arg_parser.get_str("blockShape"),
                             arg_parser.get_str("reduceOp"),
                             arg_parser.get_bool("outputIndex"),
                             arg_parser.get_bool("propagateNan")};

    Settings settings{};

    // Get the profiler instance
    auto& profiler = PoolProfiler::instance(setting); // TODO

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const ck_tile::&PoolHostArgs args, // TODO
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        // Benchmark the kernel
        profiler.benchmark(pool_problem, kernel_func);

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
