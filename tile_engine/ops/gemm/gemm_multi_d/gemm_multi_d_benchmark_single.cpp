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
#include "gemm/gemm_common.hpp"
#include "gemm_multi_d_profiler.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    // Use DataTypeTraits to get the actual type names from the generated header
    // The generated header defines ADataType, BDataType, AccDataType, CDataType
    std::string dtype_a   = ck_tile::DataTypeTraits<ADataType>::name;
    std::string dtype_b   = ck_tile::DataTypeTraits<BDataType>::name;
    std::string dtype_acc = ck_tile::DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = ck_tile::DataTypeTraits<CDataType>::name;
    std::string dtype_d0  = ck_tile::DataTypeTraits<D0DataType>::name;
    std::string dtype_d1  = ck_tile::DataTypeTraits<D1DataType>::name;

    // Layout names from the layout types
    std::string layout_a  = ALayout::name;
    std::string layout_b  = BLayout::name;
    std::string layout_c  = CLayout::name;
    std::string layout_d0 = D0Layout::name;
    std::string layout_d1 = D1Layout::name;

    // Create GemmMultiDProblem struct
    GemmMultiDProblem gemm_multi_d_problem{GemmProblem{arg_parser.get_int("split_k"),
                                                       arg_parser.get_int("m"),
                                                       arg_parser.get_int("n"),
                                                       arg_parser.get_int("k"),
                                                       arg_parser.get_int("stride_a"),
                                                       arg_parser.get_int("stride_b"),
                                                       arg_parser.get_int("stride_c"),
                                                       dtype_a,
                                                       dtype_b,
                                                       dtype_acc,
                                                       dtype_c,
                                                       layout_a,
                                                       layout_b,
                                                       layout_c,
                                                       arg_parser.get_bool("structured_sparsity")},
                                           arg_parser.get_int("stride_ds"),
                                           arg_parser.get_int("stride_ds"),
                                           dtype_d0,
                                           dtype_d1,
                                           layout_d0,
                                           layout_d1};

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
    auto& profiler = GemmMultiDProfiler::instance(setting);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const ck_tile::GemmMultiDHostArgs<DsDataType::size()>& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        // Benchmark the kernel
        profiler.benchmark(gemm_multi_d_problem, kernel_func);

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
