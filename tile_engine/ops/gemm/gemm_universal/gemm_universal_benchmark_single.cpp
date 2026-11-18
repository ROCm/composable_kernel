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
#include "gemm_universal_profiler.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME

// Returns false when an exact verification (-verify_exact 1) found mismatches.
bool benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    // Use DataTypeTraits to get the actual type names from the generated header
    // The generated header defines ADataType, BDataType, AccDataType, CDataType
    std::string dtype_a   = ck_tile::DataTypeTraits<ADataType>::name;
    std::string dtype_b   = ck_tile::DataTypeTraits<BDataType>::name;
    std::string dtype_acc = ck_tile::DataTypeTraits<AccDataType>::name;
    std::string dtype_c   = ck_tile::DataTypeTraits<CDataType>::name;

    // Layout names from the layout types
    std::string layout_a = ALayout::name;
    std::string layout_b = BLayout::name;
    std::string layout_c = CLayout::name;

    // Create GemmProblem struct
    GemmProblem gemm_problem{arg_parser.get_int("split_k"),
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
                             arg_parser.get_bool("structured_sparsity")};

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
    auto& profiler = UniversalGemmProfiler::GemmProfiler::instance(setting);
    profiler.set_verify_exact(arg_parser.get_int("verify_exact") != 0);

    try
    {
        // Create a lambda that wraps the kernel launch
        auto kernel_func = [](const ck_tile::GemmHostArgs& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        // Benchmark the kernel
        profiler.benchmark(gemm_problem, kernel_func);

        // Select best instance based on metric
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
    return !profiler.exact_verification_failed();
}

// -verify_exact 1 needs integer inputs (-init 4), a reference (-verify 1 or 2) and a
// round-to-nearest-even float -> bf16 conversion on both reference paths.
bool check_verify_exact_args(const ck_tile::ArgParser& arg_parser)
{
    if(arg_parser.get_int("verify_exact") == 0)
        return true;
    if(arg_parser.get_int("init") != 4 || arg_parser.get_int("verify") == 0)
    {
        std::cerr << "Error: -verify_exact 1 requires -init 4 (integer inputs) and -verify 1 "
                     "or 2"
                  << std::endl;
        return false;
    }
    return check_bf16_c_rounding_is_rne();
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv, 2, [](ck_tile::ArgParser& p) {
            p.insert("verify_exact",
                     "0",
                     "Set to 1 to also require an exact match (rtol = atol = 0) and report the "
                     "mismatch count. Needs -init 4 and -verify 1 or 2. Default is 0.");
        });
        if(!result)
            return EXIT_FAILURE;
        if(!check_verify_exact_args(parser))
            return EXIT_FAILURE;

        return benchmark_single(parser) ? 0 : EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
