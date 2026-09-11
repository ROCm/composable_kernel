// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <exception>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_tensor_quant_profiler.hpp"

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    GemmTensorQuantProblem gemm_problem{};
    gemm_problem.split_k_  = arg_parser.get_int("split_k");
    gemm_problem.m_        = arg_parser.get_int("m");
    gemm_problem.n_        = arg_parser.get_int("n");
    gemm_problem.k_        = arg_parser.get_int("k");
    gemm_problem.stride_a_ = arg_parser.get_int("stride_a");
    gemm_problem.stride_b_ = arg_parser.get_int("stride_b");
    gemm_problem.stride_c_ = arg_parser.get_int("stride_c");

    gemm_problem.dtype_a_             = ck_tile::DataTypeTraits<ADataType>::name;
    gemm_problem.dtype_b_             = ck_tile::DataTypeTraits<BDataType>::name;
    gemm_problem.dtype_acc_           = ck_tile::DataTypeTraits<AccDataType>::name;
    gemm_problem.dtype_c_             = ck_tile::DataTypeTraits<CDataType>::name;
    gemm_problem.layout_a_            = ALayout::name;
    gemm_problem.layout_b_            = BLayout::name;
    gemm_problem.layout_c_            = CLayout::name;
    gemm_problem.structured_sparsity_ = false;

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

    auto& profiler = GemmTensorQuantProfiler::instance(setting);

    try
    {
        auto kernel_func = [](const ck_tile::QuantGemmHostArgs& args,
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
        auto [result, parser] = create_args(argc, argv, 1);
        if(!result)
        {
            return EXIT_FAILURE;
        }

        benchmark_single(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
