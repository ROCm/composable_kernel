// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"
#include "gemm_host_api.hpp"
#include "benchmark_gemm.hpp"

auto run_single_trait(const ck_tile::ArgParser& arg_parser)
{
    KernelTraits trait;
    trait.pipeline  = arg_parser.get_str("pipeline");
    trait.scheduler = arg_parser.get_str("scheduler");
    trait.epilogue  = arg_parser.get_str("epilogue");
    trait.pad_m     = arg_parser.get_bool("pad_m");
    trait.pad_n     = arg_parser.get_bool("pad_n");
    trait.pad_k     = arg_parser.get_bool("pad_k");

    bool structured_sparsity = arg_parser.get_bool("structured_sparsity");

    return GemmDispatcher::dispatch(structured_sparsity, trait);
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;
        benchmark_gemm(parser, run_single_trait(parser));
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
