// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "gemm_splitk_two_stage_invoker.hpp"

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    using Invoker = SplitKTwoStageInvoker;

    if(data_type != "bf16")
    {
        std::cout << "WARNING: Ignoring prec option for splitk two stage, and using bf16."
                  << std::endl;
    }

    return run_gemm_example_prec_type<GemmConfigTwoStage<ck_tile::bf16_t, float>,
                                      Invoker,
                                      ck_tile::bf16_t>(a_layout, b_layout, arg_parser);
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
