// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_convolution_utils.hpp"
#include "run_grouped_convolution_bwd_weight_example.inc"
#include "grouped_convolution_bwd_weight_two_stage_invoker.hpp"

int run_grouped_convolution_bwd_weight_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    using Invoker = SplitKTwoStageInvoker;

    if(data_type == "fp16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<GemmConfigTwoStage<ck_tile::half_t, float>,
                                          Invoker,
                                          ck_tile::half_t>(in_layout, wei_layout, out_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<GemmConfigTwoStage<ck_tile::bf16_t, float>,
                                          Invoker,
                                          ck_tile::bf16_t>(in_layout, wei_layout, out_layout, arg_parser);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation!");
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_grouped_convolution_bwd_data_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
