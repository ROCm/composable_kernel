// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"
#include "grouped_convolution_backward_weight_invoker.hpp"
#include "run_grouped_convolution_bwd_weight_example.inc"

template <typename GemmWarpConfig, typename GemmTileConfig, typename GemmVectorLoads>
int run_grouped_conv_bwd_weight_example(ck_tile::ArgParser& arg_parser)
{
    using Invoker = GroupedConvolutionBackwardWeightInvoker;

    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");
    ck_tile::index_t num_groups_to_merge = arg_parser.get_int("num_groups_to_merge");

    if(data_type == "fp16")
    {
        // The undefined (negative) value corresponds to the number of
        // merged groups equal to unity, i.e., no merged groups.
        if (num_groups_to_merge <= 1)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 1,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 2)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 2,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 4)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 4,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 8)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 8,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 16)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 16,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 32)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 32,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 64)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 64,
                                                                 ck_tile::half_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported number of groups to merge! The number of groups should be a power of two and at most 64.");
        }
    }
    else if(data_type == "bf16")
    {
        // The undefined (negative) value corresponds to the number of
        // merged groups equal to unity, i.e., no merged groups.
        if (num_groups_to_merge <= 1)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 1,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 2)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 2,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 4)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 4,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 8)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 8,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 16)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 16,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 32)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 32,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else if (num_groups_to_merge == 64)
        {
            return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                                 GemmWarpConfig,
                                                                 GemmTileConfig,
                                                                 GemmVectorLoads,
                                                                 64,
                                                                 ck_tile::bf16_t>(
                in_layout, wei_layout, out_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported number of groups to merge! The number of groups should be a power of two and at most 64.");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation!");
    }
}

int main(int argc, char* argv[])
{

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    try
    {
#if CK_TILE_USE_WMMA
        return !run_grouped_conv_bwd_weight_example<
            GemmWarpConfig_Wmma
            GemmTileConfig,
            GemmVectorLoads>(arg_parser);
#else
        ck_tile::index_t num_groups_to_merge = arg_parser.get_int("num_groups_to_merge");
        if (num_groups_to_merge < 1)
        {
            // By default, we have the "num_groups_to_merge" set to -1, 
            // which means we will run the example with the default config.
            return !run_grouped_conv_bwd_weight_example<
                GemmWarpConfig_Mfma,
                GemmTileConfig,
                GemmVectorLoads>(arg_parser);
        }  
        else
        {
            // If the user specifies the "num_groups_to_merge" argument,
            // we will run the example with the merged groups config.
            // The tile size are selected such that we have number of 
            // merged groups any power of two smaller or equal to 64. 
            return !run_grouped_conv_bwd_weight_example<
                GemmWarpConfig_Mfma_merged_groups,
                GemmTileConfig_merged_groups,
                GemmVectorLoads_merged_groups>(arg_parser);
        }
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
