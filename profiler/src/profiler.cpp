// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>

#include "profiler_operation_registry.hpp"

static void print_helper_message()
{
    // clang-format off
    printf("arg1: tensor operation (gemm: GEMM\n"
           "                        gemm_splitk: Split-K GEMM\n"
           "                        gemm_bilinear: GEMM+Bilinear\n"
           "                        gemm_add_add_fastgelu: GEMM+Add+Add+FastGeLU\n"
           "                        gemm_reduce: GEMM+Reduce\n"
           "                        gemm_bias_add_reduce: GEMM+Bias+Add+Reduce\n"
           "                        batched_gemm: Batched GEMM\n"
           "                        batched_gemm_gemm: Batched+GEMM+GEMM\n"
           "                        batched_gemm_add_relu_gemm_add: Batched+GEMM+bias+gelu+GEMM+bias\n"
           "                        batched_gemm_reduce: Batched GEMM+Reduce\n"
           "                        grouped_gemm: Grouped GEMM\n"
           "                        conv_fwd: Convolution Forward\n"
           "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU\n"
           "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add\n"
           "                        conv_bwd_data: Convolution Backward Data\n"
           "                        grouped_conv_fwd: Grouped Convolution Forward\n"
           "                        grouped_conv_bwd_weight: Grouped Convolution Backward Weight\n"
           "                        softmax: Softmax\n"
           "                        reduce: Reduce\n");
    // clang-format on
}

int main(int argc, char* argv[])
{
    if(argc == 1)
    {
        print_helper_message();
    }
    else if(auto operation = ProfilerOperationRegistry::GetInstance().Get(argv[1]); operation.has_value())
    {
        return (*operation)(argc, argv);
    } else {
        std::cerr << "cannot find operation: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
}
