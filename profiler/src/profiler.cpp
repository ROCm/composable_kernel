// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstring>

int profile_gemm(int, char*[]);
int profile_gemm_splitk(int, char*[]);
int profile_gemm_bilinear(int, char*[]);
int profile_gemm_add_add_fastgelu(int, char*[]);
int profile_gemm_reduce(int, char*[]);
int profile_gemm_bias_add_reduce(int, char*[]);
int profile_batched_gemm(int, char*[]);
int profile_batched_gemm_reduce(int, char*[]);
int profile_grouped_gemm(int, char*[]);
int profile_conv_fwd(int, char*[]);
int profile_conv_fwd_bias_relu(int, char*[]);
int profile_conv_fwd_bias_relu_add(int, char*[]);
int profile_conv_bwd_data(int, char*[]);
int profile_conv_bwd_weight(int, char*[]);
int profile_grouped_conv_fwd(int, char*[]);
int profile_normalization(int, char*[]);
int profile_reduce(int, char*[]);

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
           "                        batched_gemm_reduce: Batched GEMM+Reduce\n"
           "                        grouped_gemm: Grouped GEMM\n"
           "                        conv_fwd: Convolution Forward\n"
           "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU\n"
           "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add\n"
           "                        conv_bwd_data: Convolution Backward Data\n"
           "                        conv_bwd_weight: Convolution Backward Weight\n"
           "                        grouped_conv_fwd: Grouped Convolution Forward\n"
           "                        reduce: Reduce\n");
    // clang-format on
}

int main(int argc, char* argv[])
{
    if(argc == 1)
    {
        print_helper_message();

        return 0;
    }
    else if(strcmp(argv[1], "gemm") == 0)
    {
        return profile_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_splitk") == 0)
    {
        return profile_gemm_splitk(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bilinear") == 0)
    {
        return profile_gemm_bilinear(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_add_add_fastgelu") == 0)
    {
        return profile_gemm_add_add_fastgelu(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_reduce") == 0)
    {
        return profile_gemm_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_add_reduce") == 0)
    {
        return profile_gemm_bias_add_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "batched_gemm") == 0)
    {
        return profile_batched_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "batched_gemm_reduce") == 0)
    {
        return profile_batched_gemm_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "grouped_gemm") == 0)
    {
        return profile_grouped_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd") == 0)
    {
        return profile_conv_fwd(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu") == 0)
    {
        return profile_conv_fwd_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu_add") == 0)
    {
        return profile_conv_fwd_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "conv_bwd_data") == 0)
    {
        return profile_conv_bwd_data(argc, argv);
    }
    else if(strcmp(argv[1], "conv_bwd_weight") == 0)
    {
        return profile_conv_bwd_weight(argc, argv);
    }
    else if(strcmp(argv[1], "grouped_conv_fwd") == 0)
    {
        return profile_grouped_conv_fwd(argc, argv);
    }
    else if(strcmp(argv[1], "reduce") == 0)
    {
        return profile_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "batchnorm") == 0 || strcmp(argv[1], "layernorm") == 0 ||
            strcmp(argv[1], "softmax") == 0)
    {
        return profile_normalization(argc, argv);
    }
    else
    {
        print_helper_message();

        return 0;
    }
}
