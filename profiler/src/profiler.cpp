#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

int profile_gemm(int, char*[]);
int profile_batched_gemm(int, char*[]);
int profile_gemm_bias_relu(int, char*[]);
int profile_gemm_bias_relu_add(int, char*[]);
int profile_conv_fwd(int, char*[]);
int profile_conv_fwd_bias_relu(int, char*[]);
int profile_conv_fwd_bias_relu_add(int, char*[]);
int profile_conv_fwd_bias_relu_atomic_add(int, char*[]);

int main(int argc, char* argv[])
{
    if(strcmp(argv[1], "gemm") == 0)
    {
        return profile_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu") == 0)
    {
        return profile_gemm_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu_add") == 0)
    {
        return profile_gemm_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "batched_gemm") == 0)
    {
        return profile_batched_gemm(argc, argv);
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
    else if(strcmp(argv[1], "conv_fwd_bias_relu_atomic_add") == 0)
    {
        return profile_conv_fwd_bias_relu_atomic_add(argc, argv);
    }
    else
    {
        // clang-format off
        printf("arg1: tensor operation (gemm: GEMM\n"
               "                        gemm_bias_relu: GEMM+Bias+ReLU\n"
               "                        gemm_bias_relu_add: GEMM+Bias+ReLU+Add\n"
               "                        conv_fwd: ForwardConvolution\n"
               "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU\n"
               "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add\n"
               "                        conv_fwd_bias_relu_atomic_add: ForwardConvolution+Bias+ReLU+AtomicAdd\n");
        // clang-format on

        return 0;
    }
}
