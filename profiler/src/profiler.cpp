#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

#include "profiler/include/profile_convnd_fwd.hpp"

int profile_gemm(int, char*[]);
int profile_gemm_bias_2d(int, char*[]);
int profile_gemm_bias_relu(int, char*[]);
int profile_gemm_bias_relu_add(int, char*[]);
int profile_gemm_reduce(int, char*[]);
int profile_gemm_bias_add_reduce(int, char*[]);
int profile_batched_gemm(int, char*[]);
int profile_grouped_gemm(int, char*[]);
int profile_conv_fwd(int, char*[]);
int profile_conv_fwd_bias_relu(int, char*[]);
int profile_conv_fwd_bias_relu_add(int, char*[]);
int profile_convnd_bwd_data(int, char*[], int);
int profile_reduce(int, char*[]);
int profile_conv_bwd_weight(int, char*[]);
int profile_batched_gemm_reduce(int, char*[]);
int profile_gemm_add_add_fastgelu(int, char*[]);

static void print_helper_message()
{
    // clang-format off
        printf("arg1: tensor operation (gemm: GEMM\n"
               "                        gemm_bias_2d: GEMM+Bias(2D)\n"
               "                        gemm_bias_relu: GEMM+Bias+ReLU\n"
               "                        gemm_bias_relu_add: GEMM+Bias+ReLU+Add\n"
               "                        gemm_reduce: GEMM+Reduce\n"
               "                        grouped_gemm: Grouped GEMM\n"
               "                        conv_fwd: ForwardConvolution\n"
               "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU\n"
               "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add\n"
               "                        conv1d_bwd_data: BackwardConvolution data 1 dim\n"
               "                        conv2d_bwd_data: BackwardConvolution data 2 dim\n"
               "                        conv3d_bwd_data: BackwardConvolution data 3 dim\n"
               "                        reduce: Reduce\n"
               "                        conv2d_bwd_weight: Backward Weight Convolution 2d\n"
               "                        gemm_add_add_fastgelu: GEMM+Add+Add+FastGeLU\n");
    // clang-format on
}

int main(int argc, char* argv[])
{
    if(argc == 1)
    {
        print_helper_message();

        return 0;
    }

    if(strcmp(argv[1], "gemm") == 0)
    {
        return profile_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_2d") == 0)
    {
        return profile_gemm_bias_2d(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu") == 0)
    {
        return profile_gemm_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu_add") == 0)
    {
        return profile_gemm_bias_relu_add(argc, argv);
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
        return ck::profiler::profile_convnd_fwd(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu") == 0)
    {
        return profile_conv_fwd_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu_add") == 0)
    {
        return profile_conv_fwd_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "conv1d_bwd_data") == 0)
    {
        return profile_convnd_bwd_data(argc, argv, 1);
    }
    else if(strcmp(argv[1], "conv2d_bwd_data") == 0)
    {
        return profile_convnd_bwd_data(argc, argv, 2);
    }
    else if(strcmp(argv[1], "conv3d_bwd_data") == 0)
    {
        return profile_convnd_bwd_data(argc, argv, 3);
    }
    else if(strcmp(argv[1], "reduce") == 0)
    {
        return profile_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "conv2d_bwd_weight") == 0)
    {
        return profile_conv_bwd_weight(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_add_add_fastgelu") == 0)
    {
        return profile_gemm_add_add_fastgelu(argc, argv);
    }
    else
    {
        print_helper_message();

        return 0;
    }
}
