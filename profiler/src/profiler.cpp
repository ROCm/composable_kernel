#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

bool profile_gemm(int, char*[]);
bool profile_gemm_splitk(int, char*[]);
bool profile_gemm_bias_2d(int, char*[]);
bool profile_gemm_bias_relu(int, char*[]);
bool profile_gemm_bias_relu_add(int, char*[]);
bool profile_gemm_reduce(int, char*[]);
bool profile_batched_gemm(int, char*[]);
bool profile_grouped_gemm(int, char*[]);
bool profile_conv_fwd_bias_relu(int, char*[]);
bool profile_conv_fwd_bias_relu_add(int, char*[]);
bool profile_convnd_fwd(int argc, char* argv[]);
bool profile_convnd_bwd_data(int, char*[], int);
bool profile_reduce(int, char*[]);
bool profile_conv_bwd_weight(int, char*[]);
bool profile_batched_gemm_reduce(int, char*[]);

int main(int argc, char* argv[])
{
    auto print_help_message = []() {
        // clang-format off
        printf("arg1: tensor operation, gemm: GEMM\n"
               "                        gemm_splitk: GEMM Split-K\n"
               "                        gemm_bias_2d: GEMM+Bias(2D)\n"
               "                        gemm_bias_relu: GEMM+Bias+ReLU\n"
               "                        gemm_bias_relu_add: GEMM+Bias+ReLU+Add\n"
               "                        gemm_reduce: GEMM+Reduce\n"
               "                        grouped_gemm: Grouped GEMM\n"
               "                        conv_fwd: Convolution Forward\n"
               "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU\n"
               "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add\n"
               "                        conv1d_bwd_data: Convolution Backward Data 1D\n"
               "                        conv2d_bwd_data: Convolution Backward Data 2D\n"
               "                        conv3d_bwd_data: Convolution Backward Data 3D\n"
               "                        reduce: Reduce\n"
               "                        conv2d_bwd_weight: Convolution Backward Weight 2D\n");
        // clang-format on
    };

    if(argc < 2)
    {
        print_help_message();
        exit(1);
    }

    bool pass = true;

    if(strcmp(argv[1], "gemm") == 0)
    {
        pass = profile_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_splitk") == 0)
    {
        pass = profile_gemm_splitk(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_2d") == 0)
    {
        pass = profile_gemm_bias_2d(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu") == 0)
    {
        pass = profile_gemm_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_bias_relu_add") == 0)
    {
        pass = profile_gemm_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "gemm_reduce") == 0)
    {
        pass = profile_gemm_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "batched_gemm") == 0)
    {
        pass = profile_batched_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "batched_gemm_reduce") == 0)
    {
        pass = profile_batched_gemm_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "grouped_gemm") == 0)
    {
        pass = profile_grouped_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd") == 0)
    {
        pass = profile_convnd_fwd(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu") == 0)
    {
        pass = profile_conv_fwd_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu_add") == 0)
    {
        pass = profile_conv_fwd_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "conv1d_bwd_data") == 0)
    {
        pass = profile_convnd_bwd_data(argc, argv, 1);
    }
    else if(strcmp(argv[1], "conv2d_bwd_data") == 0)
    {
        pass = profile_convnd_bwd_data(argc, argv, 2);
    }
    else if(strcmp(argv[1], "conv3d_bwd_data") == 0)
    {
        pass = profile_convnd_bwd_data(argc, argv, 3);
    }
    else if(strcmp(argv[1], "reduce") == 0)
    {
        pass = profile_reduce(argc, argv);
    }
    else if(strcmp(argv[1], "conv2d_bwd_weight") == 0)
    {
        pass = profile_conv_bwd_weight(argc, argv);
    }
    else
    {
        print_help_message();
    }

    return pass ? 0 : 1;
}
