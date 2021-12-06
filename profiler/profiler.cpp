#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

int profile_gemm(int, char*[]);
int profile_conv_fwd(int, char*[]);
// int profile_conv_fwd_bias_relu_add(int, char*[]);

int main(int argc, char* argv[])
{
    if(strcmp(argv[1], "gemm") == 0)
    {
        return profile_gemm(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd") == 0)
    {
        return profile_conv_fwd(argc, argv);
    }
#if 0
    else if(strcmp(argv[1], "conv_fwd_bias_relu_add") == 0)
    {
        return profile_conv_fwd_bias_relu_add(argc, argv);
    }
#endif
    else
    {
        printf("arg1: tensor operation (conv_fwd: ForwardConvolution)\n");
        return 0;
    }
}
