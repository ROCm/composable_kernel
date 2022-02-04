#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

// extern int profile_gemm(int, char*[]);
// extern int profile_conv_fwd(int, char*[]);
// extern int profile_conv_fwd_bias_relu(int, char*[]);
// extern int profile_conv_fwd_bias_relu_add(int, char*[]);
// extern int profile_conv_fwd_bias_relu_atomic_add(int, char*[]);
extern int reduce_profiler(int, char*[]);

int main(int argc, char* argv[])
{
    if(strcmp(argv[1], "gemm") == 0)
    {
        // return gemm_profiler(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd") == 0)
    {
        // return profile_conv_fwd(argc, argv);
    }
    else if(strcmp(argv[1], "reduce") == 0)
    {
        return reduce_profiler(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu") == 0)
    {
        // return profile_conv_fwd_bias_relu(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu_add") == 0)
    {
        // return profile_conv_fwd_bias_relu_add(argc, argv);
    }
    else if(strcmp(argv[1], "conv_fwd_bias_relu_atomic_add") == 0)
    {
        // return profile_conv_fwd_bias_relu_atomic_add(argc, argv);
    }
    else
    {
        printf("arg1: tensor operation (gemm: GEMM;\n"
               "                        conv_fwd: ForwardConvolution;\n"
               "                        conv_fwd_bias_relu: ForwardConvolution+Bias+ReLU)\n"
               "                        conv_fwd_bias_relu_add: ForwardConvolution+Bias+ReLU+Add)\n"
               "                        conv_fwd_bias_relu_atomic_add: "
               "ForwardConvolution+Bias+ReLU+AtomicAdd)\n");
        return 0;
    }
}
