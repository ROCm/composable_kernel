#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

int gemm_profiler(int, char*[]);
int conv_profiler(int, char*[]);

int main(int argc, char* argv[])
{
    if(strcmp(argv[1], "gemm") == 0)
    {
        return gemm_profiler(argc, argv);
    }
#if 0
    else if (strcmp(argv[1], "conv") == 0)
    {
        return conv_profiler(argc, argv);
    }
#endif
    else
    {
        // printf("arg1: tensor operation (gemm=GEMM, conv=Convolution)\n");
        printf("arg1: tensor operation (gemm=GEMM)\n");
    }
}
