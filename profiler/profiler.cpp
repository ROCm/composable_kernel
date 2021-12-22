#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

extern int gemm_profiler(int, char*[]);
extern int conv_profiler(int, char*[]);
extern int reduce_profiler(int, char*[]); 

int main(int argc, char* argv[])
{
    if(strcmp(argv[1], "gemm") == 0)
    {
        return gemm_profiler(argc, argv);
    }
    else if(strcmp(argv[1], "conv") == 0)
    {
        return conv_profiler(argc, argv);
    }
    else if(strcmp(argv[1], "reduce") == 0)
    {
	return reduce_profiler(argc, argv); 
    }
    else
    {
        printf("arg1: tensor operation (gemm=GEMM, conv=Convolution, reduce-Reduction)\n");
        return 0;
    }
}

