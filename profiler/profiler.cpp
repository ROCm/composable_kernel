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
#if 1
    return gemm_profiler(argc, argv);
#else
    return conv_profiler(argc, argv);
#endif
}
