#include "run.h"

int main(int argc, char* argv[]) 
{ 
    //return !run_gemm_example(argc, argv); 
    run_gemm_128_32_128_8_2(argc, argv);

    run_gemm_128_64_32_8_2(argc, argv);

    run_gemm_128_64_128_8_2(argc, argv);

    run_gemm_128_128_32_8_2(argc, argv);

    run_gemm_128_128_64_8_2(argc, argv);

    run_gemm_256_64_128_8_2(argc, argv);

    run_gemm_256_128_64_8_2(argc, argv);

    run_gemm_256_128_128_8_2(argc, argv);

    run_gemm_256_128_128_16_2(argc, argv);
}
