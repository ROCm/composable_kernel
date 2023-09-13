#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"

bool run_gemm_128_32_64_8_2(int argc, char* argv[]);

bool run_gemm_128_32_128_8_2(int argc, char* argv[]);

bool run_gemm_128_64_32_8_2(int argc, char* argv[]);

bool run_gemm_128_64_128_8_2(int argc, char* argv[]);

bool run_gemm_128_128_32_8_2(int argc, char* argv[]);

bool run_gemm_128_128_64_8_2(int argc, char* argv[]);

bool run_gemm_256_64_128_8_2(int argc, char* argv[]);

bool run_gemm_256_128_64_8_2(int argc, char* argv[]);

bool run_gemm_256_128_128_8_2(int argc, char* argv[]);

bool run_gemm_256_128_128_16_2(int argc, char* argv[]);