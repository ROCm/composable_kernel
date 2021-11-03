#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl.hpp"
#include "gemm_profiler.hpp"

int main(int argc, char* argv[])
{
    if(argc != 13)
    {
        printf("arg1: data type (0=fp32, 1=fp16)\n");
        printf("arg2: matrix layout (0=NN, 1=NT, 2=TN, 3=TT)\n");
        printf("arg3: verification (0=no, 1=yes)\n");
        printf("arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg5: print matrix value (0=no, 1=yes)\n");
        printf("arg6: run kernel # of times (>1)\n");
        printf("arg7 to 12: M, N, K, StrideA, StrideB, StrideC\n");
        exit(1);
    }

    const int data_type        = static_cast<GemmDataType>(std::stoi(argv[1]));
    const int layout           = static_cast<GemmMatrixLayout>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const int nrepeat          = std::stoi(argv[6]);

    const int M = std::stoi(argv[7]);
    const int N = std::stoi(argv[8]);
    const int K = std::stoi(argv[9]);

    const int StrideA = std::stoi(argv[10]);
    const int StrideB = std::stoi(argv[11]);
    const int StrideC = std::stoi(argv[12]);

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }

    return 1;
}
