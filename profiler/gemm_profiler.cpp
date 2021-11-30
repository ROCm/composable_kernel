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
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl.hpp"
#include "profile_gemm.hpp"

enum GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
    MK_KN_NM, // 4
    MK_NK_NM, // 5
    KM_KN_NM, // 6
    KM_NK_NM, // 7
};

enum GemmDataType
{
    F32_F32_F32, // 0
    F16_F16_F16, // 1
};

int gemm_profiler(int argc, char* argv[])
{
    if(argc != 14)
    {
        printf("arg1: tensor operation (gemm: GEMM)\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, n] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, n] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg7: run kernel # of times (>1)\n");
        printf("arg8 to 13: M, N, K, StrideA, StrideB, StrideC\n");
        exit(1);
    }

    const int data_type        = static_cast<GemmDataType>(std::stoi(argv[2]));
    const int layout           = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const int nrepeat          = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideC = std::stoi(argv[13]);

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? K : StrideA,
                                                                      (StrideB < 0) ? N : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? K : StrideA,
                                                                      (StrideB < 0) ? K : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? M : StrideA,
                                                                      (StrideB < 0) ? N : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm<ck::half_t,
                                   ck::half_t,
                                   ck::half_t,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? M : StrideA,
                                                                      (StrideB < 0) ? K : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? K : StrideA,
                                                                      (StrideB < 0) ? N : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? K : StrideA,
                                                                      (StrideB < 0) ? K : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? M : StrideA,
                                                                      (StrideB < 0) ? N : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm<float,
                                   float,
                                   float,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::ColumnMajor,
                                   ck::tensor_layout::gemm::RowMajor>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      nrepeat,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      (StrideA < 0) ? M : StrideA,
                                                                      (StrideB < 0) ? K : StrideB,
                                                                      (StrideC < 0) ? N : StrideC);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }

    return 1;
}
