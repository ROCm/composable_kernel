// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/include/profile_batched_gemm_bias_gelu_gemm_bias_impl.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

int profile_batched_gemm_bias_gelu_gemm_bias(int argc, char* argv[])
{
    enum struct GemmMatrixLayout
    {
        MK_NK_MN_ON_MO, // 0
    };

    enum struct GemmDataType
    {
        F16_F16_F32_F16_F32_F16_F16, // 0
    };

    if(argc != 21)
    {
        printf("arg1: tensor operation (batched_gemm_bias_gelu_gemm_bias: "
               "Batched+GEMM+bias+gelu+Gemm+bias)\n");
        printf("arg2: data type (0: fp16)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[n, k] = C[m, n];\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf(
            "arg8 to 20: M, N, K, O, Batch, StrideA0, StrideB0, StrideB1, StrideC1, BatchStrideA0, "
            "BatchStrideB0, BatchStrideB1, BatchStrideC1 \n");
        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M          = std::stoi(argv[8]);
    const int N          = std::stoi(argv[9]);
    const int K          = std::stoi(argv[10]);
    const int O          = std::stoi(argv[11]);
    const int BatchCount = std::stoi(argv[12]);

    const int StrideA0 = std::stoi(argv[13]);
    const int StrideB0 = std::stoi(argv[14]);
    const int StrideB1 = std::stoi(argv[15]);
    const int StrideC1 = std::stoi(argv[16]);

    const int BatchStrideA0 = std::stoi(argv[17]);
    const int BatchStrideB0 = std::stoi(argv[18]);
    const int BatchStrideB1 = std::stoi(argv[19]);
    const int BatchStrideC1 = std::stoi(argv[20]);

    if(data_type == GemmDataType::F16_F16_F32_F16_F32_F16_F16 &&
       layout == GemmMatrixLayout::MK_NK_MN_ON_MO)
    {
        ck::profiler::profile_batched_gemm_bias_gelu_gemm_bias_impl<Row,            // ALayout,
                                                                    Col,            // B0Layout,
                                                                    Row,            // D0Layout,
                                                                    Row,            // B1Layout,
                                                                    Row,            // CLayout,
                                                                    ck::Tuple<Row>, // D1sLayout,
                                                                    F16,            // ADataType,
                                                                    F16,            // B0DataType,
                                                                    F16,            // D0DataType,
                                                                    F16,            // B1DataType,
                                                                    F32,            // CDataType,
                                                                    ck::Tuple<F16>  // D1sDataType
                                                                    >(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      time_kernel,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      O,
                                                                      BatchCount,
                                                                      StrideA0,
                                                                      StrideB0,
                                                                      StrideB1,
                                                                      StrideC1,
                                                                      BatchStrideA0,
                                                                      BatchStrideB0,
                                                                      BatchStrideB1,
                                                                      BatchStrideC1);
    }
    else
    {
        throw std::runtime_error("wrong! this data_type & layout is not implemented");
    }

    return 0;
}
