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
        F16_F16_F16_F16_F16, // 0
    };
    GemmDataType data_type  = GemmDataType::F16_F16_F16_F16_F16;
    GemmMatrixLayout layout = GemmMatrixLayout::MK_NK_MN_ON_MO;
    bool do_verification    = true;
    int init_method         = 1;
    bool do_log             = 0;
    bool time_kernel        = false;

    // GEMM shape
    ck::index_t M             = 1024;
    ck::index_t N             = 1024;
    ck::index_t K             = 64;
    ck::index_t O             = 128;
    ck::index_t BatchCount    = 4;
    ck::index_t StrideA0      = -1;
    ck::index_t StrideB0      = -1;
    ck::index_t StrideB1      = -1;
    ck::index_t StrideC1      = -1;
    ck::index_t BatchStrideA0 = -1;
    ck::index_t BatchStrideB0 = -1;
    ck::index_t BatchStrideB1 = -1;
    ck::index_t BatchStrideC1 = -1;

    if(argc == 8)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);
    }
    else if(argc == 13)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);

        M          = std::stoi(argv[8]);
        N          = std::stoi(argv[9]);
        K          = std::stoi(argv[10]);
        O          = std::stoi(argv[11]);
        BatchCount = std::stoi(argv[12]);
    }
    else if(argc == 21)
    {
        data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
        layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
        do_verification = std::stoi(argv[4]);
        init_method     = std::stoi(argv[5]);
        do_log          = std::stoi(argv[6]);
        time_kernel     = std::stoi(argv[7]);

        M          = std::stoi(argv[8]);
        N          = std::stoi(argv[9]);
        K          = std::stoi(argv[10]);
        O          = std::stoi(argv[11]);
        BatchCount = std::stoi(argv[12]);

        StrideA0 = std::stoi(argv[13]);
        StrideB0 = std::stoi(argv[14]);
        StrideB1 = std::stoi(argv[15]);
        StrideC1 = std::stoi(argv[16]);

        BatchStrideA0 = std::stoi(argv[17]);
        BatchStrideB0 = std::stoi(argv[18]);
        BatchStrideB1 = std::stoi(argv[19]);
        BatchStrideC1 = std::stoi(argv[20]);
    }
    else
    {
        printf("arg1: tensor operation (batched_gemm_bias_gelu_gemm_bias: "
               "Batched+GEMM+bias+gelu+Gemm+bias)\n");
        printf("arg2: data type (0: fp16)\n");
        printf(
            "arg3: matrix layout (0: gelu(A[m, k] * B[n, k] + bias) * O[m, o] + bias = C[m, n];\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf(
            "arg8 to 20: M, N, K, O, Batch, StrideA0, StrideB0, StrideB1, StrideC1, BatchStrideA0, "
            "BatchStrideB0, BatchStrideB1, BatchStrideC1 \n");
        exit(1);
    }

    if(data_type == GemmDataType::F16_F16_F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN_ON_MO)
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
                                                                    F16,            // C1DataType,
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
