// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdint>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/include/profile_batched_gemm_impl.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
};

enum struct GemmDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

int profile_batched_gemm(int argc, char* argv[])
{
    if(argc != 15)
    {
        printf("arg1: tensor operation (batched_gemm: Batched GEMM)\n");
        printf("arg2: data type (0: fp32; 1: fp16, 2: bf16, 3: int8)\n");
        printf("arg3: matrix layout (0: A[g, m, k] * B[g, k, n] = C[g, m, n];\n");
        printf("                     1: A[g, m, k] * B[g, n, k] = C[g, m, n];\n");
        printf("                     2: A[g, k, m] * B[g, k, n] = C[g, m, n];\n");
        printf("                     3: A[g, k, m] * B[g, n, k] = C[g, m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf("arg8 to 14: M, N, K, StrideA, StrideB, StrideC, BatchCount\n");
        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideC = std::stoi(argv[13]);

    const int BatchCount = std::stoi(argv[14]);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using INT8 = int8_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto c_type,
                       auto a_layout,
                       auto b_layout,
                       auto c_layout) {
        using ADataType = decltype(a_type);
        using BDataType = decltype(b_type);
        using CDataType = decltype(c_type);

        using ALayout = decltype(a_layout);
        using BLayout = decltype(b_layout);
        using CLayout = decltype(c_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideC = ck::is_same_v<CLayout, Row> ? N : M;

        bool pass = ck::profiler::
            profile_batched_gemm_impl<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout>(
                do_verification,
                init_method,
                do_log,
                time_kernel,
                M,
                N,
                K,
                (StrideA < 0) ? DefaultStrideA : StrideA,
                (StrideB < 0) ? DefaultStrideB : StrideB,
                (StrideC < 0) ? DefaultStrideC : StrideC,
                BatchCount);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F32{}, F32{}, F32{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F32{}, F32{}, F32{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(F32{}, F32{}, F32{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(F32{}, F32{}, F32{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(F16{}, F16{}, F16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F16{}, F16{}, F16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(F16{}, F16{}, F16{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(F16{}, F16{}, F16{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(BF16{}, BF16{}, BF16{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(BF16{}, BF16{}, BF16{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(BF16{}, BF16{}, BF16{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(BF16{}, BF16{}, BF16{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(INT8{}, INT8{}, INT8{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(INT8{}, INT8{}, INT8{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(INT8{}, INT8{}, INT8{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(INT8{}, INT8{}, INT8{}, Col{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}
