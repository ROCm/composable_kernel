#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "profile_gemm_impl.hpp"

// return true if test pass
bool profile_gemm(int argc, char* argv[])
{
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

    if(argc != 14)
    {
        printf("arg1: tensor operation (gemm: GEMM)\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg7: run kernel # of times (>1)\n");
        printf("arg8 to 13: M, N, K, StrideA, StrideB, StrideC\n");
        return false;
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
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

    auto profile =
        [&](auto a_type, auto b_type, auto c_type, auto a_layout, auto b_layout, auto c_layout) {
            using ADataType = decltype(a_type);
            using BDataType = decltype(b_type);
            using CDataType = decltype(c_type);
            using ALayout   = decltype(a_layout);
            using BLayout   = decltype(b_layout);
            using CLayout   = decltype(c_layout);

            return ck::profiler::
                profile_gemm_impl<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout>(
                    do_verification,
                    init_method,
                    do_log,
                    nrepeat,
                    M,
                    N,
                    K,
                    (StrideA < 0) ? K : StrideA,
                    (StrideB < 0) ? N : StrideB,
                    (StrideC < 0) ? N : StrideC);
        };

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(ck::half_t{}, ck::half_t{}, ck::half_t{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(ck::half_t{}, ck::half_t{}, ck::half_t{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(ck::half_t{}, ck::half_t{}, ck::half_t{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(ck::half_t{}, ck::half_t{}, ck::half_t{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(float{}, float{}, float{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(float{}, float{}, float{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(float{}, float{}, float{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(float{}, float{}, float{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(int8_t{}, int8_t{}, int8_t{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(int8_t{}, int8_t{}, int8_t{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(int8_t{}, int8_t{}, int8_t{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(int8_t{}, int8_t{}, int8_t{}, Col{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        return profile(ck::bhalf_t{}, ck::bhalf_t{}, ck::bhalf_t{}, Row{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(ck::bhalf_t{}, ck::bhalf_t{}, ck::bhalf_t{}, Row{}, Col{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        return profile(ck::bhalf_t{}, ck::bhalf_t{}, ck::bhalf_t{}, Col{}, Row{}, Row{});
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        return profile(ck::bhalf_t{}, ck::bhalf_t{}, ck::bhalf_t{}, Col{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return true;
    }
}
