#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "profile_grouped_gemm_impl.hpp"

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
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

std::vector<int> stringToArray(char *input)
{
    std::vector<int> out;

    std::istringstream in(input);

    std::string item;

    while (std::getline(in, item, ',')) {
        out.push_back(std::stoi(item));
    }

    return out;
}

int profile_grouped_gemm(int argc, char* argv[])
{
    if(!(argc == 14))
    {
        printf("arg1: tensor operation (grouped_gemm: Grouped GEMM)\n");
        printf("arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8)\n");
        printf("arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg7: run kernel # of times (>1)\n");
        printf("arg8 to 13: Ms, Ns, Ks, StrideAs, StrideBs, StrideCs\n");
        exit(1);
    }

    const int data_type        = static_cast<GemmDataType>(std::stoi(argv[2]));
    const int layout           = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const int nrepeat          = std::stoi(argv[7]);

    const auto Ms = stringToArray(argv[8]);
    const auto Ns = stringToArray(argv[9]);
    const auto Ks = stringToArray(argv[10]);
    

    const auto StrideAs = stringToArray(argv[11]);
    const auto StrideBs = stringToArray(argv[12]);
    const auto StrideCs = stringToArray(argv[13]);

    if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_impl<ck::half_t,
                                        ck::half_t,
                                        ck::half_t,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs);
    }
#if 0
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm_impl<ck::half_t,
                                        ck::half_t,
                                        ck::half_t,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? K : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm_impl<ck::half_t,
                                        ck::half_t,
                                        ck::half_t,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? N : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm_impl<ck::half_t,
                                        ck::half_t,
                                        ck::half_t,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_gemm_impl<float,
                                        float,
                                        float,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? K : StrideA,
            (StrideB < 0) ? N : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm_impl<float,
                                        float,
                                        float,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? K : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_KN_MN)
    {
        ck::profiler::profile_gemm_impl<float,
                                        float,
                                        float,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? N : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::F32_F32_F32 && layout == GemmMatrixLayout::KM_NK_MN)
    {
        ck::profiler::profile_gemm_impl<float,
                                        float,
                                        float,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::INT8_INT8_INT8 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm_impl<int8_t,
                                        int8_t,
                                        int8_t,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else if(data_type == GemmDataType::BF16_BF16_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_gemm_impl<ck::bhalf_t,
                                        ck::bhalf_t,
                                        ck::bhalf_t,
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::ColumnMajor,
                                        ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            nrepeat,
            M,
            N,
            K,
            (StrideA < 0) ? M : StrideA,
            (StrideB < 0) ? K : StrideB,
            (StrideC < 0) ? N : StrideC,
            KBatch);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }
#endif

    return 1;
}
