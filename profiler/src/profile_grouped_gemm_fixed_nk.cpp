// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_gemm_fixed_nk_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
};

enum struct GemmDataType
{
    BF16_I8_BF16, // 0
    F16_F16_F16,  // 1
    F16_F8_F16,   // 2
    F16_I8_F16,   // 3

};

#define OP_NAME "grouped_gemm_fixed_nk"
#define OP_DESC "Grouped GEMM Fixed NK"

namespace {

std::vector<int> argToIntArray(char* input)
{
    std::vector<int> out;

    std::istringstream in(input);

    std::string item;

    while(std::getline(in, item, ','))
    {
        out.push_back(std::stoi(item));
    }

    return out;
}

int profile_grouped_gemm_fixed_nk(int argc, char* argv[])
{
    if(argc < 14)
    {
        std::cout
            << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
            << "arg2: data type (0: bf16@int8; 1: fp16; 2: fp16@fp8; 3: fp16@int8)\n"
            << "arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n"
            << "                     1: A[m, k] * B[n, k] = C[m, n];\n"
            << "arg4: verification (0: no; 1: yes)\n"
            << "arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n"
            << "arg6: print tensor value (0: no; 1: yes)\n"
            << "arg7: time kernel (0=n0, 1=yes)\n"
            << "arg8 to 13: Ms, Ns, Ks, StrideAs, StrideBs, StrideCs (e.g., 256,256 128,128 64,64 "
               "64,64 64,64 128,128)\n"
            << "arg15: kbatch value (default 1)\n"
            << "optional:\n"
            << "arg16: number of warm-up cycles (default 1)\n"
            << "arg17: number of iterations (default 10)\n"
            << std::endl;

        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const auto Ms = argToIntArray(argv[8]);
    const auto Ns = argToIntArray(argv[9]);
    const auto Ks = argToIntArray(argv[10]);

    const auto StrideAs = argToIntArray(argv[11]);
    const auto StrideBs = argToIntArray(argv[12]);
    const auto StrideCs = argToIntArray(argv[13]);
    const int kbatch    = argc == 15 ? std::stoi(argv[14]) : 1;

    using F32  = float;
    using F16  = ck::half_t;
    using F8   = ck::f8_t;
    using BF16 = ck::bhalf_t;
    using I8   = int8_t;

    int n_warmup = 1;
    int n_iter   = 10;
    if(argc == 17)
    {
        n_warmup = std::stoi(argv[16]);
        n_iter   = std::stoi(argv[17]);
    }

#if defined(CK_ENABLE_BF16) && defined(CK_ENABLE_INT8)
    if(data_type == GemmDataType::BF16_I8_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<BF16,
                                                         I8,
                                                         BF16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
    else if(data_type == GemmDataType::BF16_I8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<BF16,
                                                         I8,
                                                         BF16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::ColumnMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
#endif
#if defined(CK_ENABLE_FP16)
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         F16,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
    else if(data_type == GemmDataType::F16_F16_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         F16,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::ColumnMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
#endif
#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8)
    else if(data_type == GemmDataType::F16_F8_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         F8,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
    else if(data_type == GemmDataType::F16_F8_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         F8,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::ColumnMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
#endif
#if defined(CK_ENABLE_FP16) && defined(CK_ENABLE_INT8)
    else if(data_type == GemmDataType::F16_I8_F16 && layout == GemmMatrixLayout::MK_KN_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         I8,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            kbatch,
            n_warmup,
            n_iter);
    }
    else if(data_type == GemmDataType::F16_I8_F16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        ck::profiler::profile_grouped_gemm_fixed_nk_impl<F16,
                                                         I8,
                                                         F16,
                                                         F32,
                                                         ck::tensor_layout::gemm::RowMajor,
                                                         ck::tensor_layout::gemm::ColumnMajor,
                                                         ck::tensor_layout::gemm::RowMajor>(
            do_verification,
            init_method,
            do_log,
            time_kernel,
            Ms,
            Ns,
            Ks,
            StrideAs,
            StrideBs,
            StrideCs,
            1,
            n_warmup,
            n_iter);
    }
#endif
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }
    return 0;
}

} // anonymous namespace

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_gemm_fixed_nk);
