// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_gemm_multi_abd_fixed_nk_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN  // 2
};

enum struct GemmDataType
{
    BF16_I8_BF16 // 0
};

#define OP_NAME "grouped_gemm_multi_abd_fixed_nk"
#define OP_DESC "Grouped GEMM Multi ABD Fixed NK"

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

int profile_grouped_gemm_multi_abd_fixed_nk(int argc, char* argv[])
{
    if(argc < 14)
    {
        std::cout
            << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
            << "arg2: data type (0: bf16@int8; 1: fp16; 2: fp16@fp8; 3: fp16@int8)\n"
            << "arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n"
            << "                     1: A[m, k] * B[n, k] = C[m, n];\n"
            << "                     2: A[k, m] * B[k, n] = C[m, n];)\n"
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

    // const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    // const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const auto Ms = argToIntArray(argv[8]);
    const auto Ns = argToIntArray(argv[9]);
    const auto Ks = argToIntArray(argv[10]);

    const auto StrideAs = argToIntArray(argv[11]);
    const auto StrideBs = argToIntArray(argv[12]);
    const auto StrideDs = argToIntArray(argv[13]);
    const auto StrideE = StrideDs.at(0);
    const int kbatch    = argc >= 15 ? std::stoi(argv[14]) : 1;

    int n_warmup = 1;
    int n_iter   = 10;
    if(argc == 17)
    {
        n_warmup = std::stoi(argv[15]);
        n_iter   = std::stoi(argv[16]);
    }

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    ck::profiler::profile_grouped_gemm_multi_abd_fixed_nk_impl<ck::Tuple<ck::bhalf_t>,
                                                                ck::Tuple<int8_t, ck::bhalf_t>,
                                                                ck::Tuple<>,
                                                                float,
                                                                float,
                                                                ck::Tuple<Row>,
                                                                ck::Tuple<Col, Col>,
                                                                ck::Tuple<>,
                                                                Row>(do_verification,
                                                                    init_method,
                                                                    do_log,
                                                                    time_kernel,
                                                                    Ms,
                                                                    Ns,
                                                                    Ks,
                                                                    StrideAs,
                                                                    StrideBs,
                                                                    StrideDs,
                                                                    StrideE,
                                                                    kbatch,
                                                                    n_warmup,
                                                                    n_iter);

// #if defined(CK_ENABLE_INT8)
// #if defined(CK_ENABLE_BF16)
//     if(data_type == GemmDataType::BF16_I8_BF16 && layout == GemmMatrixLayout::KM_KN_MN)
//     {
//         ck::profiler::profile_grouped_gemm_multi_abd_fixed_nk_impl<ck::Tuple<ck::bhalf_t>,
//                                                                    ck::Tuple<int8_t, ck::bhalf_t>,
//                                                                    ck::Tuple<>,
//                                                                    float,
//                                                                    float,
//                                                                    ck::Tuple<Row>,
//                                                                    ck::Tuple<Col, Col>,
//                                                                    ck::Tuple<>,
//                                                                    Row>(do_verification,
//                                                                         init_method,
//                                                                         do_log,
//                                                                         time_kernel,
//                                                                         Ms,
//                                                                         Ns,
//                                                                         Ks,
//                                                                         StrideAs,
//                                                                         StrideBs,
//                                                                         StrideDs,
//                                                                         StrideE,
//                                                                         kbatch,
//                                                                         n_warmup,
//                                                                         n_iter);
//     }
//     else if(data_type == GemmDataType::BF16_I8_BF16 && layout == GemmMatrixLayout::MK_KN_MN)
//     {
//         ck::profiler::profile_grouped_gemm_multi_abd_fixed_nk_impl<ck::Tuple<ck::bhalf_t>,
//                                                                    ck::Tuple<int8_t>,
//                                                                    ck::Tuple<ck::bhalf_t>,
//                                                                    float,
//                                                                    float,
//                                                                    ck::Tuple<Row>,
//                                                                    ck::Tuple<Row>,
//                                                                    ck::Tuple<Row>,
//                                                                    Row>(do_verification,
//                                                                         init_method,
//                                                                         do_log,
//                                                                         time_kernel,
//                                                                         Ms,
//                                                                         Ns,
//                                                                         Ks,
//                                                                         StrideAs,
//                                                                         StrideBs,
//                                                                         StrideDs,
//                                                                         StrideE,
//                                                                         kbatch,
//                                                                         n_warmup,
//                                                                         n_iter);
//     }
//     else if(data_type == GemmDataType::BF16_I8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
//     {
//         ck::profiler::profile_grouped_gemm_multi_abd_fixed_nk_impl<ck::Tuple<ck::bhalf_t>,
//                                                                    ck::Tuple<int8_t>,
//                                                                    ck::Tuple<ck::bhalf_t>,
//                                                                    float,
//                                                                    float,
//                                                                    ck::Tuple<Row>,
//                                                                    ck::Tuple<Col>,
//                                                                    ck::Tuple<Row>,
//                                                                    Row>(do_verification,
//                                                                         init_method,
//                                                                         do_log,
//                                                                         time_kernel,
//                                                                         Ms,
//                                                                         Ns,
//                                                                         Ks,
//                                                                         StrideAs,
//                                                                         StrideBs,
//                                                                         StrideDs,
//                                                                         StrideE,
//                                                                         kbatch,
//                                                                         n_warmup,
//                                                                         n_iter);
//     }
// #endif // CK_ENABLE_BF16
// #endif // CK_ENABLE_INT8
//     else
//     {
//         throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
//     }
    return 0;
}

} // anonymous namespace

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_gemm_multi_abd_fixed_nk);
