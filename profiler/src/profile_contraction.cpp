// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "profiler/profile_contraction_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct ContractionMatrixLayout
{
    MK_KN_MN_MN, // 0
    MK_NK_MN_MN, // 1
    KM_KN_MN_MN, // 2
    KM_NK_MN_MN, // 3
};

enum struct ContractionDataType
{
    F32_F32_F32_F32, // 0
    F64_F64_F64_F64, // 1
};

#define OP_NAME "contraction"
#define OP_DESC "CONTRACTION"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: fp32; 1: f64)\n"
              << "arg3: matrix layout (0: A[m0, m1, k0, k1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = C[m0, m1, n0, n1];\n"
              << "                     1: A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = C[m0, m1, n0, n1];\n"
              << "                     2: A[k0, k1, m0, m1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = C[m0, m1, n0, n1];\n"
              << "                     3: A[k0, k1, m0, m1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = C[m0, m1, n0, n1])\n"
              << "arg4: verification (0: no; 1: yes)\n"
              << "arg5: initialization (0: no init; 1: integer value; 2: decimal "
              << "value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << "arg8 and arg9(optional): alpha and beta for bilinear (pass only "
              << "alpha for scale)\n"
              << "arg9/10 to 14/15: M0, M1, N0, N1, K0, K1\n"
              << "arg15/16 to 30/31: Strides for A, B, C and D (skip for default)\n"
              << std::endl;
}

void collect_index_params(char* argv[],
                          std::vector<ck::index_t>& params,
                          const int from,
                          const int num)
{
    for(int p = from; p < from + num; p++)
        params.push_back(std::stoi(argv[p]));
}

// Defualt strides for row-major: {Dim1 * Dim2 * Dim3, Dim2 * Dim3, Dim3, 1}
// Defualt strides for column-major: {Dim1, 1, Dim0 * Dim1 * Dim3, Dim0 * Dim1}
void assign_default_strides(Row, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    strides = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

void assign_default_strides(Col, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    strides = {dims[1], 1, dims[0] * dims[1] * dims[3], dims[0] * dims[1]};
}

int profile_contraction(int argc, char* argv[])
{
    const bool all_parameters_bilinear        = argc == 32;
    const bool all_parameters_scale           = argc == 31;
    const bool parameters_wo_strides_bilinear = argc == 16;
    const bool parameters_wo_strides_scale    = argc == 15;
    const bool default_strides = parameters_wo_strides_bilinear || parameters_wo_strides_scale;
    const bool with_bilinear   = all_parameters_bilinear || parameters_wo_strides_bilinear;

    if(!(all_parameters_bilinear || all_parameters_scale || parameters_wo_strides_bilinear ||
         parameters_wo_strides_scale))
    {
        print_helper_msg();
        exit(1);
    }

    const auto data_type       = static_cast<ContractionDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ContractionMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const float alpha          = std::stof(argv[8]);
    const float beta           = with_bilinear ? std::stof(argv[9]) : 0;

    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
    const int dims_arg_num = with_bilinear ? 10 : 9;
    collect_index_params(argv, M, dims_arg_num, 2);
    collect_index_params(argv, N, dims_arg_num + 2, 2);
    collect_index_params(argv, K, dims_arg_num + 4, 2);

    std::vector<ck::index_t> StridesA;
    std::vector<ck::index_t> StridesB;
    std::vector<ck::index_t> StridesC;
    std::vector<ck::index_t> StridesD;
    if(!default_strides)
    {
        collect_index_params(argv, StridesA, dims_arg_num + 6, 4);
        collect_index_params(argv, StridesB, dims_arg_num + 10, 4);
        collect_index_params(argv, StridesC, dims_arg_num + 14, 4);
        collect_index_params(argv, StridesD, dims_arg_num + 18, 4);
    }

    using F32 = float;
    using F64 = double;

    auto profile = [&](auto a_layout, auto b_layout, auto cd_layout, auto type) {
        using ALayout  = decltype(a_layout);
        using BLayout  = decltype(b_layout);
        using CDLayout = decltype(cd_layout);

        using DataType = decltype(type);

        if(default_strides)
        {
            assign_default_strides(a_layout, StridesA, {M[0], M[1], K[0], K[1]});
            assign_default_strides(b_layout, StridesB, {K[0], K[1], N[0], N[1]});
            assign_default_strides(cd_layout, StridesC, {M[0], M[1], N[0], N[1]});
            assign_default_strides(cd_layout, StridesD, {M[0], M[1], N[0], N[1]});
        }
        bool pass;
        if(with_bilinear)
        {
            pass = ck::profiler::profile_contraction_impl<ALayout,
                                                          BLayout,
                                                          CDLayout,
                                                          DataType,
                                                          ck::Tuple<DataType>,
                                                          Bilinear>(do_verification,
                                                                    init_method,
                                                                    do_log,
                                                                    time_kernel,
                                                                    Bilinear{alpha, beta},
                                                                    M,
                                                                    N,
                                                                    K,
                                                                    StridesA,
                                                                    StridesB,
                                                                    StridesC,
                                                                    StridesD);
        }
        else
        {
            pass = ck::profiler::
                profile_contraction_impl<ALayout, BLayout, CDLayout, DataType, ck::Tuple<>, Scale>(
                    do_verification,
                    init_method,
                    do_log,
                    time_kernel,
                    Scale{alpha},
                    M,
                    N,
                    K,
                    StridesA,
                    StridesB,
                    StridesC,
                    StridesD);
        }

        return pass;
    };

    if(data_type == ContractionDataType::F32_F32_F32_F32 &&
       layout == ContractionMatrixLayout::MK_KN_MN_MN)
    {
        return profile(Row{}, Row{}, Row{}, F32{});
    }
    else if(data_type == ContractionDataType::F32_F32_F32_F32 &&
            layout == ContractionMatrixLayout::MK_NK_MN_MN)
    {
        return profile(Row{}, Col{}, Row{}, F32{});
    }
    else if(data_type == ContractionDataType::F32_F32_F32_F32 &&
            layout == ContractionMatrixLayout::KM_KN_MN_MN)
    {
        return profile(Col{}, Row{}, Row{}, F32{});
    }
    else if(data_type == ContractionDataType::F32_F32_F32_F32 &&
            layout == ContractionMatrixLayout::KM_NK_MN_MN)
    {
        return profile(Col{}, Col{}, Row{}, F32{});
    }
    else if(data_type == ContractionDataType::F64_F64_F64_F64 &&
            layout == ContractionMatrixLayout::MK_KN_MN_MN)
    {
        return profile(Row{}, Row{}, Row{}, F64{});
    }
    else if(data_type == ContractionDataType::F64_F64_F64_F64 &&
            layout == ContractionMatrixLayout::MK_NK_MN_MN)
    {
        return profile(Row{}, Col{}, Row{}, F64{});
    }
    else if(data_type == ContractionDataType::F64_F64_F64_F64 &&
            layout == ContractionMatrixLayout::KM_KN_MN_MN)
    {
        return profile(Col{}, Row{}, Row{}, F64{});
    }
    else if(data_type == ContractionDataType::F64_F64_F64_F64 &&
            layout == ContractionMatrixLayout::KM_NK_MN_MN)
    {
        return profile(Col{}, Col{}, Row{}, F64{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_contraction);
