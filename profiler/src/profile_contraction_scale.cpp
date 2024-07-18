// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "profiler/profile_contraction_impl.hpp"
#include "profiler/profile_contraction_utils.hpp"
#include "profiler_operation_registry.hpp"

#define OP_NAME "contraction_scale"
#define OP_DESC "CONTRACTION+Scale"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: fp32; 1: f64)\n"
              << "arg3: matrix layout (0: A[m0, m1, k0, k1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     1: A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     2: A[k0, k1, m0, m1] * B[k0, k1, n0, n1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1];\n"
              << "                     3: A[k0, k1, m0, m1] * B[n0, n1, k0, k1] + "
                 "D[m0, m1, n0, n1] = E[m0, m1, n0, n1])\n"
              << "arg4: verification (0: no; 1: yes)\n"
              << "arg5: initialization (0: no init; 1: integer value; 2: decimal "
              << "value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << "arg8: alpha\n"
              << "arg9 to 14: M0, M1, N0, N1, K0, K1\n"
              << "arg15 to 30: Strides for A, B, D and E (skip for default)\n"
              << std::endl;
}

int profile_contraction_scale(int argc, char* argv[])
{
    const bool default_strides = argc == 15;

    if(argc != 31 && argc != 15)
    {
        print_helper_msg();
        exit(1);
    }

    const auto data_type          = static_cast<ContractionDataType>(std::stoi(argv[2]));
    const auto layout             = static_cast<ContractionMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification    = std::stoi(argv[4]);
    const ck::index_t init_method = std::stoi(argv[5]);
    const bool do_log             = std::stoi(argv[6]);
    const bool time_kernel        = std::stoi(argv[7]);
    const float alpha             = std::stof(argv[8]);

    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
    const ck::index_t dims_arg_num = 9;
    collect_index_params(argv, M, dims_arg_num, 2);
    collect_index_params(argv, N, dims_arg_num + 2, 2);
    collect_index_params(argv, K, dims_arg_num + 4, 2);

    std::vector<ck::index_t> StridesA;
    std::vector<ck::index_t> StridesB;
    std::vector<ck::index_t> StridesE;
    std::vector<ck::index_t> StridesD;
    if(!default_strides)
    {
        collect_index_params(argv, StridesA, dims_arg_num + 6, 4);
        collect_index_params(argv, StridesB, dims_arg_num + 10, 4);
        collect_index_params(argv, StridesE, dims_arg_num + 14, 4);
        collect_index_params(argv, StridesD, dims_arg_num + 18, 4);
    }

    using F32 = float;
    using F64 = double;

    auto profile = [&](auto a_layout, auto b_layout, auto cde_layout, auto type) {
        using ALayout   = decltype(a_layout);
        using BLayout   = decltype(b_layout);
        using CDELayout = decltype(cde_layout);

        using DataType = decltype(type);

        if(default_strides)
        {
            assign_default_strides(a_layout, StridesA, {M[0], M[1], K[0], K[1]});
            assign_default_strides(b_layout, StridesB, {K[0], K[1], N[0], N[1]});
            assign_default_strides(cde_layout, StridesE, {M[0], M[1], N[0], N[1]});
            assign_default_strides(cde_layout, StridesD, {M[0], M[1], N[0], N[1]});
        }

        bool pass = ck::profiler::
            profile_contraction_impl<ALayout, BLayout, CDELayout, DataType, ck::Tuple<>, Scale>(
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
                StridesE,
                StridesD);

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

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_contraction_scale);
