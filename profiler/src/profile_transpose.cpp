// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_transpose_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct MatrixLayout
{
    NCDHW, // 0
    NCHWD, // 1
};

enum struct DataType
{
    F32_F32_F32_F32_F32, // 0
    F16_F16_F16_F16_F16, // 1
};

#define OP_NAME "transpose"
#define OP_DESC "Transpose"

int profile_transpose(int argc, char* argv[])
{
    if(argc != 15)
    {
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        // printf("arg3: matrix layout (NCDHW -> NDCHW);\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=no, 1=yes)\n");
        printf("arg8 to 13: N, C, D, H, W\n");
        exit(1);
    }

    const auto data_type = static_cast<DataType>(std::stoi(argv[2]));
    // const auto layout          = static_cast<MatrixLayout>(std::stoi(argv[3]));
    const bool do_verification   = std::stoi(argv[3]);
    const int init_method        = std::stoi(argv[4]);
    const bool do_log            = std::stoi(argv[5]);
    const bool time_kernel       = std::stoi(argv[6]);
    std::vector<index_t> lengths = std::stoi(argv[7]);

    /**const int N = std::stoi(argv[7]);
    const int C = std::stoi(argv[8]);
    const int D = std::stoi(argv[9]);
    const int H = std::stoi(argv[10]);
    const int W = std::stoi(argv[11]);**/

    using F32 = float;
    using F16 = ck::half_t;

    auto profile = [&](auto a_type, auto b_type) {
        using ADataType = decltype(a_type);
        using BDataType = decltype(b_type);

        bool pass = ck::profiler::profile_transpose_impl<ADataType, BDataType>(
            do_verification, init_method, do_log, time_kernel, lengths);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F32_F32_F32_F32_F32)
    {
        return profile(F32{}, F32{});
    }
    else if(data_type == GemmDataType::F16_F16_F16_F16_F16)
    {
        return profile(F16{}, F16{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_transpose);
