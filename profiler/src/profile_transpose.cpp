// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_transpose_impl.hpp"
#include "profiler_operation_registry.hpp"

enum struct DataType
{
    F32_F32_F32_F32_F32, // 0
    F16_F16_F16_F16_F16, // 1
};

#define OP_NAME "transpose"
#define OP_DESC "Transpose"

struct TransposeArgParser
{
    std::unordered_map<std::string, std::vector<int>> long_opts = {{"lengths", {}}};

    bool parse_opt(const int argc, char* argv[], const std::string& key, int i)
    {
        if(std::string("--") + key == argv[i])
        {
            const int pos = i;
            while(++i < argc && argv[i][0] != '-') {}
            int end = i;
            for(int j = pos + 1; j < end; j++)
            {
                long_opts[key].push_back(std::stoi(argv[j]));
            }
            return true;
        }
        return false;
    }

    void operator()(int argc, char* argv[])
    {
        for(auto& kv : long_opts)
        {
            for(int i = 1; i < argc; i++)
            {
                if(parse_opt(argc, argv, kv.first, i))
                    break;
            }
        }
    }
};

static void print_helper_msg()
{
    printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
    printf("arg2: data type (0: fp32; 1: fp16)\n");
    printf("arg3: verification (0: no; 1: yes)\n");
    printf("arg4: initialization (0: no init; 1: integer value; 2: decimal value)\n");
    printf("arg5: print tensor value (0: no; 1: yes)\n");
    printf("arg6: time kernel (0=no, 1=yes)\n");
    printf("arg7: --lengths: N, C, D, H, W\n");
}

int profile_transpose(int argc, char* argv[])
{
    if(argc != 7)
    {
        print_helper_msg();
        exit(1);
    }
    TransposeArgParser arg_parser;

    const auto data_type       = static_cast<DataType>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const bool time_kernel     = std::stoi(argv[6]);
    arg_parser(argc, argv);
    const std::vector<ck::index_t> lengths = arg_parser.long_opts["lengths"];

    using F32 = float;
    using F16 = ck::half_t;

    auto profile = [&](auto a_type, auto b_type) {
        using ADataType              = decltype(a_type);
        using BDataType              = decltype(b_type);
        constexpr ck::index_t NumDim = 5;

        bool pass = ck::profiler::profile_transpose_impl<ADataType, BDataType, NumDim>(
            do_verification, init_method, do_log, time_kernel, lengths);

        return pass ? 0 : 1;
    };

    if(data_type == DataType::F32_F32_F32_F32_F32)
    {
        return profile(F32{}, F32{});
    }
    else if(data_type == DataType::F16_F16_F16_F16_F16)
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
