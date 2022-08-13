// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <unordered_map>

#include "profiler/include/profile_normalization_impl.hpp"

using ck::index_t;
using ck::profiler::NormDataType;
using ck::profiler::NormType;

struct ArgParser
{
    std::unordered_map<std::string, NormType> norm_dict = {{"batchnorm", NormType::BATCHNORM},
                                                           {"softmax", NormType::SOFTMAX}};

    std::unordered_map<std::string, std::vector<int>> long_opts = {
        {"length", {}}, {"stride", {}}, {"reduce", {}}, {"alpha", {}}, {"beta", {}}};

    bool parse_opt(int argc, char* argv[], const std::string& key, int i)
    {
        if(std::string("--") + key == argv[i])
        {
            int pos = i;
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

void print_help()
{
    std::cout << "arg1: tensor operation (layernorm/batchnorm/softmax)\n"
              << "arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8)\n"
              << "arg3: verification (0: no; 1: yes)\n"
              << "arg4: initialization (0: no init; 1: integer value; 2: decimal value)\n"
              << "arg5: print tensor value (0: no; 1: yes)\n"
              << "arg6: time kernel (0=n0, 1=yes)\n"
              << "--length: tensor extents (e.g, --length 8 4 256) \n"
              << "--stride: tensor strides (e.g, --stride 1024 256 1)\n"
              << "--reduce: to-reduce dimensions (e.g, --reduce 2)\n"
              << "--alpha: alpha scaling value\n"
              << "--beta: beta scaling value\n"
              << std::endl;
}

int profile_normalization(int argc, char* argv[])
{
    if(argc <= 2)
    {
        print_help();
        return 0;
    }

    ArgParser arg_parser;

    // short unnamed options
    const NormType norm_type     = arg_parser.norm_dict[argv[1]];
    const NormDataType data_type = static_cast<NormDataType>(std::stoi(argv[2]));
    const bool do_verification   = std::stoi(argv[3]);
    const int init_method        = std::stoi(argv[4]);
    const bool do_log            = std::stoi(argv[5]);
    const bool time_kernel       = std::stoi(argv[6]);

    // parse the long options
    arg_parser(argc, argv);
    const std::vector<index_t> length = arg_parser.long_opts["length"];
    const std::vector<index_t> stride = arg_parser.long_opts["stride"];
    const std::vector<index_t> reduce = arg_parser.long_opts["reduce"];
    const index_t alpha =
        arg_parser.long_opts["alpha"].empty() ? 1 : arg_parser.long_opts["alpha"][0];
    const index_t beta = arg_parser.long_opts["beta"].empty() ? 0 : arg_parser.long_opts["beta"][0];

    if(data_type == NormDataType::F16_F16)
    {
        ck::profiler::profile_normalization_impl<ck::half_t, float, ck::half_t>(do_verification,
                                                                                init_method,
                                                                                do_log,
                                                                                time_kernel,
                                                                                length,
                                                                                stride,
                                                                                reduce,
                                                                                float(alpha),
                                                                                float(beta),
                                                                                norm_type);
    }
    else if(data_type == NormDataType::F32_F32)
    {
        ck::profiler::profile_normalization_impl<float, float, float>(do_verification,
                                                                      init_method,
                                                                      do_log,
                                                                      time_kernel,
                                                                      length,
                                                                      stride,
                                                                      reduce,
                                                                      float(alpha),
                                                                      float(beta),
                                                                      norm_type);
    }
    else
    {
        throw std::runtime_error("not implemented yet");
    }

    return 0;
}

// hijack main() for quick debugging
// int main(int argc, char* argv[])
// {
//     profile_normalization(argc, argv);
//     return 0;
// }
