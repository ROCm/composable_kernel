// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "batched_transpose_2d.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("H", "16", "input height size.").insert("W", "16", "input width size. ");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Type>
bool run_transpose(ck_tile::ArgParser args)
{
    int dim_in[2], dim_out[2];
    int stride_dim_in[2], stride_dim_out[2];
    int H = args.get_int("H");
    int W = args.get_int("W");

    dim_in[0]         = W;
    dim_in[1]         = H;
    dim_out[0]        = H;
    dim_out[1]        = W;
    stride_dim_in[0]  = 1;
    stride_dim_in[1]  = W;
    stride_dim_out[0] = 1;
    stride_dim_out[1] = H;

    ck_tile::HostTensor<Type> x_host({dim_in[0], dim_in[1]}, {stride_dim_in[0], stride_dim_in[1]});
    ck_tile::HostTensor<Type> y_host({dim_out[0], dim_out[1]},
                                     {stride_dim_out[0], stride_dim_out[1]});

    std::iota(std::begin(x_host), std::begin(x_host) + 256, 1);
    std::iota(std::begin(x_host) + 256, std::end(x_host), 7);

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    batched_transpose_2d_kargs karg = [&]() {
        batched_transpose_2d_kargs a_;
        a_.p_input  = x_dev.GetDeviceBuffer();
        a_.p_output = y_dev.GetDeviceBuffer();
        a_.height   = H;
        a_.width    = W;
        return a_;
    }();

    ck_tile::stream_config sc{nullptr, true, 0, 0, 1, true};

    try
    {
        batched_transpose_2d(karg, sc);
    }
    catch(const std::out_of_range&)
    {
        printf("not supported\n");
        return false;
    }

    fflush(stdout);
    y_dev.FromDevice(y_host.data());
    return true;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;

    result = run_transpose<ck_tile::int8_t>(args);
    if(!result)
        return -1;

    return 0;
}
