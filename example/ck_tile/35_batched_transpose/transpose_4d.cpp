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

#include "batched_transpose.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "whether do CPU validation or not")
        .insert("pr", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("N", "2", "input batch size. ")
        .insert("C", "16", "input channel size.")
        .insert("H", "1", "input height size.")
        .insert("W", "16", "input width size. ")
        .insert("layout_in", "NCHW", "input tensor data layout - NCHW by default")
        .insert("layout_out", "NHWC", "output tensor data layout - NHWC by default ");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Type>
bool run_transpose(ck_tile::ArgParser args)
{
    int validate           = args.get_int("v");
    std::string prec       = args.get_str("pr");
    int N                  = args.get_int("N");
    int C                  = args.get_int("C");
    int H                  = args.get_int("H");
    int W                  = args.get_int("W");
    std::string layout_in  = args.get_str("layout_in");
    std::string layout_out = args.get_str("layout_out");
    int seed               = args.get_int("seed");

    int dim_in[4], dim_out[4];
    int stride_dim_in[4], stride_dim_out[4];
    bool nchw2nhwc = layout_in == "NCHW" && layout_out == "NHWC";
    bool nhwc2nchw = layout_in == "NHWC" && layout_out == "NCHW";
    assert(nchw2nhwc != nhwc2nchw);
    (void)nhwc2nchw;

    dim_in[0]         = N;
    dim_in[1]         = nchw2nhwc ? C : H;
    dim_in[2]         = nchw2nhwc ? H : W;
    dim_in[3]         = nchw2nhwc ? W : C;
    dim_out[0]        = N;
    dim_out[1]        = nchw2nhwc ? H : C;
    dim_out[2]        = nchw2nhwc ? W : H;
    dim_out[3]        = nchw2nhwc ? C : W;
    stride_dim_in[0]  = C * H * W;
    stride_dim_in[1]  = nchw2nhwc ? H * W : C * W;
    stride_dim_in[2]  = nchw2nhwc ? W : C;
    stride_dim_in[3]  = 1;
    stride_dim_out[0] = C * H * W;
    stride_dim_out[1] = nchw2nhwc ? C * W : H * W;
    stride_dim_out[2] = nchw2nhwc ? C : W;
    stride_dim_out[3] = 1;

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }

    ck_tile::HostTensor<Type> x_host(
        {dim_in[0], dim_in[1], dim_in[2], dim_in[3]},
        {stride_dim_in[0], stride_dim_in[1], stride_dim_in[2], stride_dim_in[3]});
    ck_tile::HostTensor<Type> y_host(
        {dim_out[0], dim_out[1], dim_out[2], dim_out[3]},
        {stride_dim_out[0], stride_dim_out[1], stride_dim_out[2], stride_dim_out[3]});

    ck_tile::FillUniformDistribution<Type>{-.5f, .5f}(x_host);

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    auto trait = batched_transpose_trait{prec};

    uint32_t height = nchw2nhwc ? C : H * W;
    uint32_t width  = nchw2nhwc ? H * W : C;

    batched_transpose_kargs karg = [&]() {
        batched_transpose_kargs a_;
        a_.p_input  = x_dev.GetDeviceBuffer();
        a_.p_output = y_dev.GetDeviceBuffer();
        a_.batch    = N;
        a_.height   = height;
        a_.width    = width;
        return a_;
    }();

    ck_tile::stream_config sc{nullptr, true};

    std::cout << "Running transpose with"
              << " N=" << N << ", C=" << C << ", H=" << H << ", W=" << W
              << ", layout_in=" << layout_in << ", layout_out=" << layout_out;

    float ms = 0;
    try
    {
        ms = batched_transpose(trait, karg, sc);
    }
    catch(const std::out_of_range&)
    {
        std::cerr << "Error: Unsupported precision type '" << args.get_str("pr") << "'\n";
        return false;
    }

    std::size_t num_operations = N * height * (width - 1);
    std::size_t num_bytes      = N * C * H * W * sizeof(Type);

    float ave_time   = ms * 1E-3;
    float gb_per_sec = num_bytes / ms * 1.E-6;
    float tflops     = static_cast<float>(num_operations) / ms * 1.E-6;

    std::cout << "> " << ms << " ms(" << ave_time << " ave_time), " << tflops << " TFlops, "
              << gb_per_sec << " GB/s." << std::endl;

    y_dev.FromDevice(y_host.data());

    bool rtn = true;
    if(validate)
    {
        ck_tile::HostTensor<Type> y_ref(
            {dim_out[0], dim_out[1], dim_out[2], dim_out[3]},
            {stride_dim_out[0], stride_dim_out[1], stride_dim_out[2], stride_dim_out[3]});

        ck_tile::reference_transpose_4d<Type>(x_host, y_ref, layout_in, layout_out);

        rtn &= ck_tile::check_ref<Type>(y_host, y_ref);
    }
    std::cout << "valid: "
              << "ny"[rtn] << std::endl;
    return rtn;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;

    try
    {
        using Func = std::function<bool(ck_tile::ArgParser)>;
        const std::unordered_map<std::string, Func> run_map = {
            {"fp32", run_transpose<float>},
            {"fp16", run_transpose<ck_tile::fp16_t>},
            {"bf16", run_transpose<ck_tile::bf16_t>},
            {"int8", run_transpose<ck_tile::int8_t>}};
        return run_map.at(args.get_str("pr"))(args) ? 0 : -1;
    }
    catch(const std::out_of_range&)
    {
        std::cerr << "Error: Unsupported precision type '" << args.get_str("pr") << "'\n";
        return -1;
    }
}
