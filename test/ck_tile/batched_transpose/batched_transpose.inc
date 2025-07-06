// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "batched_transpose.hpp"

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

auto create_args(int argc, char* argv[], int index = 0)
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "whether do CPU validation or not")
        .insert("pr", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("N", "1", "input batch size. ")
        .insert("C", "64", "input channel size.")
        .insert("H", "18", "input height size.")
        .insert("W", "64", "input width size. ")
        .insert("layout_in", "NCHW", "input tensor data layout - NCHW by default")
        .insert("layout_out", "NHWC", "output tensor data layout - NHWC by default ")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "t to 1 will print kernel name");

    bool result = arg_parser.parse(argc, argv, index);
    return std::make_tuple(result, arg_parser);
}

template <typename Type>
bool run_batched_transpose(ck_tile::ArgParser args)
{
    int validate           = args.get_int("v");
    std::string prec       = args.get_str("pr");
    int N                  = args.get_int("N");
    int C                  = args.get_int("C");
    int H                  = args.get_int("H");
    int W                  = args.get_int("W");
    int n_warmup           = args.get_int("warmup");
    int n_repeat           = args.get_int("repeat");
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

    auto trait = batched_transpose_trait{prec, layout_in};

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

    ck_tile::stream_config sc{nullptr, true, n_warmup, n_repeat};

    auto ms = batched_transpose(trait, karg, sc);

    std::size_t num_operations = N * C * H * (W - 1);
    std::size_t num_bytes      = N * C * H * W * sizeof(Type);

    float ave_time   = ms * 1E-3;
    float gb_per_sec = num_bytes / ms * 1.E-6;
    float tflops     = static_cast<float>(num_operations) / ms * 1.E-6;

    std::cout << "Run Batched Transpose kernel with N=" << N << ", C=" << C << ", H=" << H
              << ", W=" << W << ", layout_in=" << layout_in << ", layout_out=" << layout_out
              << " : " << ms << " ms (" << ave_time << " ave_time), " << tflops << " TFlops"
              << gb_per_sec << " GB/s, " << std::endl;

    printf("[%s]N:%d, C:%d, H:%d, W:%d, layout_in:%s, %f\n",
           prec.c_str(),
           N,
           C,
           H,
           W,
           layout_in.c_str(),
           ms);
    if(ms < 0)
        printf("------------------------------------not "
               "supported-------------------------------------\n");
    fflush(stdout);

    if(ms < 0)
    {
        return false;
    }

    y_dev.FromDevice(y_host.data());

    bool rtn = true;
    if(validate)
    {
        // this host buffer will not copy to GPU, so no need use stride
        ck_tile::HostTensor<Type> y_ref(
            {dim_out[0], dim_out[1], dim_out[2], dim_out[3]},
            {stride_dim_out[0], stride_dim_out[1], stride_dim_out[2], stride_dim_out[3]});

        ck_tile::reference_batched_transpose<Type>(x_host, y_ref, layout_in, layout_out);

        auto [rtol, atol] = get_elimit<Type>("");

        rtn &= ck_tile::check_err(
            y_host, y_ref, std::string("y Error: Incorrect results!"), rtol, atol);
    }
    printf("-----------------------------------------------------------------------valid:%s--------"
           "--------------------------------------------------------------------\n",
           rtn ? "y" : "n");
    fflush(stdout);
    return rtn;
}

template <typename PrecType>
bool run_test_case(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return false;

    return run_batched_transpose<PrecType>(args);
}

template <typename PrecType>
bool run_test_cases(std::vector<std::vector<std::string>>& test_cases)
{
    bool valid = true;
    for(std::size_t test_idx = 0; test_idx < test_cases.size(); ++test_idx)
    {
        constexpr int num_args = 7;
        char* argv[num_args];

        assert(test_cases[test_idx].size() == num_args &&
               "invalid number of arguments in test case");

        for(std::size_t idx = 0; idx < test_cases[test_idx].size(); ++idx)
        {
            argv[idx] = test_cases[test_idx][idx].data();
        }

        valid = valid && run_test_case<PrecType>(num_args, argv);

        if(!valid)
            break;
    }

    return valid;
}

std::vector<std::vector<std::string>> generate_test_cases(const std::string prec)
{
    return {
        {"-pr=" + prec, "-N=1", "-C=32", "-H=1", "-W=32", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=1", "-C=64", "-H=1", "-W=64", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=2", "-C=12", "-H=1", "-W=32", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=3", "-C=1334", "-H=1", "-W=37", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=4", "-C=27", "-H=1", "-W=32", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=5", "-C=1234", "-H=1", "-W=12", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=1", "-C=1", "-H=1", "-W=1", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=1", "-C=1", "-H=1", "-W=1", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec,
         "-N=128",
         "-C=1024",
         "-H=64",
         "-W=64",
         "-layout_in=NCHW",
         "-layout_out=NHWC"},
        {"-pr=" + prec,
         "-N=128",
         "-C=1024",
         "-H=64",
         "-W=64",
         "-layout_in=NHWC",
         "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=16", "-C=64", "-H=32", "-W=128", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=16", "-C=64", "-H=128", "-W=32", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=1", "-C=2048", "-H=1", "-W=1", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=1", "-C=2048", "-H=1", "-W=1", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec,
         "-N=1",
         "-C=1",
         "-H=1024",
         "-W=1024",
         "-layout_in=NCHW",
         "-layout_out=NHWC"},
        {"-pr=" + prec,
         "-N=1",
         "-C=1",
         "-H=1024",
         "-W=1024",
         "-layout_in=NHWC",
         "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=8", "-C=16", "-H=8", "-W=16", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=8", "-C=16", "-H=8", "-W=16", "-layout_in=NHWC", "-layout_out=NCHW"},
        {"-pr=" + prec, "-N=1", "-C=64", "-H=1", "-W=1024", "-layout_in=NCHW", "-layout_out=NHWC"},
        {"-pr=" + prec, "-N=1", "-C=64", "-H=1024", "-W=1", "-layout_in=NHWC", "-layout_out=NCHW"}};
}
