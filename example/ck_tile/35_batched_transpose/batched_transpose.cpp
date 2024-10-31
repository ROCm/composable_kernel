// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "batched_transpose_api.hpp"
#include "batched_transpose.hpp"

#ifndef TEST_BATCHED_TRANSPOSE_VERBOSE
#define TEST_BATCHED_TRANSPOSE_VERBOSE 1
#endif

template <typename T>
void dump_host_tensor_4d(const ck_tile::HostTensor<T>& x)
{
    auto len = x.get_lengths();
    assert(len.size() == 4);
    std::cout << "[";
    for(size_t i = 0; i < len[0]; i++)
    {
        std::cout << i << ": [";
        for(size_t j = 0; j < len[1]; j++)
        {
            std::cout << j << ": [";
            for(size_t k = 0; k < len[2]; k++)
            {
                std::cout << k << ": [";
                for(size_t v = 0; v < len[3]; v++)
                {
                    if constexpr(std::is_same_v<T, ck_tile::fp16_t>)
                    {
                        auto m =
                            ck_tile::type_convert<float>(x(std::vector<std::size_t>{i, j, k, v}));

                        std::cout << m;
                        if(v != len[3] - 1)
                            std::cout << ",";
                    }
                    else
                    {
                        std::cout << x(std::vector<std::size_t>{i, j, k, v}) << " ";
                    }
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------" << std::endl;
}

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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("pr", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("N", "1", "input batch size. ")
        .insert("C", "15", "input channel size.")
        .insert("H", "1", "input height size.")
        .insert("W", "32", "input width size. ")
        .insert("stride_dim0", "480", "input dim0 stride. ")
        .insert("stride_dim1", "32", "input dim1 stride.")
        .insert("stride_dim2", "32", "input dim2 stride.")
        .insert("stride_dim3", "1", "input dim3 stride. ")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "t to 1 will print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Type>
bool test_batched_transpose(ck_tile::ArgParser args)
{
    int validate     = args.get_int("v");
    std::string prec = args.get_str("pr");
    int N            = args.get_int("N");
    int C            = args.get_int("C");
    int H            = args.get_int("H");
    int W            = args.get_int("W");
    int stride_dim0  = args.get_int("stride_dim0");
    int stride_dim1  = args.get_int("stride_dim1");
    int stride_dim2  = args.get_int("stride_dim2");
    int stride_dim3  = args.get_int("stride_dim3");
    int seed         = args.get_int("seed");

    std::string layout_in = "NCHW", layout_out = "NHWC";
    int stride_out_dim0 = stride_dim0, stride_out_dim1 = W * C, stride_out_dim2 = C,
        stride_out_dim3 = 1;
    if(stride_dim0 == C * H * W && stride_dim1 == 1 && stride_dim2 == W * C && stride_dim3 == C)
    {
        layout_in       = "NHWC";
        layout_out      = "NCHW";
        stride_out_dim0 = stride_dim0;
        stride_out_dim1 = H * W;
        stride_out_dim2 = W;
        stride_out_dim3 = 1;
    }

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }
    // int kname = args.get_int("kname");
    // int warmup = args.get_int("warmup");
    // int repeat = args.get_int("repeat");

    // tokens already considered batch size
    ck_tile::HostTensor<Type> x_host({N, C, H, W},
                                     {stride_dim0, stride_dim1, stride_dim2, stride_dim3});
    ck_tile::HostTensor<Type> y_host(
        {N, H, W, C}, {stride_out_dim0, stride_out_dim1, stride_out_dim2, stride_out_dim3});

    {
        // random require per-row unique
        auto rand_gen =
            ck_tile::FillUniformDistribution_Unique<Type>{-5.f, 5.f, static_cast<uint32_t>(seed)};

        for(int i_t = 0; i_t < N; i_t++)
        {
            ck_tile::HostTensor<Type> x_batch({stride_dim0});
            for(int j = 0; j < stride_dim0; j++)
                x_batch(j) = static_cast<Type>(j);
            // rand_gen(x_batch);
            std::copy(x_batch.begin(), x_batch.end(), x_host.begin() + i_t * stride_dim0);
            rand_gen.clear();
        }
    }

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    transpose_kernel_param_t kparam;
    for(auto iter_kparam : get_transpose_all_kernel(prec))
    {
        bool is_valid = is_kernel_valid(N, C, H, W, &iter_kparam, layout_in);
        if(is_valid)
        {
            kparam = iter_kparam;
            break;
        }
    }
    printf("kparam:tile_size:[%d %d] pack[%d, %d] ediv[%d %d]\n",
           kparam.tile_x,
           kparam.tile_y,
           kparam.pack_x,
           kparam.pack_y,
           kparam.ediv_x,
           kparam.ediv_y);

    auto trait = batched_transpose_trait{prec, layout_in};

    uint32_t height    = layout_in == "NCHW" ? C : H * W;
    uint32_t width     = layout_in == "NCHW" ? H * W : C;
    uint32_t dim_h     = (height + kparam.tile_y - 1) / kparam.tile_y;
    uint32_t dim_w     = (width + kparam.tile_x - 1) / kparam.tile_x;
    uint32_t dim_total = N * dim_h * dim_w;
    size_t grid_size   = N * dim_h * dim_w;

    // auto magic_h = ck_tile::magic_division::calculate_magic_numbers(dim_h);
    // auto magic_w = ck_tile::magic_division::calculate_magic_numbers(dim_w);

    batched_transpose_kargs karg = [&]() {
        batched_transpose_kargs a_;
        a_.p_input    = x_dev.GetDeviceBuffer();
        a_.p_output   = y_dev.GetDeviceBuffer();
        a_.batch      = N;
        a_.height     = height;
        a_.width      = width;
        a_.dim_stride = grid_size;
        a_.dim_total  = dim_total;
        a_.magic_h    = 1;
        a_.shift_h    = 1;
        a_.magic_w    = 1;
        a_.shift_w    = 1;
        a_.dim_h      = dim_h;
        a_.dim_w      = dim_w;
        // a_.magic_h  = magic_h.magic;
        // a_.shift_h  = magic_h.shift;
        // a_.magic_w  = magic_w.magic;
        // a_.shift_w  = magic_w.shift;
        return a_;
    }();

#if TEST_BATCHED_TRANSPOSE_VERBOSE
    ck_tile::stream_config sc{nullptr, true};
    // ck_tile::stream_config sc{nullptr};

    auto ms = batched_transpose(trait, karg, sc);
    printf("[%s]N:%d, C:%d, H:%d, W:%d, layout_in:%s, %f\n",
           prec.c_str(),
           N,
           C,
           H,
           W,
           layout_in.c_str(),
           ms);
    if(ms < 0)
        printf("not supported\n");
    fflush(stdout);
#else
    ck_tile::stream_config sc{nullptr};
    auto ms = batched_transpose(trait, karg, sc);
#endif
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
            {N, H, W, C}, {stride_out_dim0, stride_out_dim1, stride_out_dim2, stride_out_dim3});

        // dump_host_tensor_4d(x_host);
        ck_tile::reference_batched_transpose<Type>(x_host, y_ref, layout_in);
        // dump_host_tensor_4d(y_ref);

        // printf("y host:\n");
        // dump_host_tensor_4d(y_host);

        auto [rtol, atol] = get_elimit<Type>("");

        rtn &= ck_tile::check_err(
            y_host, y_ref, std::string("y Error: Incorrect results!"), rtol, atol);
    }
#if TEST_BATCHED_TRANSPOSE_VERBOSE
    printf("valid:%s\n", rtn ? "y" : "n");
    fflush(stdout);
#endif
    return rtn;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;
    std::string prec = args.get_str("pr");

    bool r = true;
    if(prec.compare("fp32") == 0)
    {
        r &= test_batched_transpose<float>(args);
    }
    else if(prec.compare("fp16") == 0)
    {
        r &= test_batched_transpose<ck_tile::fp16_t>(args);
    }
    else if(prec.compare("bf16") == 0)
    {
        r &= test_batched_transpose<ck_tile::bf16_t>(args);
    }
    else if(prec.compare("int8") == 0)
    {
        r &= test_batched_transpose<ck_tile::int8_t>(args);
    }

    return r ? 0 : -1;
}
