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
#include "topk_softmax_api.hpp"

#ifndef TEST_TOPK_SOFTMAX_VERBOSE
#define TEST_TOPK_SOFTMAX_VERBOSE 1
#endif

// set this to 1 if input/output have stride
#ifndef TEST_TOPK_VERIFY_PER_TOKEN
#define TEST_TOPK_VERIFY_PER_TOKEN 1
#endif

template <typename T>
void dump_host_tensor_2d(const ck_tile::HostTensor<T>& x)
{
    auto len = x.get_lengths();
    assert(len.size() == 2);
    std::cout << "[";
    for(size_t i = 0; i < len[0]; i++)
    {
        std::cout << i << ": [";
        for(size_t j = 0; j < len[1]; j++)
        {
            if constexpr(std::is_same_v<T, ck_tile::fp16_t>)
            {
                auto v = ck_tile::type_convert<float>(x(std::vector<std::size_t>{i, j}));

                std::cout << v;
                if(j != len[1] - 1)
                    std::cout << ",";
            }
            else
            {
                std::cout << x(std::vector<std::size_t>{i, j}) << " ";
            }
        }
        std::cout << "]";
        if(i != len[0] - 1)
            std::cout << ",";
        else
            std::cout << "]";
        std::cout << std::endl;
    }
    std::cout << "--------------------" << std::endl;
}

// CPU reference
template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
auto reference_topk_softmax(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::index_t k,
                            ck_tile::index_t dim = -1,
                            bool largest         = true,
                            bool sorted          = true)
{
    using namespace ck_tile;

    auto y = reference_softmax<InputType, WeightType, WeightType>(x, dim);

    auto [y_values, y_indices] = reference_topk(y, k, dim, largest, sorted);

    return ck_tile::make_tuple(y_values, y_indices);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
auto reference_topk_softmax(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::HostTensor<WeightType>& y_values,
                            ck_tile::HostTensor<IndexType>& y_indices,
                            ck_tile::index_t k,
                            ck_tile::index_t dim = -1,
                            bool largest         = true,
                            bool sorted          = true)
{
    using namespace ck_tile;

    // dump_host_tensor_2d(x);

    auto y = reference_softmax<InputType, WeightType, WeightType>(x, dim);

    // dump_host_tensor_2d(y);
    reference_topk(y, y_values, y_indices, k, dim, largest, sorted);
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
        .insert("pr_i", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("pr_w", "fp32", "weight data type(currently only fp32 supported now)")
        .insert("t", "32", "number of input tokens")
        .insert("e", "8", "number of experts")
        .insert("k", "2", "topk")
        .insert("st_i", "-1", "row stride of input, -1 means same as experts")
        .insert("st_o", "-1", "row stride of output/indices, -1 means same as topk")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "t to 1 will print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool test_topk_softmax(ck_tile::ArgParser args)
{
    int validate            = args.get_int("v");
    std::string input_prec  = args.get_str("pr_i");
    std::string weight_prec = args.get_str("pr_w");
    int tokens              = args.get_int("t");
    int experts             = args.get_int("e");
    int topk                = args.get_int("k");
    int seed                = args.get_int("seed");
    int stride_input        = args.get_int("st_i");
    int stride_output       = args.get_int("st_o");
    if(stride_input < 0)
    {
        stride_input = experts;
    }
    if(stride_output < 0)
    {
        stride_output = topk;
    }
    assert(stride_input >= experts);
    assert(stride_output >= topk);

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }
    // int kname = args.get_int("kname");
    // int warmup = args.get_int("warmup");
    // int repeat = args.get_int("repeat");

    if(topk > experts)
    {
#if TEST_TOPK_SOFTMAX_VERBOSE
        printf("topk:%d should smaller than (or equal to) experts:%d\n", topk, experts);
#endif
        return false;
    }

    // tokens already considered batch size
    ck_tile::HostTensor<InputType> x_host({tokens, experts}, {stride_input, 1});
    ck_tile::HostTensor<WeightType> value_host({tokens, topk}, {stride_output, 1});
    ck_tile::HostTensor<IndexType> index_host({tokens, topk}, {stride_output, 1});

    {
        // random require per-row unique
        auto rand_gen = ck_tile::FillUniformDistribution_Unique<InputType>{
            -5.f, 5.f, static_cast<uint32_t>(seed)};

        for(int i_t = 0; i_t < tokens; i_t++)
        {
            ck_tile::HostTensor<InputType> x_row({experts});
            rand_gen(x_row);
            std::copy(x_row.begin(), x_row.end(), x_host.begin() + i_t * stride_input);
            rand_gen.clear();
        }
    }

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_dev(value_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_dev(index_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    {
        // using sss = ck_tile::sequence<2, 3, 4>;
        // using pks = ck_tile::sequence<2, 1, 4>;
        // using ord = ck_tile::sequence<2, 0, 1>;
        // ck_tile::static_uford<sss, pks, ord>{}(
        //     [&](auto i_0, auto i_1, auto i_2, auto i_3) {
        //         i_0.fo_0();
        //         i_1.fo_1();
        //         i_2.fo_2();
        //         i_3.fo_3();
        //     }
        // );

        // constexpr auto uf = ck_tile::static_uford<sss, pks, ord>{};
        // ck_tile::static_for<0, uf.get_num_of_access(), 1>{}([&](auto i_access){
        //     uf([&](auto i_0, auto i_1, auto i_2, auto i_3, auto i_4, auto i_5, auto i_6, auto
        //     i_7) {
        //             decltype(i_0)::push_front(i_access).fo_0();
        //             decltype(i_1)::push_front(i_access).fo_1();
        //             decltype(i_2)::push_front(i_access).fo_2();
        //             decltype(i_3)::push_front(i_access).fo_3();
        //             decltype(i_4)::push_front(i_access).fo_4();
        //             decltype(i_5)::push_front(i_access).fo_5();
        //             decltype(i_6)::push_front(i_access).fo_6();
        //             decltype(i_7)::push_front(i_access).fo_7();
        //         },
        //         i_access);
        // });
    }

    topk_softmax_trait trait = [&]() {
        topk_softmax_trait t_;
        t_.input_type  = input_prec;
        t_.weight_type = weight_prec;
        t_.experts     = experts;
        return t_;
    }();

    topk_softmax_kargs karg = [&]() {
        topk_softmax_kargs a_;
        a_.p_input       = x_dev.GetDeviceBuffer();
        a_.p_output      = value_dev.GetDeviceBuffer();
        a_.p_indices     = index_dev.GetDeviceBuffer();
        a_.num_rows      = tokens;
        a_.num_experts   = experts;
        a_.topk          = topk;
        a_.stride_input  = stride_input;
        a_.stride_output = stride_output;
        return a_;
    }();

#if TEST_TOPK_SOFTMAX_VERBOSE
    ck_tile::stream_config sc{nullptr, true};
    // ck_tile::stream_config sc{nullptr};
    auto ms = topk_softmax(trait, karg, sc);
    printf("[%s|%s]tokens:%d, experts:%d, topk:%d, st_i:%d, st_o:%d, ms:%f, ",
           input_prec.c_str(),
           weight_prec.c_str(),
           tokens,
           experts,
           topk,
           stride_input,
           stride_output,
           ms);
    if(ms < 0)
        printf("not supported\n");
    fflush(stdout);
#else
    ck_tile::stream_config sc{nullptr};
    auto ms = topk_softmax(trait, karg, sc);
#endif
    if(ms < 0)
    {
        return false;
    }

    value_dev.FromDevice(value_host.data());
    index_dev.FromDevice(index_host.data());

    bool rtn = true;
    if(validate)
    {
        // this host buffer will not copy to GPU, so no need use stride
        ck_tile::HostTensor<WeightType> value_ref({tokens, topk}, {stride_output, 1});
        ck_tile::HostTensor<IndexType> index_ref({tokens, topk}, {stride_output, 1});

        // auto [value_ref, index_ref] =
        reference_topk_softmax<InputType, WeightType, IndexType>(
            x_host, value_ref, index_ref, topk);

        auto [rtol, atol] = get_elimit<InputType>("");
#if TEST_TOPK_VERIFY_PER_TOKEN
        for(int i_t = 0; i_t < tokens; i_t++)
        {
            auto s_begin = std::vector<size_t>{static_cast<size_t>(i_t), static_cast<size_t>(0)};
            auto s_end =
                std::vector<size_t>{static_cast<size_t>(i_t + 1), static_cast<size_t>(topk)};
            auto s_value_host = value_host.slice(s_begin, s_end);
            auto s_value_ref  = value_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_value_host,
                                      s_value_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Value Error:"),
                                      rtol,
                                      atol);
            auto s_index_host = index_host.slice(s_begin, s_end);
            auto s_index_ref  = index_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_index_host,
                                      s_index_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Index Error:"),
                                      rtol,
                                      atol);
        }
#else
        rtn &= ck_tile::check_err(
            value_host, value_ref, std::string("Value Error: Incorrect results!"), rtol, atol);
        rtn &= ck_tile::check_err(
            index_host, index_ref, std::string("Index Error: Incorrect results!"), rtol, atol);
#endif
    }
#if TEST_TOPK_SOFTMAX_VERBOSE
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
    std::string input_prec  = args.get_str("pr_i");
    std::string weight_prec = args.get_str("pr_w");

    bool r = true;
    if(input_prec.compare("fp16") == 0 && weight_prec.compare("fp32") == 0)
    {
        r &= test_topk_softmax<ck_tile::fp16_t, float, ck_tile::index_t>(args);
    }
    else if(input_prec.compare("bf16") == 0 && weight_prec.compare("fp32") == 0)
    {
        r &= test_topk_softmax<ck_tile::bf16_t, float, ck_tile::index_t>(args);
    }

    return r ? 0 : -1;
}
