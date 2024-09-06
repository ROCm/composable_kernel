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

// #ifndef TEST_TOPK_SOFTMAX_VERBOSE
// #define TEST_TOPK_SOFTMAX_VERBOSE 0
// #endif

// #define BLOCK_SIZE 256

template <typename T>
void dump_host_tensor_2d(const ck_tile::HostTensor<T>& x)
{
    auto len = x.get_lengths();
    assert(len.size() == 2);
    std::cout << "[";
    for(size_t i = 0; i < len[0]; i++)
    {
        std::cout << "[";
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

    // dump_host_tensor_2d(x);

    auto y = reference_softmax<InputType, WeightType, WeightType>(x, dim);

    // dump_host_tensor_2d(y);
    auto [y_values, y_indices] = reference_topk(y, k, dim, largest, sorted);

    // dump_host_tensor_2d(y_values);
    // dump_host_tensor_2d(y_indices);

    return ck_tile::make_tuple(y_values, y_indices);
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
        .insert(
            "input_prec", "fp16", "input data type. fp8/fp16/fp32 (representing 8/16/32 bit data)")
        .insert("weight_prec", "fp32", "weight data type")
        .insert("t", "32", "number of input tokens")
        .insert("e", "8", "number of experts")
        .insert("k", "2", "topk")
        .insert("kname", "0", "t to 1 will print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool test_topk_softmax(ck_tile::ArgParser args)
{
    int validate            = args.get_int("v");
    std::string input_prec  = args.get_str("input_prec");
    std::string weight_prec = args.get_str("weight_prec");
    int tokens              = args.get_int("t");
    int experts             = args.get_int("e");
    int topk                = args.get_int("k");
    // int kname = args.get_int("kname");
    // int warmup = args.get_int("warmup");
    // int repeat = args.get_int("repeat");
    std::srand(std::time(nullptr));

    // tokens already considered batch size
    ck_tile::HostTensor<InputType> x_host({tokens, experts});
    ck_tile::HostTensor<WeightType> value_host({tokens, topk});
    ck_tile::HostTensor<IndexType> index_host({tokens, topk});

    ck_tile::FillUniformDistribution<InputType>{-5.f, 5.f}(x_host);

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_dev(value_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_dev(index_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    topk_softmax_trait trait = [&]() {
        topk_softmax_trait t_;
        t_.input_type  = input_prec;
        t_.weight_type = weight_prec;
        t_.experts     = experts;
        return t_;
    }();

    topk_softmax_kargs karg = [&]() {
        topk_softmax_kargs a_;
        a_.p_input     = x_dev.GetDeviceBuffer();
        a_.p_output    = value_dev.GetDeviceBuffer();
        a_.p_indices   = index_dev.GetDeviceBuffer();
        a_.num_rows    = tokens;
        a_.num_experts = experts;
        a_.topk        = topk;
        return a_;
    }();

    ck_tile::stream_config sc{nullptr};

    topk_softmax(trait, karg, sc);

    value_dev.FromDevice(value_host.data());
    index_dev.FromDevice(index_host.data());

    bool rtn = true;
    if(validate)
    {
        ck_tile::HostTensor<WeightType> value_host_ref({tokens, topk});
        ck_tile::HostTensor<IndexType> index_host_ref({tokens, topk});

        auto [value_ref, index_ref] =
            reference_topk_softmax<InputType, WeightType, IndexType>(x_host, topk);

        auto [rtol, atol] = get_elimit<InputType>("");
        rtn &= ck_tile::check_err(
            value_host, value_ref, std::string("Value Error: Incorrect results!"), rtol, atol);
        rtn &= ck_tile::check_err(
            index_host, index_ref, std::string("Index Error: Incorrect results!"), rtol, atol);
    }

    return rtn;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;
    std::string input_prec  = args.get_str("input_prec");
    std::string weight_prec = args.get_str("weight_prec");

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
