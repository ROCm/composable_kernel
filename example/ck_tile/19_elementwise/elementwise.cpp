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
#include "elementwise_api.hpp"

#ifndef TEST_ELEMENTWISE_VERBOSE
#define TEST_ELEMENTWISE_VERBOSE 1
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

struct Cast
{
    template <typename DstType, typename SrcType>
    CK_TILE_HOST_DEVICE void operator()(DstType& y, const SrcType& x) const
    {
        y = ck_tile::type_convert<DstType>(x);
    };
};

// CPU reference
template <typename DstType, typename SrcType, typename UnaryF>
auto reference_elementwise_unary(const ck_tile::HostTensor<SrcType>& x)
{
    using namespace ck_tile;
    auto y = ck_tile::HostTensor<DstType>(x.get_lengths());
    y.ForEach([&](auto& self, auto idx) { UnaryF{}(self(idx), x(idx)); });

    return y;
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
        .insert("op", "cast", "which elementwise operator to run")
        .insert("pr_i", "fp16", "input precision")
        .insert("pr_o", "fp32", "output precision")
        .insert("n", "1000", "number of pixels to cast")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "t to 1 will print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DstType, typename SrcType>
bool test_cast(ck_tile::ArgParser args)
{
    int validate            = args.get_int("v");
    std::string input_prec  = args.get_str("pr_i");
    std::string output_prec = args.get_str("pr_o");
    uint64_t num_pixels     = args.get_uint64("n");
    int seed                = args.get_int("seed");
    if(seed < 0)
    {
        seed = std::time(nullptr);
    }

    // tokens already considered batch size
    ck_tile::HostTensor<SrcType> x_host({num_pixels});
    ck_tile::HostTensor<DstType> y_host({num_pixels});

    ck_tile::FillUniformDistribution<SrcType>{-5, 5, seed}(x_host);

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    elementwise_trait trait = [&]() {
        elementwise_trait t_;
        t_.input_type  = input_prec;
        t_.output_type = output_prec;
        t_.op          = std::string("cast");
        return t_;
    }();

    elementwise_kargs karg = [&]() {
        elementwise_kargs a_;
        a_.p_input    = x_dev.GetDeviceBuffer();
        a_.p_output   = y_dev.GetDeviceBuffer();
        a_.num_pixels = num_pixels;
        return a_;
    }();

#if TEST_ELEMENTWISE_VERBOSE
    ck_tile::stream_config sc{nullptr, true};
    // ck_tile::stream_config sc{nullptr};
    auto ms = elementwise(trait, karg, sc);
    printf(
        "[cast] %s->%s, n:%lu,  ms:%f, ", input_prec.c_str(), output_prec.c_str(), num_pixels, ms);
    if(ms < 0)
        printf("not supported\n");
    fflush(stdout);
#else
    ck_tile::stream_config sc{nullptr};
    auto ms = elementwise_unary(trait, karg, sc);
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
        auto y_ref = reference_elementwise_unary<DstType, SrcType, Cast>(x_host);

        auto [rtol, atol] = get_elimit<SrcType>("");

        rtn &= ck_tile::check_err(
            y_host, y_ref, std::string("Value Error: Incorrect results!"), rtol, atol);

        printf("valid:%s", rtn ? "y" : "n");
        fflush(stdout);
    }
#if TEST_ELEMENTWISE_VERBOSE
    printf("\n");
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
    std::string output_prec = args.get_str("pr_o");
    std::string op          = args.get_str("op");

    bool r = true;
    if(op.compare("cast") == 0)
    {
        if(input_prec.compare("fp16") == 0 && output_prec.compare("fp32") == 0)
        {
            r &= test_cast<float, ck_tile::fp16_t>(args);
        }
        else if(input_prec.compare("fp32") == 0 && output_prec.compare("fp16") == 0)
        {
            r &= test_cast<ck_tile::fp16_t, float>(args);
        }
    }

    return r ? 0 : -1;
}
