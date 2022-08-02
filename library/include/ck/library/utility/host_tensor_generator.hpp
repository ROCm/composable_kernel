// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cmath>
#include <numeric>

#include "ck/ck.hpp"

template <typename T>
struct GeneratorTensor_0
{
    template <typename... Is>
    T operator()(Is...)
    {
        return T{0};
    }
};

template <typename T>
struct GeneratorTensor_1
{
    T value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return value;
    }
};

template <>
struct GeneratorTensor_1<ck::bhalf_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        return ck::type_convert<ck::bhalf_t>(value);
    }
};

template <>
struct GeneratorTensor_1<int8_t>
{
    int8_t value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return value;
    }
};

template <typename T>
struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return static_cast<T>((std::rand() % (max_value - min_value)) + min_value);
    }
};

template <>
struct GeneratorTensor_2<ck::bhalf_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::bhalf_t>(tmp);
    }
};

template <>
struct GeneratorTensor_2<int8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <typename T>
struct GeneratorTensor_3
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return static_cast<T>(min_value + tmp * (max_value - min_value));
    }
};

template <>
struct GeneratorTensor_3<ck::bhalf_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::bhalf_t>(fp32_tmp);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {static_cast<ck::index_t>(Xs)...};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

template <ck::index_t Dim>
struct GeneratorTensor_Sequential
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return dims[Dim];
    }
};
