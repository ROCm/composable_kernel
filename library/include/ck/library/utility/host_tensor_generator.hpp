// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cmath>
#include <numeric>
#include <random>

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
struct GeneratorTensor_1<ck::half_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        return ck::type_convert<ck::half_t>(value);
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

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_1<ck::f8_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        return ck::type_convert<ck::f8_t>(value);
    }
};
#endif

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

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_2<ck::f8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::f8_t>(tmp);
    }
};
#endif

#if defined CK_ENABLE_BF8
template <>
struct GeneratorTensor_2<ck::bf8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bf8_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::bf8_t>(tmp);
    }
};
#endif

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

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_3<ck::f8_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::f8_t>(fp32_tmp);
    }
};
#endif

#if defined CK_ENABLE_BF8
template <>
struct GeneratorTensor_3<ck::bf8_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bf8_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::bf8_t>(fp32_tmp);
    }
};
#endif

template <typename T>
struct GeneratorTensor_4
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = distribution(generator);

        return ck::type_convert<T>(tmp);
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

/**
 * @brief Is used to generate sequential values based on the specified dimension.
 *
 * @tparam T The type of the tensor values.
 * @tparam Dim The specific dimension used for generation.
 *
 * GeneratorTensor_Sequential<1>{} will generate the following values for a 3x3 tensor:
 *
 * 0 1 2
 * 0 1 2
 * 0 1 2
 *
 * Essentially, the values generated are logical coordinates of the generated element that
 * correspond to dimension Dim. E.g. for 2-dimensional tensor and Dim=1, the values are the column
 * indices.
 *
 */
template <typename T, ck::index_t Dim>
struct GeneratorTensor_Sequential
{
    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        float tmp = dims[Dim];
        return ck::type_convert<T>(tmp);
    }
};

template <typename T, size_t NumEffectiveDim = 2>
struct GeneratorTensor_Diagonal
{
    T value{1};

    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        size_t start_dim                            = dims.size() - NumEffectiveDim;
        bool pred                                   = true;
        for(size_t i = start_dim + 1; i < dims.size(); i++)
        {
            pred &= (dims[start_dim] == dims[i]);
        }
        return pred ? value : T{0};
    }
};

/**
 * @brief Used to generate tensor entries from coefficients of Leibniz formula for Pi.
 *
 * @tparam T The type of the tensor values.
 *
 * Usage: For verification of GEMM
 *    a_m_k.GenerateTensorValue(GeneratorTensor_PI<ADataType>{});
 *    b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
 *
 *    c = a * b;
 *
 *    We expect that |c[i][j]-M_PI| <= truncation_error(K)
 */
template <typename T>
struct GeneratorTensor_PI
{
    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        static constexpr double pi = 3.14159265358979323846;

        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        if constexpr(dims.size() > 0)
        {
            constexpr auto last_dim = dims.size() - 1;
            size_t i                = dims[last_dim];
            float fi                = i;
            float tmp               = (i % 2 == 0) ? 4.0 : -4.0;
            tmp /= (2.0 * fi + 1.0);
            return ck::type_convert<T>(tmp);
        }
        else
        {
            return ck::type_convert<T>(pi);
        }
    }

    static double truncation_error(size_t N) { return 4.0 / (2.0 * N + 1.0); }
};

/**
 * @brief Used to generate tensor entries from coefficients of non-alternating version of Leibniz
 * formula for Pi.
 *
 * @tparam T The type of the tensor values.
 *
 * Usage: For verification of GEMM
 *    a_m_k.GenerateTensorValue(GeneratorTensor_PI_A<ADataType>{});
 *    b_k_n.GenerateTensorValue(GeneratorTensor_PI_B<BDataType>{});
 *
 *    c = a * b;
 *
 *    We expect that |c[i][j]-M_PI| <= 0.00013 for K >= 4096 and a,b,c are float.
 */
template <typename T>
struct GeneratorTensor_PI_A
{
    static constexpr double pi = 3.14159265358979323846;

    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        if constexpr(dims.size() > 0)
        {
            constexpr auto last_dim = dims.size() - 1;
            size_t i                = dims[last_dim];
            float fi                = i;
            float tmp               = 2.0 / (4.0 * fi + 1.0);
            return ck::type_convert<T>(tmp);
        }
        else
        {
            return ck::type_convert<T>(pi / 2.0);
        }
    }
};

template <typename T>
struct GeneratorTensor_PI_B
{
    static constexpr double pi = 3.14159265358979323846;

    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        if constexpr(dims.size() > 0)
        {
            size_t i  = dims[0];
            float fi  = i;
            float tmp = 4.0 / (4.0 * fi + 3.0);
            return ck::type_convert<T>(tmp);
        }
        else
        {
            return ck::type_convert<T>(2.0);
        }
    }
};
