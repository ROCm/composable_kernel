#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"

template <typename T = float>
struct GeneratorTensor_1
{
    int value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        return value;
    }
};

template <>
struct GeneratorTensor_1<ushort>
{
    float value = 1.0;

    template <typename... Is>
    float operator()(Is...)
    {
        return float_to_bfloat16(value);
    }
};

struct GeneratorTensor_0
{
    int value = 0;

    template <typename... Is>
    float operator()(Is...)
    {
        return value;
    }
};

template <typename T = float>
struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <>
struct GeneratorTensor_2<ushort>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return float_to_bfloat16(tmp);
    }
};

template <typename T = float>
struct GeneratorTensor_3
{
    T min_value = 0;
    T max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return min_value + tmp * (max_value - min_value);
    }
};

template <>
struct GeneratorTensor_3<ushort>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return float_to_bfloat16(fp32_tmp);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

#endif
