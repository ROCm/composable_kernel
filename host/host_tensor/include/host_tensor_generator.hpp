#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"
#include "data_type.hpp"

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
    int value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return ck::type_convert<T>(value);
    }
};

template <>
struct GeneratorTensor_1<ushort>
{
    float value = 1.0;

    template <typename... Is>
    ushort operator()(Is...)
    {
        return ck::type_convert<ushort>(value);
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
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <>
struct GeneratorTensor_2<ushort>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ushort operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ushort>(tmp);
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
    T min_value = 0;
    T max_value = 1;

    template <typename... Is>
    T operator()(Is...)
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
    ushort operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ushort>(fp32_tmp);
    }
};

template <>
struct GeneratorTensor_3<int8_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        int8_t min_tmp = static_cast<int8_t>(min_value);
        int8_t max_tmp = static_cast<int8_t>(max_value);

        return (std::rand() % (max_tmp - min_tmp)) + min_tmp;
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

#endif
