#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"

struct GeneratorTensor_1
{
    int value = 1;

    template <typename... Is>
    double operator()(Is... is)
    {
        return value;
    }
};

struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    double operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

struct GeneratorTensor_3
{
    template <typename... Is>
    double operator()(Is... is)
    {
        std::array<ck::index_t, sizeof...(Is)> dims = {{static_cast<ck::index_t>(is)...}};

        auto f_acc = [](auto a, auto b) { return 10 * a + b; };

        return std::accumulate(dims.begin(), dims.end(), ck::index_t(0), f_acc);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    double operator()(Ts... Xs) const
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
