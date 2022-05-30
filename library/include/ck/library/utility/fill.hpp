#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include "data_type.hpp"

namespace ck {
namespace utils {

template <typename T>
struct FillUniform
{
    float a_{-5.f};
    float b_{5.f};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(11939);
        std::uniform_real_distribution<> dis(a_, b_);
        std::generate(first, last, [&dis, &gen]() { return ck::type_convert<T>(dis(gen)); });
    }
};

template <typename T>
struct FillUniformIntegerValue
{
    float a_{-5.f};
    float b_{5.f};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(11939);
        std::uniform_real_distribution<> dis(a_, b_);
        std::generate(
            first, last, [&dis, &gen]() { return ck::type_convert<T>(std::round(dis(gen))); });
    }
};

template <typename T>
struct FillMonotonicSeq
{
    T init_value_{0};
    T step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, n = init_value_]() mutable {
            auto tmp = n;
            n += step_;
            return tmp;
        });
    }
};

template <typename T>
struct FillConstant
{
    T value_{0};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::fill(first, last, value_);
    }
};

} // namespace utils
} // namespace ck
