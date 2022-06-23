#pragma once

#include <algorithm>
#include <random>

#include "ck/utility/data_type.hpp"

namespace ck {
namespace utils {

// template <typename T, class Enable = void>
// struct FillUniform;

// TODO: what's wrong with this specialization???
// err: segmentation fault in mt19937 - infinite loop like.
// template <typename T>
// struct FillUniform<T, typename std::enable_if<std::is_integral<T>::value &&
//                                               !std::is_same<T, bhalf_t>::value>::type>
// {
//     int a_{0};
//     int b_{5};
//     // T a_ = T{0};
//     // T b_ = T{5};

//     template <typename ForwardIter>
//     void operator()(ForwardIter first, ForwardIter last) const
//     {
//         std::mt19937 gen{11939};
//         std::uniform_int_distribution<int> dis(a_, b_);
//         std::generate(first, last, [&dis, &gen]() { return ck::type_convert<T>(dis(gen)); });
//     }
// };

// struct FillUniform<T, typename std::enable_if<std::is_floating_point<T>::value ||
//                                               std::is_same<T, bhalf_t>::value>::type>
template <typename T>
struct FillUniform
{
    float a_{0};
    float b_{5};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen{11939};
        std::uniform_real_distribution<> dis(a_, b_);
        std::generate(first, last, [&dis, &gen]() { return ck::type_convert<T>(dis(gen)); });
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
