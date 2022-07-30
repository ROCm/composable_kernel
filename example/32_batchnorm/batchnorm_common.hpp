// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>
#include <array>
#include <type_traits>

#include "ck/utility/data_type.hpp"

// binary operation used to calculate invVariance from mean and meansquare
struct InvVariance
{
    InvVariance(double epsilon) : epsilon_(epsilon){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& mean, const T& meansquare) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T tmp_epsilon = type_convert<T>(epsilon_);

        y = meansquare - mean * mean;
        y = 1.0f / sqrt(tmp_epsilon + y);
    };

    double epsilon_;
};

// (4-in, 2-out) element-wise operation used to update the moving average of mean and variance
struct MovingAverage
{
    MovingAverage(double factor) : factor_(factor){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y0,
                                                  T& y1,
                                                  const T& mean,
                                                  const T& runningMean,
                                                  const T& meansquare,
                                                  const T& runningVariance) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;

        T tmp_factor = type_convert<T>(factor_);
        T variance   = meansquare - mean * mean;

        y0 = runningMean * (type_convert<T>(1.0f) - tmp_factor) + mean * tmp_factor;
        y1 = runningVariance * (type_convert<T>(1.0f) - tmp_factor) + variance * tmp_factor;
    };

    double factor_;
};

struct MovingAverageAndInvVariance
{
    MovingAverageAndInvVariance(double epsilon, double factor)
        : epsilon_(epsilon), factor_(factor){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y0, // resultRunningMean
                                                  T& y1, // resultRunningVariance
                                                  T& y2, // saveInvVariance
                                                  const T& mean,
                                                  const T& runningMean,
                                                  const T& meansquare,
                                                  const T& runningVariance) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T tmp_epsilon = type_convert<T>(epsilon_);
        T tmp_factor  = type_convert<T>(factor_);
        T variance    = meansquare - mean * mean;

        y0 = runningMean * (type_convert<T>(1.0f) - tmp_factor) + mean * tmp_factor;
        y1 = runningVariance * (type_convert<T>(1.0f) - tmp_factor) + variance * tmp_factor;

        y2 = 1.0f / sqrt(tmp_epsilon + variance);
    };

    double epsilon_;
    double factor_;
};

struct NormalizeInInfer
{
    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& variance,
                                                  const T2& gamma,
                                                  const T2& beta) const
    {
        static_assert(std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) * gamma + beta;
        y     = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

struct NormalizeInForward
{
    NormalizeInForward(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& meansquare,
                                                  const T2& gamma,
                                                  const T2& beta) const
    {
        static_assert(std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;
        T2 variance = meansquare - mean * mean;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) * gamma + beta;
        y     = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

template <int Rank, int NumReduceDim>
static inline std::array<int, Rank - NumReduceDim>
get_invariant_dims(const std::array<int, NumReduceDim>& reduceDims)
{
    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::array<int, Rank - NumReduceDim> invariantDims;

    // collect invariant dimensions
    int dim = 0;
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims[dim] = i;
            dim++;
        };

    return invariantDims;
};
