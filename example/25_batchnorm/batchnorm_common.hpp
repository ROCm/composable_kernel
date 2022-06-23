#pragma once

#include <cassert>
#include <vector>

namespace batchnorm {

namespace detail {

// binary operation used to calculate variance from mean and meansquare
struct Variance
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& mean, const T& meansquare) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = meansquare - mean * mean;
    };
};

// binary operation used to update the moving average of mean and variance
struct MovingAverage
{
    MovingAverage(double factor) : factor_(factor){};

    __host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
    {
        y = x0 * static_cast<float>(1.0 - factor_) + x1 * static_cast<float>(factor_);
    }

    __host__ __device__ constexpr void
    operator()(double& y, const double& x0, const float& x1) const
    {
        y = x0 * (1.0 - factor_) + x1 * factor_;
    }

    double factor_;
};

// 4-ary operation used in middle of batchnorm-backward
struct NormalizeAndMultiply
{
    template <typename T>
    __host__ __device__ constexpr void
    operator()(T& z, const T& x, const T& mean, const T& invVariance, const T& val4) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        z = (x - mean) * invVariance * val4;
    };
};

// 5-ary operation used as final step for calculating diffX of batchnorm-backward
struct FinalDiffX
{
    __host__ __device__ FinalDiffX(int32_t reduceSize) : reduceSize_(reduceSize){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& z,
                                                  const T& dy,
                                                  const T& invVariance,
                                                  const T& scale,
                                                  const T& biasDiff,
                                                  const T& val5) const
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        z = static_cast<T>(1.0f) / static_cast<T>(reduceSize_) * invVariance * scale *
            (static_cast<T>(reduceSize_) * dy - biasDiff - val5);
    };

    int32_t reduceSize_;
};

struct NormalizeInInfer
{
    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T>
    __host__ __device__ constexpr void operator()(
        T& y, const T& x, const T& mean, const T& variance, const T& gamma, const T& beta) const;

    template <>
    __host__ __device__ constexpr void operator()<float>(float& y,
                                                         const float& x,
                                                         const float& mean,
                                                         const float& variance,
                                                         const float& gamma,
                                                         const float& beta) const
    {
        using ck::math::sqrt;

        y = ((x - mean) / sqrt(variance + static_cast<float>(epsilon_))) * gamma + beta;
    };

    template <>
    __host__ __device__ constexpr void operator()<double>(double& y,
                                                          const double& x,
                                                          const double& mean,
                                                          const double& variance,
                                                          const double& gamma,
                                                          const double& beta) const
    {
        using ck::math::sqrt;

        y = ((x - mean) / sqrt(variance + epsilon_)) * gamma + beta;
    };

    double epsilon_;
};

}; // end of namespace detail

template <int Rank, int NumReduceDim>
static inline std::vector<int> get_invariant_dims(const std::vector<int>& reduceDims)
{
    assert(NumReduceDim == reduceDims.size());

    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::vector<int> invariantDims;

    // collect invariant dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims.push_back(i);
        };

    return invariantDims;
};

}; // end of namespace batchnorm
