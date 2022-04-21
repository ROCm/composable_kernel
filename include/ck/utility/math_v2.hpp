#ifndef CK_MATH_V2_HPP
#define CK_MATH_V2_HPP

#include <cmath>
#include "data_type.hpp"
#include "half.hpp"

namespace ck {
namespace math {

static inline __host__ float abs(float x) { return std::abs(x); };

static inline __host__ double abs(double x) { return std::abs(x); };

static inline __host__ int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

static inline __host__ half_t abs(half_t x)
{
    half_float::half xx = *reinterpret_cast<half_float::half*>(&x);

    half_float::half abs_xx = half_float::abs(xx);

    half_t abs_x = *reinterpret_cast<half_t*>(&abs_xx);

    return abs_x;
};

static inline __host__ float isnan(float x) { return std::isnan(x); };

static inline __host__ double isnan(double x) { return std::isnan(x); };

static inline __host__ int8_t isnan(int8_t x)
{
    (void)x;
    return false;
};

static inline __host__ int32_t isnan(int32_t x)
{
    (void)x;
    return false;
};

static inline __host__ bool isnan(half_t x)
{
    half_float::half xx = *reinterpret_cast<half_float::half*>(&x);

    return half_float::isnan(xx);
};

} // namespace math
} // namespace ck

#endif
