#ifndef CK_MATH_V2_HPP
#define CK_MATH_V2_HPP

#include "data_type.hpp"

namespace ck {
namespace math {

static inline __device__ half_t abs(half_t x) { return __habs(x); };
static inline __device__ half_t sqrtf(half_t x) { return hsqrt(x); };
static inline __device__ bool isnan(half_t x) { return __hisnan(x); };

} // namespace math
} // namespace ck

#endif

