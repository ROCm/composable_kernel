#ifndef CK_ELEMENT_WISE_OPERATION_HPP
#define CK_ELEMENT_WISE_OPERATION_HPP

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        y = x;
    }
};

struct AddRelu
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const
    {
#if 0
        T a = x0 + x1;
        y   = a > 0 ? a : 0;
#else
        // hack for Hardswich
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        y       = c;
#endif
    }
};

struct AddReluAdd
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1, const T& x2) const
    {
#if 0
        T a = x0 + x1;
        T b = a > 0 ? a : 0;
        y   = b + x2;
#else
        // hack for Hardswich
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
#endif
    }
};

// Unary operators are usually called element-wisely before the reduction is executed on the
// elements.
// They are needed for easy implementation of reduction types of AVG, NRM1, NRM2

template <typename Y, typename X, bool hasDividing = false> 
struct unary_identic
{
    __host__ __device__ unary_identic(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    };
};

template <typename Y, typename X>
struct unary_identic<Y, X, true>
{   
    __host__ __device__ unary_identic(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };
    
    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x) * type_convert<Y>(scaler);
    };

    float scaler = 1.0f;
};

template <typename Y, typename X, bool hasDividing = false>
struct unary_square
{
    __host__ __device__ unary_square(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x) * type_convert<Y>(x);
    };
};

template <typename Y, typename X>
struct unary_square<Y, X, true>
{
    __host__ __device__ unary_square(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x) * type_convert<Y>(x) * type_convert<Y>(scaler);
    };

    float scaler = 1.0f;
};

static inline __device__ half_t abs(half_t x) { return __habs(x); };
static inline __device__ half_t sqrtf(half_t x) { return hsqrt(x); };

template <typename Y, typename X, bool hasDividing = false>
struct unary_abs
{
    __host__ __device__ unary_abs(const int divider = 1) { (void)divider; };

    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = abs(type_convert<Y>(x));
    };
};

template <typename Y, typename X>
struct unary_abs<Y, X, true>
{
    __host__ __device__ unary_abs(const int divider = 1)
    {
        scaler = 1.0f / static_cast<float>(divider);
    };

    __host__ __device__ inline constexpr void operator()(Y& y, const X& x) const
    {
        y = abs(type_convert<Y>(x) * type_convert<Y>(scaler));
    };

    float scaler = 1.0f;
};

template <typename Y, typename X>
struct unary_sqrt
{
    __host__ __device__ unary_sqrt(const int divider = 1) { (void)divider; };

    __host__ __device__ inline void operator()(Y& y, const X& x) const
    {
        y = sqrtf(type_convert<Y>(x));
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
