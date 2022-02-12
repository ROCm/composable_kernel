#ifndef CK_INTEGRAL_CONSTANT_HPP
#define CK_INTEGRAL_CONSTANT_HPP

namespace ck {

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};

template <typename Index0, Index0 X, typename Index1, Index1 Y>
__host__ __device__ constexpr auto operator+(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return integral_constant<decltype(X + Y), X + Y>{};
}

template <typename Index0, Index0 X, typename Index1, Index1 Y>
__host__ __device__ constexpr auto operator-(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y <= X, "wrong!");
    return integral_constant<decltype(X - Y), X - Y>{};
}

template <typename Index0, Index0 X, typename Index1, Index1 Y>
__host__ __device__ constexpr auto operator*(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    return integral_constant<decltype(X * Y), X * Y>{};
}

template <typename Index0, Index0 X, typename Index1, Index1 Y>
__host__ __device__ constexpr auto operator/(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X / Y), X / Y>{};
}

template <typename Index0, Index0 X, typename Index1, Index1 Y>
__host__ __device__ constexpr auto operator%(integral_constant<Index0, X>,
                                             integral_constant<Index1, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X % Y), X % Y>{};
}

} // namespace ck
#endif
