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

template <class X, class Y>
struct is_same : public integral_constant<bool, false>
{
};

template <class X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator+(Number<X>, Number<Y>)
{
    return Number<X + Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator-(Number<X>, Number<Y>)
{
    static_assert(Y <= X, "wrong!");
    return Number<X - Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator*(Number<X>, Number<Y>)
{
    return Number<X * Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator/(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X / Y>{};
}

template <index_t X, index_t Y>
__host__ __device__ constexpr auto operator%(Number<X>, Number<Y>)
{
    static_assert(Y > 0, "wrong!");
    return Number<X % Y>{};
}

#if 0
static constexpr Number<0> 0_c;
static constexpr Number<1> 1_c;
static constexpr Number<2> 2_c;
static constexpr Number<3> 3_c;
static constexpr Number<4> 4_c;
static constexpr Number<5> 5_c;
static constexpr Number<6> 6_c;
static constexpr Number<7> 7_c;
static constexpr Number<8> 8_c;
static constexpr Number<9> 9_c;
#endif

} // namespace ck
#endif
