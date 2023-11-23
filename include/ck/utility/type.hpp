// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/integral_constant.hpp"

namespace ck {
#ifdef __HIPCC_RTC__
// NOLINTNEXTLINE
#define CK_BUILTIN_TYPE_TRAIT1(name)         \
    template <class T>                       \
    struct name : bool_constant<__##name(T)> \
    {                                        \
    }

// NOLINTNEXTLINE
#define CK_BUILTIN_TYPE_TRAIT2(name)            \
    template <class T, class U>                 \
    struct name : bool_constant<__##name(T, U)> \
    {                                           \
    }

// NOLINTNEXTLINE
#define CK_BUILTIN_TYPE_TRAITN(name)             \
    template <class... Ts>                       \
    struct name : bool_constant<__##name(Ts...)> \
    {                                            \
    }

CK_BUILTIN_TYPE_TRAIT1(is_class);
CK_BUILTIN_TYPE_TRAIT1(is_const);
CK_BUILTIN_TYPE_TRAIT1(is_pointer);
CK_BUILTIN_TYPE_TRAIT1(is_reference);
CK_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
CK_BUILTIN_TYPE_TRAIT1(is_unsigned);
CK_BUILTIN_TYPE_TRAIT2(is_base_of);

template <class T>
struct remove_const
{
    typedef T type;
};
template <class T>
struct remove_const<const T>
{
    typedef T type;
};

template <class T>
struct remove_cv
{
    using type = T;
};

template <class T>
struct remove_cv<const T> : remove_cv<T>
{
};

template <class T>
struct remove_cv<volatile T> : remove_cv<T>
{
};

template <class T>
struct remove_reference
{
    typedef T type;
};
template <class T>
struct remove_reference<T&>
{
    typedef T type;
};
template <class T>
struct remove_reference<T&&>
{
    typedef T type;
};

template <class T>
struct remove_pointer
{
    typedef T type;
};
template <class T>
struct remove_pointer<T*>
{
    typedef T type;
};
template <class T>
struct remove_pointer<T* const>
{
    typedef T type;
};
template <class T>
struct remove_pointer<T* volatile>
{
    typedef T type;
};
template <class T>
struct remove_pointer<T* const volatile>
{
    typedef T type;
};

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type& t_) noexcept
{
    return static_cast<T&&>(t_);
}

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type&& t_) noexcept
{
    return static_cast<T&&>(t_);
}
template <typename... Ts>
struct make_void
{
    typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// namespace detail {
// template <class T>
// struct type_identity
// {
//     using type = T;
// };

// template <class T> // Note that `cv void&` is a substitution failure
// auto try_add_lvalue_reference(int) -> type_identity<T&>;
// template <class T> // Handle T = cv void case
// auto try_add_lvalue_reference(...) -> type_identity<T>;

// template <class T>
// auto try_add_rvalue_reference(int) -> type_identity<T&&>;
// template <class T>
// auto try_add_rvalue_reference(...) -> type_identity<T>;
// } // namespace detail

// template <class T>
// struct add_lvalue_reference : decltype(detail::try_add_lvalue_reference<T>(0))
// {
// };

// template <class T>
// struct add_rvalue_reference : decltype(detail::try_add_rvalue_reference<T>(0))
// {
// };

// template <class T>
// typename add_rvalue_reference<T>::type declval();

template <class T, class U = T&&>
U private_declval(int);

template <class T>
T private_declval(long);

template <class T>
auto declval() noexcept -> decltype(private_declval<T>(0));
#else
#include <utility>
#include <type_traits>
using std::declval;
using std::forward;
using std::is_base_of;
using std::is_class;
using std::is_const;
using std::is_pointer;
using std::is_reference;
using std::is_trivially_copyable;
using std::is_unsigned;
using std::remove_const;
using std::remove_cv;
using std::remove_pointer;
using std::remove_reference;
using std::void_t;
#endif

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename T>
inline constexpr bool is_reference_v = is_reference<T>::value;

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;

template <typename X, typename Y>
inline constexpr bool is_base_of_v = is_base_of<X, Y>::value;

template <typename T>
inline constexpr bool is_const_v = is_const<T>::value;

template <typename T>
inline constexpr bool is_unsigned_v = is_unsigned<T>::value;

template <class T>
using remove_const_t = typename remove_const<T>::type;

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

template <typename T>
inline constexpr bool is_pointer_v = is_pointer<T>::value;

template <typename Y,
          typename X,
          typename ck::enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST
    Y y;

    __builtin_memcpy(&y, &x, sizeof(X));

    return y;
#else
    union AsType
    {
        X x;
        Y y;
    };

    return AsType{x}.y;
#endif
}

} // namespace ck
