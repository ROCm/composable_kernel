// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/integral_constant.hpp"

#ifdef __HIPCC_RTC__
namespace std {
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
CK_BUILTIN_TYPE_TRAIT1(is_reference);
CK_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
CK_BUILTIN_TYPE_TRAIT1(is_unsigned);
CK_BUILTIN_TYPE_TRAIT2(is_base_of);
CK_BUILTIN_TYPE_TRAITN(is_constructible);
CK_BUILTIN_TYPE_TRAITN(is_nothrow_constructible);

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
struct remove_volatile
{
    typedef T type;
};
template <class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};

template <class T>
struct remove_cv
{
    typedef typename remove_volatile<typename remove_const<T>::type>::type type;
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

template <class T>
struct is_pointer_helper : std::false_type
{
};

template <class T>
struct is_pointer_helper<T*> : std::true_type
{
};

template <class T>
struct is_pointer : is_pointer_helper<typename std::remove_cv<T>::type>
{
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

template <typename T>
inline constexpr bool is_reference_v = is_reference<T>::value;
} // namespace std
#endif
namespace ck {

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
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
