// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/enable_if.hpp"

namespace ck {
#ifdef __HIPCC_RTC__
template <bool B>
using bool_constant = integral_constant<bool, B>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

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
CK_BUILTIN_TYPE_TRAIT1(is_pointer);
CK_BUILTIN_TYPE_TRAIT1(is_reference);
CK_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
CK_BUILTIN_TYPE_TRAIT1(is_unsigned);
CK_BUILTIN_TYPE_TRAIT2(is_base_of);

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

template<class T> struct is_const          : false_type {};
template<class T> struct is_const<const T> : true_type {};
template< class T >
inline constexpr bool is_const_v = is_const<T>::value;

template< class T >
inline constexpr bool is_reference_v = is_reference<T>::value;

template<class T> struct remove_const { typedef T type; };
template<class T> struct remove_const<const T> { typedef T type; };
template< class T >
using remove_const_t = typename remove_const<T>::type;

template< class T >
inline constexpr bool is_class_v = is_class<T>::value;

template< class T >
inline constexpr bool is_trivially_copyable_v = is_trivially_copyable<T>::value;

template< class... >
using void_t = void;

using __hip::declval;
#else
#include <utility>
#include <type_traits>
using std::forward;
using std::is_base_of;
using std::is_class;
using std::is_pointer;
using std::is_reference;
using std::is_trivially_copyable;
using std::is_unsigned;
using std::remove_cv;
using std::remove_pointer;
using std::remove_reference;
using std::is_const_v;
using std::is_reference_v;
using std::remove_const_t;
using std::is_class_v;
using std::is_trivially_copyable_v;
using std::void_t;
using std::false_type;
using std::true_type;
using std::declval;
#endif

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
using remove_reference_t = typename remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

template <typename T>
inline constexpr bool is_pointer_v = is_pointer<T>::value;

template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");
    static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

} // namespace ck
