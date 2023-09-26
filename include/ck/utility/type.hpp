// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/integral_constant.hpp"

#ifdef __HIPCC_RTC__
namespace std {
// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAIT1(name)   \
    template <class T>                       \
    struct name : bool_constant<__##name(T)> \
    {                                        \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAIT2(name)      \
    template <class T, class U>                 \
    struct name : bool_constant<__##name(T, U)> \
    {                                           \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAITN(name)       \
    template <class... Ts>                       \
    struct name : bool_constant<__##name(Ts...)> \
    {                                            \
    }

// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_arithmetic);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_destructible);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_nothrow_destructible);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_pointer);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_scalar);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_signed);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_void);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_abstract);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_aggregate);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_array);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_class);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_compound);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_const);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_empty);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_enum);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_final);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_floating_point);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_function);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_fundamental);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_integral);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_literal_type);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_lvalue_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_function_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_object_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_object);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_pod);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_polymorphic);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_rvalue_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_standard_layout);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivial);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivially_destructible);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_union);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_unsigned);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_volatile);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_assignable);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_base_of);
// MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_convertible);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_nothrow_assignable);
// MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_same);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_trivially_assignable);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_constructible);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_nothrow_constructible);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_trivially_constructible);
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
/// Default deleter
template <typename T>
struct default_delete
{
    void operator()(T* ptr) const { delete ptr; }
};

/// Partial specialization for deleting array types
template <typename T>
struct default_delete<T[]>
{
    void operator()(T* ptr) const { delete[] ptr; }
};

/// std::unique_ptr
template <class T, class Deleter = default_delete<T>>
class unique_ptr
{
    public:
    typedef T* pointer;
    typedef T element_type;
    typedef Deleter deleter_type;

    private:
    /// Pointer to memory
    pointer _ptr;

    /// Deleter
    deleter_type _deleter;

    public:
    unique_ptr() : _ptr(nullptr) {}
    unique_ptr(pointer p) : _ptr(p) {}

    ~unique_ptr()
    {
        if(_ptr)
        {
            _deleter(_ptr);
        }
    }
    /// Returns a pointer to the managed object or nullptr if no object is owned.
    pointer get() const noexcept { return _ptr; }

    /// Releases ownership of the managed object, if any
    pointer release() noexcept
    {
        pointer p(_ptr);
        _ptr = nullptr;
        return p;
    }

    /// Replaces the managed object, deleting the old object.
    void reset(pointer p = pointer()) noexcept
    {
        pointer old_ptr = _ptr;
        _ptr            = p;
        if(old_ptr != nullptr)
        {
            get_deleter()(old_ptr);
        }
    }

    /// Swaps the managed objects with *this and another unique_ptr
    void swap(unique_ptr& other) noexcept { swap(_ptr, other._ptr); }

    /// Returns the deleter object
    Deleter& get_deleter() noexcept { return _deleter; }

    /// Returns the deleter object
    Deleter const& get_deleter() const noexcept { return _deleter; }

    /// Checks whether an object is owned
    operator bool() const noexcept { return _ptr != nullptr; }

    /// Dereferences the unique_ptr
    T& operator*() const { return *_ptr; }

    /// Returns a pointer to the managed object
    pointer operator->() const noexcept { return _ptr; }

    /// Array access to managed object
    T& operator[](size_t i) const { return _ptr[i]; }
};

/// Specializes the swap algorithm
template <typename T, typename Deleter>
void swap(unique_ptr<T, Deleter>& lhs, unique_ptr<T, Deleter>& rhs) noexcept
{
    lhs.swap(rhs);
}

template <class T>
struct remove_extent
{
    using type = T;
};

template <class T>
struct remove_extent<T[]>
{
    using type = T;
};

template <class T, std::size_t N>
struct remove_extent<T[N]>
{
    using type = T;
};

template <class T>
using remove_extent_t = typename remove_extent<T>::type;

namespace detail {
template <class>
constexpr bool is_unbounded_array_v = false;
template <class T>
constexpr bool is_unbounded_array_v<T[]> = true;

template <class>
constexpr bool is_bounded_array_v = false;
template <class T, std::size_t N>
constexpr bool is_bounded_array_v<T[N]> = true;
} // namespace detail

template <class T, class... Args>
enable_if_t<!is_array<T>::value, unique_ptr<T>> make_unique(Args&&... args)
{
    return unique_ptr<T>(new T(forward<Args>(args)...));
}

template <class T>
enable_if_t<detail::is_unbounded_array_v<T>, unique_ptr<T>> make_unique(uint64_t n)
{
    return unique_ptr<T>(new remove_extent_t<T>[n]());
}

template <class T, class... Args>
enable_if_t<detail::is_bounded_array_v<T>> make_unique(Args&&...) = delete;
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
