// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/numeric.hpp"

#include <tuple>
#include <type_traits>
#include <stdint.h>

namespace ck_tile {

// `always_false_v<T...>` — a value-template that is always `false` but whose
// evaluation is deferred until template instantiation. The canonical use is
// inside the `else` arm of an `if constexpr` chain or under an arch-gated
// `#if` to fire a `static_assert` ONLY when the offending instantiation is
// actually requested, e.g.:
//
//     if constexpr (...) { ... }
//     else { static_assert(always_false_v<T>, "unsupported T"); }
//
// A bare `static_assert(false, ...)` would fire at template-definition
// parse time on conforming compilers, breaking the whole TU.
template <typename...>
inline constexpr bool always_false_v = false;

// remove_cvref_t
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename From, typename To>
struct copy_const
{
    static_assert(!std::is_const_v<From>);

    using type = To;
};

template <typename From, typename To>
struct copy_const<const From, To>
{
    using type = std::add_const_t<typename copy_const<From, To>::type>;
};

template <typename From, typename To>
using copy_const_t = typename copy_const<From, To>::type;

namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type    = Op<Args...>;
};
} // namespace detail

struct nonesuch
{
    ~nonesuch()                     = delete;
    nonesuch(nonesuch const&)       = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

namespace impl {

template <typename T>
using has_is_static = decltype(T::is_static());

template <typename T>
struct is_static_impl
{
    static constexpr bool value = []() {
        if constexpr(is_detected<has_is_static, T>{})
            return T::is_static();
        else
            return std::is_arithmetic<T>::value;
    }();
};
} // namespace impl

template <typename T>
using is_static = impl::is_static_impl<remove_cvref_t<T>>;

template <typename T>
inline constexpr bool is_static_v = is_static<T>::value;

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = is_static<T>;
// TODO: if evaluating a rvalue, e.g. a const integer
// , this helper will also return false, which is not good(?)
//       do we need something like is_constexpr()?

// FIXME: do we need this anymore?
template <
    typename PY,
    typename PX,
    typename std::enable_if<std::is_pointer_v<PY> && std::is_pointer_v<PX>, bool>::type = false>
CK_TILE_HOST_DEVICE PY c_style_pointer_cast([[clang::lifetimebound]] PX p_x)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
    return (PY)p_x; // NOLINT(old-style-cast, cast-align)
#pragma clang diagnostic pop
}

// Template ternary: if Cond == Match, use TrueType, else FalseType
// Usage: if_select_t<T, int, float, double> evaluates to float if T==int, else double
template <typename Cond, typename Match, typename TrueType, typename FalseType>
using if_select_t = std::conditional_t<std::is_same_v<Cond, Match>, TrueType, FalseType>;

template <typename CompareTo, typename... Rest>
struct is_any_of : std::false_type
{
};

template <typename CompareTo, typename FirstType>
struct is_any_of<CompareTo, FirstType> : std::is_same<CompareTo, FirstType>
{
};

template <typename CompareTo, typename FirstType, typename... Rest>
struct is_any_of<CompareTo, FirstType, Rest...>
    : std::integral_constant<bool,
                             std::is_same<CompareTo, FirstType>::value ||
                                 is_any_of<CompareTo, Rest...>::value>
{
};

/**
 * @brief Helper to check if a value is in a list of values
 * @tparam T The type of the search value
 * @tparam Ts The types of the search list values
 * @param search The value to search for
 * @param searchList The list of values to search in
 * @return true if the search value is in the search list, false otherwise
 */
template <typename T, typename... Ts>
// TODO: c++20    requires((std::is_convertible<Ts, T>::value && ...) && (sizeof...(Ts) >= 1))
CK_TILE_HOST_DEVICE static constexpr bool is_any_value_of(T search, Ts... searchList)
{
    static_assert((std::is_convertible<Ts, T>::value && ...),
                  "All searchList values must be convertible to the type of search");
    static_assert(sizeof...(Ts) >= 1, "searchList must contain at least one value");

    return ((search == static_cast<T>(searchList)) || ...);
}

// Helper to check if a type is a specialization of a given template
template <typename Test, template <typename...> class RefTemplate>
struct is_specialization_of : std::false_type
{
};

template <template <typename...> class RefTemplate, typename... Args>
struct is_specialization_of<RefTemplate<Args...>, RefTemplate> : std::true_type
{
};

// Helper to get a tuple element or default type
namespace detail {

template <bool IsWithinBounds, std::size_t Idx, typename Tuple, typename DefaultType>
struct tuple_element_or_default_dispatch
{
    using type = DefaultType;
};

template <std::size_t Idx, typename Tuple, typename DefaultType>
struct tuple_element_or_default_dispatch<true, Idx, Tuple, DefaultType>
{
    using type = std::tuple_element_t<Idx, Tuple>;
};

} // namespace detail

template <typename Tuple_, std::size_t Idx, typename DefaultType>
struct tuple_element_or_default
{
    using Tuple                            = remove_cvref_t<Tuple_>;
    static constexpr bool is_within_bounds = Idx < std::tuple_size_v<Tuple>;
    using type                             = typename detail::
        tuple_element_or_default_dispatch<is_within_bounds, Idx, Tuple, DefaultType>::type;
};
template <typename Tuple_, std::size_t Idx, typename DefaultType>
using tuple_element_or_default_t =
    typename tuple_element_or_default<Tuple_, Idx, DefaultType>::type;

// Helper struct to determine if a type is packed (more than 1 element per byte)
template <typename T>
struct is_packed_type
{
    static constexpr bool value = numeric_traits<T>::PackedSize > 1;
};

template <typename T>
static constexpr bool is_packed_type_v = is_packed_type<T>::value;

// Helper definition to take the largest sizes type
template <typename ADataType, typename BDataType>
using largest_type_t =
    std::conditional_t<sizeof(ADataType) >= sizeof(BDataType), ADataType, BDataType>;

/**
 * @brief Type trait to detect whether a type is a @c std::tuple specialization.
 * @tparam T The type to inspect.
 */
template <typename T>
struct is_std_tuple : std::false_type
{
};

template <typename... Args>
struct is_std_tuple<std::tuple<Args...>> : std::true_type
{
};

template <typename T>
static constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

} // namespace ck_tile
