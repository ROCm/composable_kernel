#ifndef IS_64BIT_HPP
#define IS_64BIT_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"

namespace ck {

template <typename T>
struct is_64bit
{
};

template <typename T>
inline constexpr bool is_64bit_v = is_64bit<T>::value;

template <>
struct is_64bit<index_t>
{
    static constexpr bool value = false;
};

template <>
struct is_64bit<long_index_t>
{
    static constexpr bool value = true;
};

template <typename T, T N>
struct is_64bit<integral_constant<T, N>>
{
    static constexpr bool value = is_64bit_v<T>;
};

} // namespace ck
#endif
