// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/utils.hpp"

namespace ck {
namespace host {

std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

const std::unordered_set<std::string>& get_xdlop_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx90a", "gfx908", "gfx940", "gfx942"};
    return supported_archs;
}

/**template <typename T, typename ForwardIterator, typename Size, typename BinaryOperation>
auto accumulate_n(ForwardIterator first, Size count, T init, BinaryOperation op)
    -> decltype(std::accumulate(first, std::next(first, count), init, op))
{
    return std::accumulate(first, std::next(first, count), init, op);
}**/

} // namespace host
} // namespace ck
