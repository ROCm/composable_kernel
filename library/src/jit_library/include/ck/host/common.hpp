// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>

namespace ck {
namespace host {

struct Solution
{
    std::string template_str;
    std::size_t block_size;
    std::size_t grid_size;
};

enum class DataType
{
    Half,
    Float,
    Int8,
    Int32
};

std::string ToString(DataType dt);

std::unordered_map<std::string_view, std::string_view> GetHeaders();

std::size_t integer_divide_ceil(std::size_t x, std::size_t y);

} // namespace host
} // namespace ck
