// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename YDataType, typename... Args>
CK_TILE_HOST void reference_add(HostTensor<YDataType>& y, Args&&... rest_args)
{
    // Lambda function implementing a binary operation: addition
    constexpr auto operation = [](auto& accumulator, auto& arg, auto idx) {
        accumulator += ck_tile::type_convert<YDataType>(arg(idx));
    };

    y.ForEach([&](auto& self, auto i) {
        YDataType accumulator = static_cast<YDataType>(0);
        YDataType dummy[]     = {
            static_cast<YDataType>(0),
            ((void)(operation(accumulator, rest_args, i)), static_cast<YDataType>(0))...};
        (void)dummy; // Suppress unused variable warning for dummy array
        self(i) = accumulator;
    });
}

} // namespace ck_tile
