// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename YDataType>
CK_TILE_HOST void reference_square(HostTensor<YDataType>& y, HostTensor<YDataType>& x)
{
    y.ForEach([&](auto& self, auto i) { self(i) = static_cast<YDataType>(x(i) * x(i)); });
}

} // namespace ck_tile
