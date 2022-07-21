// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>

#include "ck/tensor_description/tensor_descriptor.hpp"

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const ck::TensorDescriptor<Ts...>& desc)
{
    constexpr ck::index_t nDim = ck::remove_cvref_t<decltype(desc)>::GetNumOfDimension();

    os << "{";

    ck::static_for<0, nDim - 1, 1>{}([&](auto i) { os << desc.GetLength(i) << ", "; });

    os << desc.GetLength(ck::Number<nDim - 1>{});

    os << "}";

    return os;
}
