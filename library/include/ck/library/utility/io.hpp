// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}
