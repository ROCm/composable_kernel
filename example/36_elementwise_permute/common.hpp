// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;

struct ExecutionConfig final
{
    bool do_verification = true;
    bool time_kernel     = false;
};

struct Problem final
{
    std::array<std::size_t, 4> shape = {4, 16, 32, 32};
    std::array<std::size_t, 4> axes  = {0, 2, 3, 1};
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& config, Problem& problem)
{
    return true;
}
