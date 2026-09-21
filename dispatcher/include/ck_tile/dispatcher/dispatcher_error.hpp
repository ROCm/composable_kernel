// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

namespace ck_tile {
namespace dispatcher {

struct DispatcherError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct NoKernelFound : DispatcherError
{
    using DispatcherError::DispatcherError;
};

struct UnsupportedProblem : DispatcherError
{
    using DispatcherError::DispatcherError;
};

} // namespace dispatcher
} // namespace ck_tile
