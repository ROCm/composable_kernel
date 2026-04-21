
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "ck_tile/host/hip_check_error.hpp"

#define HSTU_CHECK(COND, ERR)                  \
    if(!(COND))                                \
    {                                          \
        std::ostringstream ostr;               \
        ostr << "'" #COND "' failed: " << ERR; \
        throw std::runtime_error(ostr.str());  \
    }

static inline int get_number_of_cu()
{
    int device;

    HIP_CHECK_ERROR(hipGetDevice(&device));

    hipDeviceProp_t props;

    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    return props.multiProcessorCount;
}

namespace ck_tile {

namespace detail {

// A helper struct for detecting kUseTrLoad
// T is the pipeline class used by the kernel instance
template <typename T, typename = void>
struct has_use_trload_flag : std::false_type
{
};

template <typename T>
struct has_use_trload_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kUseTrLoad), bool> && T::kUseTrLoad>>
    : std::true_type
{
};

template <typename T>
static inline constexpr bool is_using_trload_v = has_use_trload_flag<T>::value;

} // namespace detail

} // namespace ck_tile
