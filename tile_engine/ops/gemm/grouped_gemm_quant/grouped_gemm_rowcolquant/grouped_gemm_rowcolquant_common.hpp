// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "common/utils.hpp"

// Structure to hold kernel traits for dispatcher
struct KernelTraits
{
    std::string pipeline;  // compv3
    std::string scheduler; // intrawave
    std::string epilogue;  // cshuffle
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;

    // Constructor with defaults
    KernelTraits()
        : pipeline("compv3"),
          scheduler("intrawave"),
          epilogue("cshuffle"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};
