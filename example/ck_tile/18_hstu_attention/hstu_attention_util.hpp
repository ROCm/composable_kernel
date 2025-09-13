
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#define HSTU_CHECK(COND, ERR)                  \
    if(!(COND))                                \
    {                                          \
        std::ostringstream ostr;               \
        ostr << "'" #COND "' failed: " << ERR; \
        throw std::runtime_error(ostr.str());  \
    }
