// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef STATIC_ASSERT
#ifndef NDEBUG
#define STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#else
#define STATIC_ASSERT(...)
#endif
#endif
