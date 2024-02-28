// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include <stdint.h>

#pragma once

#define CK_TILE_ARITHMETIC_USING_FLOAT(type_)                        \
    CK_TILE_HOST_DEVICE                                              \
    bool operator==(const type_& x, const type_& y)                  \
    {                                                                \
        return static_cast<float>(x) == static_cast<float>(y);       \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    bool operator!=(const type_& x, const type_& y)                  \
    {                                                                \
        return static_cast<float>(x) != static_cast<float>(y);       \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    bool operator<(const type_& x, const type_& y)                   \
    {                                                                \
        return static_cast<float>(x) < static_cast<float>(y);        \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    bool operator<=(const type_& x, const type_& y)                  \
    {                                                                \
        return static_cast<float>(x) <= static_cast<float>(y);       \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    bool operator>(const type_& x, const type_& y)                   \
    {                                                                \
        return static_cast<float>(x) > static_cast<float>(y);        \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    bool operator>=(const type_& x, const type_& y)                  \
    {                                                                \
        return static_cast<float>(x) >= static_cast<float>(y);       \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator+(const type_& x, const type_& y)                  \
    {                                                                \
        return type_(static_cast<float>(x) + static_cast<float>(y)); \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator-(const type_& x)                                  \
    {                                                                \
        constexpr uint32_t bits = sizeof(type_) * 8;                 \
        constexpr uint32_t mask = 1 << (bits - 1);                   \
        type_ y                 = x;                                 \
        y.data ^= static_cast<typename type_::raw_type>(mask);       \
        return y;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator-(const type_& x, const type_& y)                  \
    {                                                                \
        return type_(static_cast<float>(x) - static_cast<float>(y)); \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator*(const type_& x, const type_& y)                  \
    {                                                                \
        return type_(static_cast<float>(x) * static_cast<float>(y)); \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator/(const type_& x, const type_& y)                  \
    {                                                                \
        return type_(static_cast<float>(x) / static_cast<float>(y)); \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator+=(type_& x, const type_& y)                      \
    {                                                                \
        x = type_(static_cast<float>(x) + static_cast<float>(y));    \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator-=(type_& x, const type_& y)                      \
    {                                                                \
        x = type_(static_cast<float>(x) - static_cast<float>(y));    \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator*=(type_& x, const type_& y)                      \
    {                                                                \
        x = type_(static_cast<float>(x) * static_cast<float>(y));    \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator/=(type_& x, const type_& y)                      \
    {                                                                \
        x = type_(static_cast<float>(x) / static_cast<float>(y));    \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator++(type_& x)                                      \
    {                                                                \
        x = type_(static_cast<float>(x) + 1.f);                      \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_& operator--(type_& x)                                      \
    {                                                                \
        x = type_(static_cast<float>(x) - 1.f);                      \
        return x;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator++(type_& x, int)                                  \
    {                                                                \
        type_ y(x);                                                  \
        x = type_(static_cast<float>(x) + 1.f);                      \
        return y;                                                    \
    }                                                                \
    CK_TILE_HOST_DEVICE                                              \
    type_ operator--(type_& x, int)                                  \
    {                                                                \
        type_ y(x);                                                  \
        x = type_(static_cast<float>(x) - 1.f);                      \
        return y;                                                    \
    }
