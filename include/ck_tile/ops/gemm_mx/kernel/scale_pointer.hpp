// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"


namespace ck_tile {

template <int SharedGranularityMN, int SharedGranularityK = 0>
struct MXScalePointer
{
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = SharedGranularityK;

    const float* ptr;

    CK_TILE_HOST_DEVICE MXScalePointer() = default;
    CK_TILE_HOST_DEVICE MXScalePointer(const float* ptr_) : ptr(ptr_) {}
    CK_TILE_HOST_DEVICE MXScalePointer(const float* ptr_, [[maybe_unused]] index_t length_)
        : ptr(ptr_)
    {
    }

    CK_TILE_HOST_DEVICE MXScalePointer operator+(index_t offset) const
    {
        MXScalePointer ret;
        if constexpr(GranularityMN == 0)
        {
            ret.ptr = ptr + offset / GranularityK;
        }
        else
        {
            ret.ptr = ptr + offset / GranularityMN / GranularityK;
        }
        return ret;
    }

    CK_TILE_HOST_DEVICE float operator[](index_t i) const = delete;
};

template <int SharedGranularityMN>
struct MXScalePointer<SharedGranularityMN, 0>
{
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = 0;

    static_assert(GranularityMN != 0);

    const float* ptr;
    index_t length;

    CK_TILE_HOST_DEVICE MXScalePointer() = default;
    CK_TILE_HOST_DEVICE MXScalePointer(const float* ptr_) : ptr(ptr_), length(1) {}
    CK_TILE_HOST_DEVICE MXScalePointer(const float* ptr_, index_t length_)
        : ptr(ptr_), length(length_)
    {
    }

    CK_TILE_HOST_DEVICE MXScalePointer operator+(index_t offset) const
    {
        MXScalePointer ret;
        if constexpr(GranularityMN == 1)
        {
            ret.ptr    = ptr + offset;
            ret.length = length - offset;
        }
        else
        {
            ret.ptr    = ptr + offset / GranularityMN;
            ret.length = length - offset / GranularityMN;
        }
        return ret;
    }

    CK_TILE_HOST_DEVICE float operator[](index_t i) const
    {
        // with additional oob check
        if constexpr(GranularityMN == 1)
            return i < length ? ptr[i] : 0;
        else
            return i / GranularityMN < length ? ptr[i / GranularityMN] : 0;
    }
};

// shared granularityMN = -1 means no scale
template <>
struct MXScalePointer<-1, 0>
{
    static constexpr int GranularityMN = -1;
    static constexpr int GranularityK  = 0;

    const float* ptr = nullptr;

    CK_TILE_HOST_DEVICE constexpr MXScalePointer() = default;
    CK_TILE_HOST_DEVICE constexpr MXScalePointer(const float*) {}
    CK_TILE_HOST_DEVICE constexpr MXScalePointer(const float*, index_t) {}

    CK_TILE_HOST_DEVICE constexpr MXScalePointer operator+(index_t) const
    {
        return MXScalePointer{};
    }
    CK_TILE_HOST_DEVICE constexpr float operator[](index_t) const
    {
        return 1; // alway return 1, it doesn't change the result
    }
};

} // namespace ck_tile