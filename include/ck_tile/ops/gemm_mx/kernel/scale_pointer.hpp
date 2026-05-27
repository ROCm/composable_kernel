// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile {

template <typename ScaleType, int SharedGranularityMN, int SharedGranularityK = 0>
struct MXScalePointer
{
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = SharedGranularityK;

    static_assert(GranularityK != 0,
                  "GranularityK cannot be zero in primary template; "
                  "use the partial specialization for GranularityK == 0");

    const ScaleType* ptr;

    CK_TILE_HOST_DEVICE MXScalePointer() = default;
    CK_TILE_HOST_DEVICE MXScalePointer(const ScaleType* ptr_) : ptr(ptr_) {}
    CK_TILE_HOST_DEVICE MXScalePointer(const ScaleType* ptr_, [[maybe_unused]] index_t length_)
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

    CK_TILE_HOST_DEVICE ScaleType operator[](index_t i) const = delete;
};

template <typename ScaleType, int SharedGranularityMN>
struct MXScalePointer<ScaleType, SharedGranularityMN, 0>
{
    static constexpr int GranularityMN = SharedGranularityMN;
    static constexpr int GranularityK  = 0;

    static_assert(GranularityMN != 0);

    const ScaleType* ptr;
    index_t length;

    CK_TILE_HOST_DEVICE MXScalePointer() = default;
    CK_TILE_HOST_DEVICE MXScalePointer(const ScaleType* ptr_) : ptr(ptr_), length(1) {}
    CK_TILE_HOST_DEVICE MXScalePointer(const ScaleType* ptr_, index_t length_)
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

    CK_TILE_HOST_DEVICE ScaleType operator[](index_t i) const
    {
        // with additional oob check
        if constexpr(GranularityMN == 1)
            return i < length ? ptr[i] : 0;
        else
            return i / GranularityMN < length ? ptr[i / GranularityMN] : 0;
    }
};

// shared granularityMN = -1 means no scale
template <typename ScaleType>
struct MXScalePointer<ScaleType, -1, 0>
{
    static constexpr int GranularityMN = -1;
    static constexpr int GranularityK  = 0;

    const ScaleType* ptr = nullptr;

    CK_TILE_HOST_DEVICE constexpr MXScalePointer() = default;
    CK_TILE_HOST_DEVICE constexpr MXScalePointer(const ScaleType*) {}
    CK_TILE_HOST_DEVICE constexpr MXScalePointer(const ScaleType*, index_t) {}

    CK_TILE_HOST_DEVICE constexpr MXScalePointer operator+(index_t) const
    {
        return MXScalePointer{};
    }
    CK_TILE_HOST_DEVICE constexpr ScaleType operator[](index_t) const
    {
        return 1; // alway return 1, it doesn't change the result
    }
};

} // namespace ck_tile
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
