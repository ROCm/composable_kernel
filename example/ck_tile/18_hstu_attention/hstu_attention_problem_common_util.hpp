// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include <ck_tile/core.hpp>

namespace ck_tile {

namespace detail {

template <typename DataType, index_t ElemPerThread>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxVectorSize()
{
    if constexpr(std::is_same_v<DataType, half_t> || std::is_same_v<DataType, bf16_t>)
    {
        // ToDo: need support in ck_tile for using buffer_load_dwordx3
        // if constexpr(ElemPerThread % 6 == 0)
        //    return 6;
        if constexpr(ElemPerThread % 8 == 0)
            return 8;
        else if constexpr(ElemPerThread % 4 == 0)
            return 4;
        else if constexpr(ElemPerThread % 2 == 0)
            return 2;
        return 1;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        // ToDo: need support in ck_tile for using buffer_load_dwordx3
        // if constexpr(ElemPerThread % 3 == 0)
        //    return 3;
        if constexpr(ElemPerThread % 4 == 0)
            return 4;
        else if constexpr(ElemPerThread % 2 == 0)
            return 2;
        return 1;
    }
    else
        static_assert(false, "The data type is not supported!");
};

template <typename DataType,
          index_t kThreadBlockSize,
          index_t kHigherDimSize,
          index_t kLowerDimSize>
CK_TILE_HOST_DEVICE static constexpr auto GetDramTileAccessMaxVectorSize()
{
    constexpr index_t ElemPerThread = (kHigherDimSize * kLowerDimSize) / kThreadBlockSize;

    return GetMaxVectorSize<DataType, ElemPerThread>();
}

}; // namespace detail

} // namespace ck_tile
