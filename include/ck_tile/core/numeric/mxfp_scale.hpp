// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include <cstdint>
#include "ck_tile/core/numeric/e8m0.hpp"

namespace ck_tile {

template <typename ScaleType>
struct Packed4Scale
{
    using scale_type     = ScaleType;
    using raw_type       = uint32_t;
    using raw_scale_type = typename ScaleType::raw_type;

    static constexpr int num_pack = 4;
    union
    {
        raw_type data_;
        raw_scale_type scales_[num_pack]; // Direct byte/element access
    };

    // Constructors
    CK_TILE_HOST_DEVICE constexpr Packed4Scale() = default;
    CK_TILE_HOST_DEVICE constexpr Packed4Scale(raw_type val) : data_(val) {}
    CK_TILE_HOST_DEVICE constexpr Packed4Scale(float s0, float s1, float s2, float s3)
    {
        set_scales_from_float(s0, s1, s2, s3); // set_scales_from_float will assign data_
    }

    CK_TILE_HOST_DEVICE constexpr Packed4Scale(ScaleType s0,
                                               ScaleType s1,
                                               ScaleType s2,
                                               ScaleType s3)
    {
        set_scales(s0, s1, s2, s3);
    }

    /**
     * @brief Set 4 scales from float values
     */
    CK_TILE_HOST_DEVICE constexpr void set_scales_from_float(float s0, float s1, float s2, float s3)
    {
        set_scales(ScaleType(s0), ScaleType(s1), ScaleType(s2), ScaleType(s3));
    }

    /**
     * @brief Set 4 scales from scale_type values
     */
    CK_TILE_HOST_DEVICE constexpr void
    set_scales(ScaleType s0, ScaleType s1, ScaleType s2, ScaleType s3)
    {
        data_ = 0;
        pack_scale(s0, 3);
        pack_scale(s1, 2);
        pack_scale(s2, 1);
        pack_scale(s3, 0);
    }

    CK_TILE_HOST_DEVICE constexpr operator raw_type() const { return data_; }
    CK_TILE_HOST_DEVICE constexpr raw_type& data() [[clang::lifetimebound]] { return data_; }
    CK_TILE_HOST_DEVICE constexpr raw_type data() const { return data_; }

    /**
     * @brief Extract the ith scale and convert to float
     * @param i Scale index (0-3)
     */
    CK_TILE_HOST_DEVICE constexpr float unpack_to_float(int i) const
    {
        return static_cast<float>(unpack_scale(i));
    }

    /**
     * @brief Extract the ith scale as scale_type
     * @param i Scale index (0-3)
     */
    CK_TILE_HOST_DEVICE constexpr ScaleType unpack_scale(int i) const
    {
        return ScaleType(scales_[i]);
    }

    /**
     * @brief Pack a float scale value into the ith position
     * @param scale Scale value as float
     * @param i Position index (0-3)
     */
    CK_TILE_HOST_DEVICE constexpr void pack_from_float(float scale, int i)
    {
        pack_scale(ScaleType(scale), i);
    }

    /**
     * @brief Pack a scale_type value into the ith position
     * @param scale Scale value
     * @param i Position index (0-3)
     */
    CK_TILE_HOST_DEVICE constexpr void pack_scale(ScaleType scale, int i)
    {
        scales_[i] = scale.get();
    }
};

// Type alias for e8m0_t scales
using Packed4Scale_E8M0 = Packed4Scale<e8m0_t>;

} // namespace ck_tile
