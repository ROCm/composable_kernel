// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

namespace ck_tile {
template <index_t pk_size>
struct pk_fp6_t
{
    static constexpr index_t num_bits_elem = 6;
    using element_type                     = int32_t; // element storage fundamental type
    static constexpr index_t packed_size   = pk_size;
    static constexpr index_t num_bits_vec_elem =
        sizeof(element_type) * 8; // 32-bit uint for storage
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr index_t vector_size = (packed_size * num_bits_elem) / num_bits_vec_elem;
    element_type data_[vector_size]; // packed data
    using type = pk_fp6_t<packed_size>;
    CK_TILE_HOST_DEVICE constexpr explicit pk_fp6_t(int value = 0)
    {
        for(size_t i = 0; i < vector_size; ++i)
        {
            data_[i] = value;
        }
    }
    CK_TILE_HOST_DEVICE void pack(const int32_t x, const index_t i)
    {
        int32_t bits         = static_cast<int32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        int32_t old_value    = data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            int32_t next_value = data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data_[arr_index + 1] = next_value;
        }
    }

    template <typename T>
    CK_TILE_HOST_DEVICE static int32_t unpack(const T& pk, const index_t i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        int32_t bits = pk.data_[arr_idx] >> bit_offset;
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data_[arr_idx + 1] & ((1u << overhang) - 1)) << (num_bits_elem - overhang);
        }

        return bits & 0x3F;
    }

    CK_TILE_HOST_DEVICE int32_t unpack(const index_t i) const { return unpack(*this, i); }

    CK_TILE_HOST_DEVICE int32_t operator[](index_t i) const { return data_[i]; }

    CK_TILE_HOST_DEVICE static float fp6_e2m3_to_float(int32_t fp6_bits)
    {
        fp6_bits = fp6_bits & 0x3F;

        uint32_t sign     = (fp6_bits >> 5) & 0x1; // bit 5
        uint32_t exponent = (fp6_bits >> 3) & 0x3; // bits 4-3
        uint32_t mantissa = fp6_bits & 0x7;        // bits 2-0

        float result;
        if(exponent == 0 && mantissa == 0)
        {
            result = 0.f;
        }
        else if(exponent != 0)
        {
            result               = std::exp2f(static_cast<int>(exponent) - 1);
            float mantissa_value = 1.0f + mantissa / 8.0f;
            result *= mantissa_value;
        }
        else
        {
            result = mantissa / 8.0f;
        }
        return sign == 1 ? -1 * result : result;
    }
};

using pk_fp6x16_t = pk_fp6_t<16>;
using pk_fp6x32_t = pk_fp6_t<32>;
template <>
struct numeric_traits<pk_fp6x16_t>
{
    static constexpr int PackedSize = 16;
};
} // namespace ck_tile
