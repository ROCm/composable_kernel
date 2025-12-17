// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

namespace ck_tile {
template <index_t pk_size>
struct pk_f6_t
{
    static constexpr index_t num_bits_elem = 6;
    using element_type                     = uint32_t; // element storage fundamental type
    static constexpr index_t packed_size   = pk_size;
    static constexpr index_t num_bits_vec_elem =
        sizeof(element_type) * 8; // 32-bit uint for storage
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr index_t vector_size = (packed_size * num_bits_elem) / num_bits_vec_elem;
    // using storage_type = element_type __attribute__((ext_vector_type(vector_size)));
    // storage_type data_{storage_type(0)}; // packed data
    element_type data_[3]; // packed data
    using type = pk_f6_t<packed_size>;
    void pack(const uint32_t x, const index_t i)
    {
        uint32_t bits        = static_cast<uint32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        uint32_t old_value   = data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            uint32_t next_value = data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data_[arr_index + 1] = next_value;
        }
    }

    template <typename type>
    static inline uint32_t unpack(const type& pk, const index_t i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        uint32_t bits = pk.data_[arr_idx] >> bit_offset;
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data_[arr_idx + 1] & ((1u << overhang) - 1)) << (num_bits_elem - overhang);
        }

        return bits & 0x3F;
    }

    inline uint32_t unpack(const index_t i) const { return unpack(*this, i); }

    float fp6_e2m3_to_float(uint32_t fp6_bits)
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
            result               = std::pow(2, exponent - 1);
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

using f6x16_pk_t = pk_f6_t<16>;
template <>
struct numeric_traits<f6x16_pk_t>
{
    static constexpr int PackedSize = 16;
};
} // namespace ck_tile
