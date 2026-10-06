#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile::detail {

template <typename DataType, index_t RowElements, index_t PadElements>
CK_TILE_HOST_DEVICE constexpr auto make_fmha_bwd_tdm_padding_config()
{
    constexpr index_t dword_bytes = 4;
    constexpr index_t pad_dwords  = PadElements * sizeof(DataType) / dword_bytes;
    constexpr index_t row_dwords  = RowElements * sizeof(DataType) / dword_bytes;
    static_assert(PadElements >= 0 && RowElements > 0);
    static_assert(pad_dwords * dword_bytes == PadElements * sizeof(DataType),
                  "LDS pad must be a whole number of dwords");
    static_assert(row_dwords * dword_bytes == RowElements * sizeof(DataType),
                  "LDS row must be a whole number of dwords");
    static_assert(pad_dwords <= 128, "pad_amount must fit its 7-bit biased field");

    if constexpr(pad_dwords == 0)
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }
    else
    {
        static_assert(row_dwords <= 256, "pad_interval must fit its 3-bit biased field");
        static_assert(row_dwords >= 2 && (row_dwords & (row_dwords - 1)) == 0,
                      "pad_interval requires a power-of-two row of at least two dwords");
        constexpr index_t row_log2 = [] {
            index_t value  = row_dwords;
            index_t result = 0;
            while(value > 1)
            {
                value >>= 1;
                ++result;
            }
            return result;
        }();
        return make_tuple(number<true>{}, number<pad_dwords - 1>{}, number<row_log2 - 1>{});
    }
}

} // namespace ck_tile::detail
