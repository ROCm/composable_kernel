// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float fused_moegemm_(const ck_tile::stream_config& s, fused_moegemm_args a);

float fused_moegemm(fused_moegemm_traits t, fused_moegemm_args a, const ck_tile::stream_config& s)
{
    template <ck_tile::index_t... Is>
    using S = ck_tile::sequence<Is...>;
    float r = -1;
    if(t.prec_i == "bf16" && t.prec_w == "bf16" && t.prec_o == "bf16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && block_m == 32 &&
       gate_only == 1)
    {
        using t_ = fmoe_<ck_tile::bf16_t,
                         ck_tile::bf16_t,
                         ck_tile::bf16_t,
                         float,
                         float,
                         float,
                         float,
                         S<32, 512, 128, 128>,
                         S<4, 1, 1>,
                         S<32, 32, 16>,
                         1,
                         0>;
        fused_moegemm_<t_>(s, a);
    }
    return r;
}
