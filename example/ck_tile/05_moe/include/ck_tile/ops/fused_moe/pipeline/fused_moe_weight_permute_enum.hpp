// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {
enum class FusedMoeWeightPermuteEnum
{
    // permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    // permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
    permute_b_nr_kr_kw_nw_kv    = 2, // 0,1,3,4,2,5
    permute_b_nr_kr_waveflatten = permute_b_nr_kr_kw_nw_kv,
    no_permute                  = 999,
};
}
