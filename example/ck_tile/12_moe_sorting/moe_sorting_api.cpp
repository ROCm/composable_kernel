// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_sorting_api.hpp"

float moe_sorting(moe_sorting_trait t, moe_sorting_kargs a, ck_tile::stream_config s)
{
    if(t.weight_type == "fp32")
    {
        using index_t         = ck_tile::index_t;
        using ms_weight_type  = float;
        using ms_problem      = ck_tile::MoeSortingProblem<index_t, ms_weight_type>;
        using ms_pipeline     = ck_tile::MoeSortingPipeline<ms_problem>;
        using kernel          = ck_tile::MoeSortingKernel<ms_pipeline>;
        auto kargs            = kernel::MakeKargs(a);
        const dim3 grids      = 1;
        const dim3 blocks     = ck_tile::max(t.experts, ck_tile::get_warp_size());
        const size_t lds_size = ((blocks.x + 1) * t.experts + (t.experts + 1)) * sizeof(index_t);
        float ave_time        = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<64, 1>(kernel{}, grids, blocks, lds_size, kargs));
        return ave_time;
    }
    return -1;
}
