// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_sorting_api.hpp"

float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s)
{
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        if(t.experts > 127)
        {
            printf("lds size exceed, only support experts <127 \n");
            return -1;
        }
        using index_t        = ck_tile::index_t;
        using ms_weight_type = float;
        using ms_problem     = ck_tile::MoeSortingProblem<index_t, ms_weight_type>;
        using kernel         = ck_tile::MoeSortingKernel<ms_problem>;
        auto kargs           = kernel::MakeKargs(a);
        const dim3 grids     = kernel::GridSize(a);
        const dim3 blocks    = kernel::BlockSize(a);
        float ave_time =
            ck_tile::launch_kernel(s, ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    }
    return -1;
}
