// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "transpose_example.hpp"
#include <iostream>

template <typename ts_type,
          ck_tile::index_t block_x,
          ck_tile::index_t block_y,
          ck_tile::index_t warp_x,
          ck_tile::index_t warp_y>
float batched_transpose_dispatch(batched_transpose_kargs& a, ck_tile::stream_config& s)
{
    uint32_t dim_block_h = (a.height + block_y - 1) / block_y;
    uint32_t dim_block_w = (a.width + block_x - 1) / block_x;
    uint32_t dim_stride  = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = dim_block_h;
    a.dim_block_w = dim_block_w;

    using ts_problem  = ck_tile::TransposePipelineProblem<ts_type,
                                                         ck_tile::tensor_layout::gemm::RowMajor,
                                                         64,
                                                         1,
                                                         1,
                                                         block_y,
                                                         block_x,
                                                         warp_y,
                                                         warp_x>;
    using ts_pipeline = ck_tile::BlockTranspose<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.type == "fp16")
    {
        return batched_transpose_dispatch<ck_tile::fp16_t, 16, 32, 16, 32>(a, s);
    }
    else if(t.type == "fp8")
    {
        return batched_transpose_dispatch<ck_tile::fp8_t, 16, 64, 16, 64>(a, s);
    }

    return -1;
}
