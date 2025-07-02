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
    uint32_t dim_stride = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = block_y;
    a.dim_block_w = block_x;

    printf("batched_transpose_kargs: {dim_stride=%d dim_block_h=%d dim_block_w=%d}\n",
           a.dim_stride,
           a.dim_block_h,
           a.dim_block_w);

    using ts_problem =
        ck_tile::TransposePipelineProblem<ts_type,                                // dtype
                                          ck_tile::tensor_layout::gemm::RowMajor, // layout
                                          64,                                     // blocksize
                                          1,                                      // row warps
                                          1,                                      // col warps
                                          block_y,                                // row per block
                                          block_x,                                // col per block
                                          warp_y,                                 // row per xdl
                                          warp_x>;                                // col per xdl
    using ts_pipeline = ck_tile::BlockTranspose<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    printf("Grid: x=%u y=%u z=%u\n", grids.x, grids.y, grids.z);
    printf("Block: x=%u y=%u z=%u\n", blocks.x, blocks.y, blocks.z);
    printf(
        "Host args: batch=%d, height=%d, width=%d, dim_stride=%d, dim_block_h=%d, dim_block_w=%d\n",
        a.batch,
        a.height,
        a.width,
        a.dim_stride,
        a.dim_block_h,
        a.dim_block_w);
    printf("kargs: kargs.batch=%d kargs.height=%d kargs.width=%d kargs.dim_stride=%d\n",
           kargs.batch,
           kargs.height,
           kargs.width,
           kargs.dim_stride);

    printf("Launching Kernel...\n");

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
        return batched_transpose_dispatch<ck_tile::fp16_t, 64, 64, 64, 64>(a, s);
    }
    else if(t.type == "fp8")
    {
        return batched_transpose_dispatch<ck_tile::fp8_t, 64, 64, 64, 64>(a, s);
    }

    return -1;
}
