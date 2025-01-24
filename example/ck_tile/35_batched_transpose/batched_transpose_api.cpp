// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "batched_transpose_example.hpp"
#include <iostream>

#define BATCHED_TRANSPOSE_DISPATCH()                                                   \
    print_kparam_values(kparam);                                                       \
    populate_values(kparam, a);                                                        \
    using block_tile  = ck_tile::sequence<16, 16>;                                     \
    using warp_tile   = ck_tile::sequence<8, 8>;                                       \
    using thread_tile = ck_tile::sequence<1, 1>;                                       \
    using ts_problem =                                                                 \
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_tile, thread_tile>; \
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;                 \
                                                                                       \
    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;                       \
                                                                                       \
    auto kargs = kernel::MakeKargs(a);                                                 \
                                                                                       \
    const dim3 grids      = kernel::GridSize(a);                                       \
    constexpr dim3 blocks = kernel::BlockSize();                                       \
                                                                                       \
    float ave_time = ck_tile::launch_kernel(                                           \
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));      \
                                                                                       \
    return ave_time;

void print_kparam_values(const transpose_kernel_param_t& kparam)
{
    std::cout << "tile_x: " << kparam.tile_x << std::endl;
    std::cout << "tile_y: " << kparam.tile_y << std::endl;
    std::cout << "pack_x: " << kparam.pack_x << std::endl;
    std::cout << "pack_y: " << kparam.pack_y << std::endl;
    std::cout << "ediv_x: " << kparam.ediv_x << std::endl;
    std::cout << "ediv_y: " << kparam.ediv_y << std::endl;
}

void populate_values(const transpose_kernel_param_t& kparam, batched_transpose_kargs& a)
{
    uint32_t dim_block_h = (a.height + kparam.tile_y - 1) / kparam.tile_y;
    uint32_t dim_block_w = (a.width + kparam.tile_x - 1) / kparam.tile_x;
    uint32_t dim_stride  = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = dim_block_h;
    a.dim_block_w = dim_block_w;
}

float batched_transpose(batched_transpose_trait t,
                        transpose_kernel_param_t kparam,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{

    if(t.type == "fp16")
    {
        using ts_type = ck_tile::fp16_t;
        BATCHED_TRANSPOSE_DISPATCH()
    }
    else if(t.type == "bf16")
    {
        using ts_type = ck_tile::bf16_t;
        BATCHED_TRANSPOSE_DISPATCH()
    }
    else if(t.type == "fp32")
    {
        using ts_type = ck_tile::fp32_t;
        BATCHED_TRANSPOSE_DISPATCH()
    }
    else if(t.type == "int8")
    {
        using ts_type = ck_tile::int8_t;
        BATCHED_TRANSPOSE_DISPATCH()
    }
    return -1;
}
