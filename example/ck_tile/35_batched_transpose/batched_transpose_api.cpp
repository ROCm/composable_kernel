// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "batched_transpose_example.hpp"
#include <iostream>

void print_kparam_values(const transpose_kernel_param_t& kparam)
{
    std::cout << "block_tile_x: " << kparam.block_tile_x << std::endl;
    std::cout << "block_tile_y: " << kparam.block_tile_y << std::endl;
    std::cout << "warp_tile_x: " << kparam.warp_tile_x << std::endl;
    std::cout << "warp_tile_y: " << kparam.warp_tile_y << std::endl;
    std::cout << "thread_tile_x: " << kparam.thread_tile_x << std::endl;
    std::cout << "thread_tile_y: " << kparam.thread_tile_y << std::endl;
}

void populate_values(const transpose_kernel_param_t& kparam, batched_transpose_kargs& a)
{
    uint32_t dim_block_h = (a.height + kparam.block_tile_y - 1) / kparam.block_tile_y;
    uint32_t dim_block_w = (a.width + kparam.block_tile_x - 1) / kparam.block_tile_x;
    uint32_t dim_stride  = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = dim_block_h;
    a.dim_block_w = dim_block_w;
}

template <typename ts_type>
float batched_transpose_dispatch(const transpose_kernel_param_t& kparam,
                                 batched_transpose_kargs& a,
                                 ck_tile::stream_config& s)
{
    print_kparam_values(kparam);
    populate_values(kparam, a);

    ck_tile::index_t block_tile_x = kparam.block_tile_x;
    ck_tile::index_t block_tile_y = kparam.block_tile_y;

    ck_tile::index_t warp_tile_x = kparam.warp_tile_x;
    ck_tile::index_t warp_tile_y = kparam.warp_tile_y;

    ck_tile::index_t thread_tile_x = kparam.thread_tile_x;
    ck_tile::index_t thread_tile_y = kparam.thread_tile_y;

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "block_tile_x: " << block_tile_x << std::endl;
    std::cout << "block_tile_y: " << block_tile_y << std::endl;
    std::cout << "warp_tile_x: " << warp_tile_x << std::endl;
    std::cout << "warp_tile_y: " << warp_tile_y << std::endl;
    std::cout << "thread_tile_x: " << thread_tile_x << std::endl;
    std::cout << "thread_tile_y: " << thread_tile_y << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    using block_tile  = ck_tile::sequence<16, 16>;
    using warp_tile   = ck_tile::sequence<8, 8>;
    using thread_tile = ck_tile::sequence<1, 1>;

    using ts_problem =
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_tile, thread_tile>;
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

float batched_transpose(batched_transpose_trait t,
                        transpose_kernel_param_t kparam,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.type == "fp16")
    {
        return batched_transpose_dispatch<ck_tile::fp16_t>(kparam, a, s);
    }
    else if(t.type == "bf16")
    {
        return batched_transpose_dispatch<ck_tile::bf16_t>(kparam, a, s);
    }
    else if(t.type == "fp32")
    {
        return batched_transpose_dispatch<ck_tile::fp32_t>(kparam, a, s);
    }
    else if(t.type == "int8")
    {
        return batched_transpose_dispatch<ck_tile::int8_t>(kparam, a, s);
    }
    return -1;
}
