// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/ops/batched_transpose.hpp"

#include <vector>
#include <string>

#pragma once

struct batched_transpose_2d_kargs : public ck_tile::BatchedTransposeHostArgs
{
    batched_transpose_2d_kargs() { batch = 1; }
};

template <typename ts_type,
          ck_tile::index_t block_w,
          ck_tile::index_t block_h,
          ck_tile::index_t warp_w,
          ck_tile::index_t warp_h,
          ck_tile::index_t thread_w,
          ck_tile::index_t thread_h>
float batched_transpose_dispatch(batched_transpose_2d_kargs& a, ck_tile::stream_config& s)
{
    uint32_t dim_stride = a.height * a.width;
    a.dim_stride        = dim_stride;
    a.dim_block_h       = static_cast<int>(block_h);
    a.dim_block_w       = static_cast<int>(block_w);

    using block_tile  = ck_tile::sequence<block_w, block_h>;
    using warp_tile   = ck_tile::sequence<warp_w, warp_h>;
    using thread_tile = ck_tile::sequence<thread_w, thread_h>;

    using ts_problem =
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_tile, thread_tile>;
    using ts_pipeline = ck_tile::BatchedTransposePipeline<ts_problem>;

    using kernel = ck_tile::BatchedTransposeKernel<ts_pipeline>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids      = kernel::GridSize(a);
    constexpr dim3 blocks = kernel::BlockSize();

    std::cout << "Running transpose with"
              << " Width=" << a.width << ", Height=" << a.height << ", block_w=" << block_w
              << ", block_h=" << block_h << ", warp_w=" << warp_w << ", warp_h=" << warp_h
              << ", thread_w=" << thread_w << ", thread_h=" << thread_h << std::endl;

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

float batched_transpose_2d(batched_transpose_2d_kargs a, ck_tile::stream_config s, int opt = 1)
{
    using TransposeFunc = float (*)(batched_transpose_2d_kargs&, ck_tile::stream_config&);

    constexpr std::array<TransposeFunc, 2> transpose_table = {
        batched_transpose_dispatch<ck_tile::int8_t, 16, 16, 8, 8, 1, 1>,
        batched_transpose_dispatch<ck_tile::int8_t, 16, 16, 8, 8, 1, 1>};

    return transpose_table.at(opt)(a, s);
}
