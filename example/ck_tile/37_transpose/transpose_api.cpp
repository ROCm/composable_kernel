// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "transpose_example.hpp"
#include <iostream>

template <typename Config>
float batched_transpose_dispatch(batched_transpose_kargs& a, ck_tile::stream_config& s)
{
    using C              = Config;
    uint32_t dim_block_h = (a.height + C::block_y - 1) / C::block_y;
    uint32_t dim_block_w = (a.width + C::block_x - 1) / C::block_x;
    uint32_t dim_stride  = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = dim_block_h;
    a.dim_block_w = dim_block_w;

    using ts_problem  = ck_tile::TransposePipelineProblem<typename C::dtype,
                                                         ck_tile::tensor_layout::gemm::RowMajor,
                                                         64,
                                                         1,
                                                         1,
                                                         C::block_y,
                                                         C::block_x,
                                                         C::warp_y,
                                                         C::warp_x>;
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
    float res   = -1;
    using types = ck_tile::tuple<ck_tile::fp16_t, ck_tile::fp8_t>;
    std::vector<std::string> type_names{"fp16", "fp8"};

    ck_tile::static_for<0, types::size(), 1>{}([&](auto type_id) {
        if(type_names[type_id()] == t.type)
        {
            using Type                 = ck_tile::remove_cvref_t<decltype(types{}(type_id))>;
            constexpr auto num_configs = batched_transpose_config_list<Type>.size();
            ck_tile::static_for<0, num_configs, 1>{}([&](auto i) {
                if(t.config == i())
                {
                    using Config = batched_transpose_config<Type, i()>;
                    if(t.kname)
                        std::cout << "Running batched transpose with config: "
                                  << Config::to_string() << std::endl;
                    res = batched_transpose_dispatch<Config>(a, s);
                }
            });
        }
    });
    return res;
}
