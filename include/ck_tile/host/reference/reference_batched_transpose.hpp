// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename Type>
CK_TILE_HOST void reference_transpose_2d(const HostTensor<Type>& x, HostTensor<Type>& y)
{
    const int W = x.mDesc.get_lengths()[0];

    auto f = [&](auto w) {
        const int H = x.mDesc.get_lengths()[1];
        for(int h = 0; h < H; ++h)
        {
            Type v_x = x(w, h);
            y(h, w)  = v_x;
        }
    };

    make_ParallelTensorFunctor(f, W)(std::thread::hardware_concurrency());
}

template <typename Type>
CK_TILE_HOST void reference_transpose_4d(const HostTensor<Type>& x,
                                         HostTensor<Type>& y,
                                         std::string layout_in  = "NCHW",
                                         std::string layout_out = "NHWC")
{
    const int N = x.mDesc.get_lengths()[0];

    auto f = [&](auto batch) {
        if(layout_in == "NCHW" && layout_out == "NHWC")
        {
            const int C = x.mDesc.get_lengths()[1];
            const int H = x.mDesc.get_lengths()[2];
            const int W = x.mDesc.get_lengths()[3];
            for(int c = 0; c < C; ++c)
            {
                for(int h = 0; h < H; ++h)
                {
                    for(int w = 0; w < W; ++w)
                    {
                        Type v_x          = x(batch, c, h, w);
                        y(batch, h, w, c) = v_x;
                    }
                }
            }
        }
        else if(layout_in == "NHWC" && layout_out == "NCHW")
        {
            const int H = x.mDesc.get_lengths()[1];
            const int W = x.mDesc.get_lengths()[2];
            const int C = x.mDesc.get_lengths()[3];
            for(int h = 0; h < H; ++h)
            {
                for(int w = 0; w < W; ++w)
                {
                    for(int c = 0; c < C; ++c)
                    {
                        Type v_x          = x(batch, h, w, c);
                        y(batch, c, h, w) = v_x;
                    }
                }
            }
        }
    };

    make_ParallelTensorFunctor(f, N)(std::thread::hardware_concurrency());
}
// different threshold for different dtype
template <typename DataType>
inline auto batched_transpose_get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
inline auto batched_transpose_get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
inline auto batched_transpose_get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

template <typename Type>
inline bool check_ref(const HostTensor<Type>& x, const HostTensor<Type>& y)
{
    auto [rtol, atol] = ck_tile::batched_transpose_get_elimit<Type>("");
    return ck_tile::check_err(x, y, std::string("y Error: Incorrect results!"), rtol, atol);
}

} // namespace ck_tile
