// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename Type>
CK_TILE_HOST void reference_batched_transpose(const HostTensor<Type>& x,
                                              HostTensor<Type>& y,
                                              std::string layout = "NCHW")
{
    // index_t rank = x.get_num_of_dimension();
    // assert(rank == y.get_num_of_dimension());

    const int N = x.mDesc.get_lengths()[0];
    const int C = x.mDesc.get_lengths()[1];
    const int H = x.mDesc.get_lengths()[2];
    const int W = x.mDesc.get_lengths()[3];

    auto f = [&](auto batch) {
        if(layout == "NCHW")
        {
            for(int h = 0; h < H; ++h)
            {
                for(int w = 0; w < W; ++w)
                {
                    for(int c = 0; c < C; ++c)
                    {
                        Type v_x          = x(batch, c, h, w);
                        y(batch, h, w, c) = v_x;
                    }
                }
            }
        }
        else if(layout == "NHWC")
        {
            for(int c = 0; c < C; ++c)
            {
                for(int h = 0; h < H; ++h)
                {
                    for(int w = 0; w < W; ++w)
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
} // namespace ck_tile
