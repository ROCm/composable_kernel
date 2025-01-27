// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/ops/batched_transpose.hpp"

#include <vector>
#include <string>

#pragma once

struct batched_transpose_trait
{
    std::string type;
    std::string layout;
};

struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
{
};

struct transpose_kernel_param_t
{
    int block_tile_x;
    int block_tile_y;
    int warp_tile_x;
    int warp_tile_y;
    int thread_tile_x;
    int thread_tile_y;
};

template <size_t type_size>
struct transpose_kernel_get_all_param_t
{
};

template <>
struct transpose_kernel_get_all_param_t<4>
{
    static std::vector<transpose_kernel_param_t> get()
    {
        std::vector<transpose_kernel_param_t> the_list{
            {16, 16, 8, 8, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 8, 8, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},
        };
        return the_list;
    }
};

template <>
struct transpose_kernel_get_all_param_t<2>
{
    static std::vector<transpose_kernel_param_t> get()
    {
        std::vector<transpose_kernel_param_t> the_list{
            {16, 16, 8, 8, 1, 1}, {32, 16, 1, 1, 1, 1}, {16, 32, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},  {64, 4, 1, 1, 1, 1},  {4, 128, 1, 1, 1, 1}, {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1}, {256, 4, 1, 1, 1, 1},

            {32, 32, 2, 2, 2, 2}, {32, 32, 2, 2, 1, 2}, {32, 32, 2, 2, 2, 1}, {32, 32, 2, 2, 1, 1},

            {64, 32, 4, 2, 4, 2}, {64, 32, 4, 2, 2, 2}, {64, 32, 4, 2, 2, 1},

            {32, 64, 2, 4, 2, 4}, {32, 64, 2, 4, 2, 2}, {32, 64, 2, 4, 1, 2},

            {16, 64, 1, 4, 1, 2}, {64, 16, 4, 1, 2, 1},

            {64, 64, 4, 4, 4, 4}, {64, 64, 4, 4, 2, 2},
        };
        return the_list;
    }
};

template <>
struct transpose_kernel_get_all_param_t<1>
{
    static std::vector<transpose_kernel_param_t> get()
    {
        std::vector<transpose_kernel_param_t> the_list{
            {16, 16, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},
        };
        return the_list;
    }
};

float batched_transpose(batched_transpose_trait t,
                        transpose_kernel_param_t kparam,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s);
