// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <string>

#pragma once

struct transpose_kernel_param_t
{
    int tile_x;
    int tile_y;
    int pack_x;
    int pack_y;
    int ediv_x;
    int ediv_y;
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

template <>
struct transpose_kernel_get_all_param_t<2>
{
    static std::vector<transpose_kernel_param_t> get()
    {
        std::vector<transpose_kernel_param_t> the_list{
            {16, 16, 1, 1, 1, 1}, {32, 16, 1, 1, 1, 1}, {16, 32, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 1},

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

std::vector<transpose_kernel_param_t> get_transpose_all_kernel(std::string fp_str)
{
    if(fp_str == "fp32")
        return transpose_kernel_get_all_param_t<4>::get();
    else if(fp_str == "fp16" || fp_str == "bf16")
        return transpose_kernel_get_all_param_t<2>::get();
    else if(fp_str == "int8")
        return transpose_kernel_get_all_param_t<1>::get();
    else
        return {};
}

bool transpose_kernel_is_valid(uint32_t,
                               uint32_t height,
                               uint32_t width,
                               const transpose_kernel_param_t* kparam)
{
    return width % kparam->ediv_x == 0 && height % kparam->ediv_y == 0;
}

bool is_kernel_valid(uint32_t n,
                     uint32_t c,
                     uint32_t h,
                     uint32_t w,
                     const transpose_kernel_param_t* kparam,
                     std::string layout_in)
{
    if(layout_in == "nchw")
    {
        return transpose_kernel_is_valid(n, c, h * w, kparam);
    }
    else
    {
        return transpose_kernel_is_valid(n, h * w, c, kparam);
    }
}
