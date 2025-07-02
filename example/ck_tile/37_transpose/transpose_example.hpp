// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "batched_transpose_kernel.hpp"
#include "block_transpose.hpp"
#include "transpose_policy.hpp"

#include <vector>
#include <string>

#pragma once

template <typename DataType>
struct batched_transpose_config_item
{
    ck_tile::index_t block_x;
    ck_tile::index_t block_y;
    ck_tile::index_t warp_x;
    ck_tile::index_t warp_y;
    std::string to_string() const
    {
        return "block_x: " + std::to_string(block_x) + ", block_y: " + std::to_string(block_y) +
               ", warp_x: " + std::to_string(warp_x) + ", warp_y: " + std::to_string(warp_y);
    }
};
template <typename DataType>
inline static constexpr std::array<batched_transpose_config_item<DataType>, 0>
    batched_transpose_config_list{};
template <>
inline constexpr std::array batched_transpose_config_list<ck_tile::fp16_t>{
    batched_transpose_config_item<ck_tile::fp16_t>{16, 32, 16, 32},
    batched_transpose_config_item<ck_tile::fp16_t>{32, 16, 32, 16},
};
template <>
inline constexpr std::array batched_transpose_config_list<ck_tile::fp8_t>{
    batched_transpose_config_item<ck_tile::fp8_t>{16, 64, 16, 64},
    batched_transpose_config_item<ck_tile::fp8_t>{32, 32, 32, 32},
};
template <typename DataType, ck_tile::index_t i>
struct batched_transpose_config
{
    static constexpr auto item    = batched_transpose_config_list<DataType>[i];
    using dtype                   = DataType;
    static constexpr auto block_x = item.block_x;
    static constexpr auto block_y = item.block_y;
    static constexpr auto warp_x  = item.warp_x;
    static constexpr auto warp_y  = item.warp_y;
    static std::string to_string() { return item.to_string(); }
};

struct batched_transpose_trait
{
    std::string type;
    std::string layout;
    ck_tile::index_t config; // Index of the configuration in the list
    bool kname;
};

struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
{
};

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s);
