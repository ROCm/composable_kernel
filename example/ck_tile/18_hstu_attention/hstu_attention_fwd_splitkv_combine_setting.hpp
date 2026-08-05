// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_fwd_tile_setting_define.hpp"

template <ck_tile::index_t kOHeaddim>
struct HstuAttentionFwdSplitKVCombineTileSetting;

template <>
struct HstuAttentionFwdSplitKVCombineTileSetting<64>
{
    using Type = ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<32, 4, 64>;
};

template <>
struct HstuAttentionFwdSplitKVCombineTileSetting<96>
{
    using Type = ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<16, 4, 96>;
};

template <>
struct HstuAttentionFwdSplitKVCombineTileSetting<128>
{
    using Type = ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<16, 4, 128>;
};

template <>
struct HstuAttentionFwdSplitKVCombineTileSetting<256>
{
    using Type = ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<8, 4, 256>;
};
