// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/moe_sorting.hpp"

struct moe_sorting_trait
{
    std::string input_type;
    std::string weight_type; // currently always float
    int experts;
    int topk;
    int unit_size;
    int tokens;
};

struct moe_sorting_kargs : public ck_tile::MoeSortingHostArgs
{
};

float moe_sorting(moe_sorting_trait t, moe_sorting_kargs a, ck_tile::stream_config s);
