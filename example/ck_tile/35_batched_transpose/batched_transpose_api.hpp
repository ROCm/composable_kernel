// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_transpose.hpp"
#include <string>

struct batched_transpose_trait
{
    std::string type;
    std::string layout;
};

struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
{
};

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s);
