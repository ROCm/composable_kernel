// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise_unary.hpp"
#include <string>

struct elementwise_trait
{
    std::string input_type;  // input type
    std::string acc_type;    // type to do intermediate computation
    std::string output_type; // type to store out
    std::string op;
};

struct elementwise_kargs : public ck_tile::ElementwiseUnaryHostArgs
{
};

float elementwise(elementwise_trait t, elementwise_kargs a, ck_tile::stream_config s);
