// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd_runner.hpp"

#include "gtest/gtest.h"

using DataTypeConfig = FmhaBwdFp16;

using ::testing::Values;
using ::testing::ValuesIn;

const auto HDimValues =
    Values(std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1});

const auto ModeValues = Values(mode_enum::batch, mode_enum::group);

constexpr std::string init_method = "uf";

#include "test_fmha_bwd.inc"
