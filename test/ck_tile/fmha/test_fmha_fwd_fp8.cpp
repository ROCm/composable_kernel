// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#include <tuple>
#include <string>

using ::testing::Values;

using DataTypeConfig = FmhaFwdFp8;

// Currently there are no fp8 instances for splitkv, pagedkv by default (the tests pass if such
// instances are added), however the corresponding tests are not disabled (they will be skipped)
// in case such instances will be added in the future.

const auto HDimValues = Values(std::tuple{64, -1}, std::tuple{128, -1});

const auto SplitKVHDimValues = Values(std::tuple{64, -1}, std::tuple{128, -1});

const auto AppendKVHDimValues = Values(std::tuple{64, -1}, std::tuple{128, -1});

// There are no fp8 instances with seqlen padding (mode_enum::group requires it)
const auto ModeValues = Values(mode_enum::batch);

const auto IsVRowmajorValues = Values(false);

const auto squant             = true;
const std::string init_method = "uf";
const bool def_lse            = false;
const bool def_is_v_rowmajor  = true;

int adjust_seqlen(int seqlen)
{
    // There are no fp8 instances with padding, pad seqlen to avoid skipping most of the tests
    return ck_tile::integer_least_multiple(seqlen, 128);
}

#include "test_fmha_fwd.inc"
