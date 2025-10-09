// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#include <tuple>
#include <string>

using ::testing::Values;

using DataTypeConfig = FmhaFwdFp16;

const auto HDimValues = Values(std::tuple{32, -1},
                               std::tuple{64, -1},
                               std::tuple{96, 128},
                               std::tuple{128, -1},
                               std::tuple{192, 128},
                               std::tuple{192, -1},
                               std::tuple{256, -1});

const auto SplitKVHDimValues = Values(std::tuple{32, -1},
                                      std::tuple{64, -1},
                                      std::tuple{96, -1},
                                      std::tuple{128, -1},
                                      std::tuple{256, -1});

const auto AppendKVHDimValues =
    Values(std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1});

const auto ModeValues = Values(mode_enum::batch, mode_enum::group);

const auto IsVRowmajorValues = Values(false, true);

const bool squant             = false;
const std::string init_method = "uf";
const bool def_lse            = true;
const bool def_is_v_rowmajor  = true;

int adjust_seqlen(int seqlen) { return seqlen; }

#include "test_fmha_fwd.inc"
