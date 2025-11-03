// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <utility>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"

#include "fmha_fwd.hpp"

namespace ck_tile {

float fmha_fwd_v3(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&);

} // namespace ck_tile
