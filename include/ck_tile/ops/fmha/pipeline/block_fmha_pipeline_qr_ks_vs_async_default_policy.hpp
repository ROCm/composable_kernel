// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
using BlockFmhaPipelineQRKSVSAsyncDefaultPolicy =
    BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                        /* AsyncCopyK = */ true,
                                        /* AsyncCopyV = */ false,
                                        /* NumPrefetchK = */ 3,
                                        /* NumPrefetchV = */ 3>;

} // namespace ck_tile
