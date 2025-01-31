// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSAsyncDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ true,
                                          /* NumPrefetchV = */ 2>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // assume V can reuse the shared memory by K
        // assume Dropout can reuse the shared memory by V
        return max(GetSmemSizeK<Problem>(),
                   max(GetSmemSizeV<Problem>(), GetSmemSizeDropout<Problem>(0)));
    }
};

} // namespace ck_tile
