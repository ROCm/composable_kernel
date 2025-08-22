// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

struct null_tensor
{
    CK_TILE_HOST_DEVICE static constexpr auto is_valid()
    {
        return false;
    }
};

} // namespace ck_tile
