// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {
template <auto... val>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
template <typename... type>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
} // namespace ck_tile
