// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum class FmhaTdmSchedLoadKind : index_t
{
    KRead,
    VRead,
    IgnoredLegacy,
};

// These are compiler scheduling points, not hardware synchronization.
enum class FmhaTdmSchedSchedulePoint : index_t
{
    AfterWmma,
    BetweenTokenHalves,
    AfterTokens,
};

} // namespace ck_tile
