// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// TODO: support specifying more elementwise functions for input/output tensors
template <typename PComputeElementFunction_, typename OAccElementFunction_>
struct TileFmhaElementFunctions
{
    using PComputeElementFunction = remove_cvref_t<PComputeElementFunction_>;
    using OAccElementFunction     = remove_cvref_t<OAccElementFunction_>;

    PComputeElementFunction p_compute_element_func;
    OAccElementFunction o_acc_element_func;
};

} // namespace ck_tile
