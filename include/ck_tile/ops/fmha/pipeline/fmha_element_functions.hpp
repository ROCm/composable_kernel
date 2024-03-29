// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QElementFunction_,
          typename KElementFunction_,
          typename VElementFunction_,
          typename BiasElementFunction_,
          typename LSEElementFunction_,
          typename SAccElementFunction_,
          typename PComputeElementFunction_,
          typename OAccElementFunction_>
struct FmhaElementFunctions
{
    using QElementFunction        = remove_cvref_t<QElementFunction_>;
    using KElementFunction        = remove_cvref_t<KElementFunction_>;
    using VElementFunction        = remove_cvref_t<VElementFunction_>;
    using BiasElementFunction     = remove_cvref_t<BiasElementFunction_>;
    using LSEElementFunction      = remove_cvref_t<LSEElementFunction_>;
    using SAccElementFunction     = remove_cvref_t<SAccElementFunction_>;
    using PComputeElementFunction = remove_cvref_t<PComputeElementFunction_>;
    using OAccElementFunction     = remove_cvref_t<OAccElementFunction_>;

    QElementFunction q_element_func;
    KElementFunction k_element_func;
    VElementFunction v_element_func;
    BiasElementFunction bias_element_func;
    LSEElementFunction lse_element_func;
    SAccElementFunction s_acc_element_func;
    PComputeElementFunction p_compute_element_func;
    OAccElementFunction o_acc_element_func;
};

} // namespace ck_tile
