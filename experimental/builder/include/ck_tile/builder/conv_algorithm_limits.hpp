// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <concepts>

namespace ck_tile::builder 
{

// Limits for output vector transfer.
template <auto Value>
concept OutputVectorTransferLimits = requires {
    requires Value.scalar_per_vector > 0 &&
             Value.m_xdl_per_wave_per_shuffle > 0 &&
             Value.n_xdl_per_wave_per_shuffle > 0 ;
};

} // namespace ck_tile::builder
