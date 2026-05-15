// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — PhysicalTensor. No runtime, no CK deps.
//
// A PhysicalTensor maps a named tensor from the Signature graph to a slot
// in the generic Args buffer. Not every tensor in a compute graph is physical —
// intermediate values (e.g., the S matrix in FMHA = Q*K^T) live only in
// registers and never appear in device memory. The physical tensor table
// describes exactly what the host needs to pack into Args.

#pragma once

#include <rocm_ck/datatype.hpp>
#include <rocm_ck/fixed_string.hpp>
#include <rocm_ck/layout.hpp>

namespace rocm_ck {

inline constexpr int kMaxPhysicalTensors = 8;

struct PhysicalTensor
{
    FixedString<16> name;
    DataType dtype = DataType::FP32;
    Layout layout  = Layout::Row;
    int args_slot  = 0;
};

} // namespace rocm_ck
