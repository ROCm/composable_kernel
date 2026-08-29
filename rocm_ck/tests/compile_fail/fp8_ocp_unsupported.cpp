// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP8_OCP is not yet supported in GEMM wave tiles.
// Expected error: "FP8_OCP/BF8_OCP not yet supported"

#include <rocm_ck/arch_properties.hpp>

using namespace rocm_ck;

constexpr bool bad = isValidWaveTile(DataType::FP8_OCP, 16, 16, 16, GpuTarget::gfx942);
