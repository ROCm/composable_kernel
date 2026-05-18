// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — DataType enum, constexpr queries. No runtime, no CK deps.

#pragma once

#include "rocm_ck/platform.hpp"

#include <cstdint>

namespace rocm_ck {

// FP8 = e4m3, BF8 = e5m2 (CK convention).
enum class DataType : uint8_t
{
    // Floating point — standard widths
    FP64,
    FP32,
    FP16,
    BF16,

    // FP8 variants — see note below
    FP8_FNUZ,
    BF8_FNUZ,
    FP8_OCP,
    BF8_OCP,

    // Integer types — signed and unsigned at each width
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64
};

// FP8 variants — FNUZ and OCP are different number formats, not just HW hints.
//   FNUZ: gfx942 native (higher bias, no Inf, max 240)
//   OCP:  gfx950 native (OCP standard, has Inf, max 448)
// Non-native formats run in software (slower) and produce different numerical
// results. Choose based on target GPU and model training format.
// We keep FNUZ and OCP explicit rather than a generic FP8 — the numerical
// differences matter for compatibility and schema-driven test coverage.
// TODO - We may introduce a generic FP8/BF8 that resolves to the hardware-native type.
// See: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html

// Bits (not bytes) so sub-byte types (I4) are clean integers.
constexpr int dataTypeBits(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 64;
    case DataType::FP32: return 32;
    case DataType::FP16: return 16;
    case DataType::BF16: return 16;
    case DataType::FP8_FNUZ: return 8;
    case DataType::BF8_FNUZ: return 8;
    case DataType::FP8_OCP: return 8;
    case DataType::BF8_OCP: return 8;
    case DataType::I4: return 4;
    case DataType::I8: return 8;
    case DataType::I16: return 16;
    case DataType::I32: return 32;
    case DataType::I64: return 64;
    case DataType::U8: return 8;
    case DataType::U16: return 16;
    case DataType::U32: return 32;
    case DataType::U64: return 64;
    }
    ROCM_CK_UNREACHABLE();
}

constexpr const char* dataTypeName(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return "FP64";
    case DataType::FP32: return "FP32";
    case DataType::FP16: return "FP16";
    case DataType::BF16: return "BF16";
    case DataType::FP8_FNUZ: return "FP8_FNUZ";
    case DataType::BF8_FNUZ: return "BF8_FNUZ";
    case DataType::FP8_OCP: return "FP8_OCP";
    case DataType::BF8_OCP: return "BF8_OCP";
    case DataType::I4: return "I4";
    case DataType::I8: return "I8";
    case DataType::I16: return "I16";
    case DataType::I32: return "I32";
    case DataType::I64: return "I64";
    case DataType::U8: return "U8";
    case DataType::U16: return "U16";
    case DataType::U32: return "U32";
    case DataType::U64: return "U64";
    }
    ROCM_CK_UNREACHABLE();
}

} // namespace rocm_ck
