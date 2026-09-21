// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: device -- maps DataType to CK Tile types. Requires --cuda-device-only.
//
// Maps DataType enum values to CK Tile C++ numeric types.

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "ck_type_map.hpp requires CK Tile headers (device compilation only)"
#endif

#include <rocm_ck/datatype.hpp>

#include <ck_tile/core.hpp>

namespace rocm_ck {

/// Maps a DataType enum value to the corresponding CK Tile numeric type.
/// Primary template is intentionally undefined -- only valid specializations compile.
/// Add specializations as new DataType values are used in device kernels.
template <DataType>
struct CkTypeMap;

template <>
struct CkTypeMap<DataType::FP64>
{
    using type = double;
};
template <>
struct CkTypeMap<DataType::FP32>
{
    using type = float;
};
template <>
struct CkTypeMap<DataType::FP16>
{
    using type = ck_tile::half_t;
};
template <>
struct CkTypeMap<DataType::BF16>
{
    using type = ck_tile::bf16_t;
};
template <>
struct CkTypeMap<DataType::FP8_FNUZ>
{
    using type = ck_tile::fp8_t;
};
template <>
struct CkTypeMap<DataType::BF8_FNUZ>
{
    using type = ck_tile::bf8_t;
};
// FP8_OCP/BF8_OCP: add when CK Tile exposes distinct OCP types.
// Currently ck_tile::fp8_t/bf8_t are selected at compile time via CK_TILE_USE_OCP_FP8.
template <>
struct CkTypeMap<DataType::I8>
{
    using type = int8_t;
};
template <>
struct CkTypeMap<DataType::I32>
{
    using type = int32_t;
};
template <>
struct CkTypeMap<DataType::I4>
{
    using type = ck_tile::pk_int4_t;
};

} // namespace rocm_ck
