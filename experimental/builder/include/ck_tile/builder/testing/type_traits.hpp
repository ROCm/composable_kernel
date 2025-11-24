// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include "ck_tile/builder/types.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

// TODO(Robin): Test to check that all DataType variants are covered?
// TODO(Robin): Put this file somewhere else?

namespace ck_tile::builder::test {

/// This structure contains some useful traits for CK-Builder's DataType
/// type. Its main usecase is to convert a CK-Builder DataType into an
/// equivalent C++ type.
template <DataType DT>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::FP32>
{
    using Type = float;
};

template <>
struct DataTypeTraits<DataType::FP16>
{
    using Type = ck::half_t;
};

template <>
struct DataTypeTraits<DataType::BF16>
{
    using Type = ck::bhalf_t;
};

template <>
struct DataTypeTraits<DataType::FP8>
{
    using Type = ck::f8_t;
};

template <>
struct DataTypeTraits<DataType::I8>
{
    using Type = int8_t;
};

template <>
struct DataTypeTraits<DataType::U8>
{
    using Type = uint8_t;
};

} // namespace ck_tile::builder::test
