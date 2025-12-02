// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/builder_utils.hpp"

namespace ck_tile::builder::factory::internal {

// Type mappings from builder convolution data type to CK tensor types.
template <DataType T>
struct ConvTensorTypes
{
    // This will trigger if a specialization for the given DataType is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(UnsupportedEnumValue<T>) == 0,
                  "Internal error. Unsupported data type for convolution factory.");
};

template <>
struct ConvTensorTypes<DataType::FP16>
{
    using ADataType        = ck::half_t;
    using AComputeType     = ck::half_t;
    using BDataType        = ck::half_t;
    using BComputeType     = ck::half_t;
    using CShuffleDataType = ck::half_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::half_t;
};

template <>
struct ConvTensorTypes<DataType::BF16>
{
    using ADataType        = ck::bhalf_t;
    using AComputeType     = ck::bhalf_t;
    using BDataType        = ck::bhalf_t;
    using BComputeType     = ck::bhalf_t;
    using CShuffleDataType = ck::bhalf_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::bhalf_t;
};

template <>
struct ConvTensorTypes<DataType::FP32>
{
    using ADataType        = float;
    using AComputeType     = float;
    using BDataType        = float;
    using BComputeType     = float;
    using CShuffleDataType = float;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = float;
};

template <>
struct ConvTensorTypes<DataType::I8>
{
    using ADataType        = int8_t;
    using AComputeType     = int8_t;
    using BDataType        = int8_t;
    using BComputeType     = int8_t;
    using CShuffleDataType = int8_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = int32_t;
    using EDataType        = int8_t;
};

template <>
struct ConvTensorTypes<DataType::FP8>
{
    using ADataType        = ck::f8_t;
    using AComputeType     = ck::f8_t;
    using BDataType        = ck::f8_t;
    using BComputeType     = ck::f8_t;
    using CShuffleDataType = ck::f8_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::f8_t;
};

} // namespace ck_tile::builder::factory::internal
