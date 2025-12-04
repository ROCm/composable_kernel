// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory::internal {

template <ElementwiseOperation T>
struct TileElementwiseOps
{
    // This will trigger if a specialization for the given DataType is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(UnsupportedEnumValue<T>) == 0,
                  "Internal error. Unsupported elementwise operation for convolution factory.");
};

template <>
struct TileElementwiseOps<ElementwiseOperation::PASS_THROUGH>
{
    using AElementwiseOp   = ck_tile::element_wise::PassThrough;
    using BElementwiseOp   = ck_tile::element_wise::PassThrough;
    using CDEElementwiseOp = ck_tile::element_wise::PassThrough;
};

template <>
struct TileElementwiseOps<ElementwiseOperation::SCALE>
{
    using AElementwiseOp   = ck_tile::element_wise::PassThrough;
    using BElementwiseOp   = ck_tile::element_wise::PassThrough;
    using CDEElementwiseOp = ck_tile::element_wise::Scale;
};

} // namespace ck_tile::builder::factory::internal
