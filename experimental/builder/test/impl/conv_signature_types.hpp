// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    GroupConvLayout layout;
    DataType data_type;
    ElementwiseOperation elementwise_operation;
    GroupConvDeviceOp device_operation;
};
static_assert(ConvSignatureDescriptor<ConvSignature>);

} // namespace ck_tile::builder::test
