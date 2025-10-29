// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

template <typename GroupConvLayout>
struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    GroupConvLayout layout;
    DataType data_type;
    ElementwiseOperation elementwise_operation;
};
static_assert(ConvSignatureDescriptor<ConvSignature<GroupConvLayout1D>>);
static_assert(ConvSignatureDescriptor<ConvSignature<GroupConvLayout2D>>);
static_assert(ConvSignatureDescriptor<ConvSignature<GroupConvLayout3D>>);

} // namespace ck_tile::builder::test
