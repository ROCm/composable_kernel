// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "<ck_tile/builder/conv_signature_concepts.hpp>"

namespace ck_tile::builder {

struct ConvSignature 
{
    int spatial_dim;
    ConvDirection direction;
    GroupConvLayout layout;
    DataType data_type;
};
static_assert(ConvSignatureDescriptor<ConvSignature>);

} // namespace ck_tile::builder
