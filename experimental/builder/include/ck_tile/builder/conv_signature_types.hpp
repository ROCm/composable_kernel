#pragma once

#include "conv_signature.hpp"

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
