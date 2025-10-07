#pragma once

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature.hpp>

namespace ck_tile::reflect {

// Primary template for ConvTraits - extracts compile-time information from a ConvBuilder
// This is a facade that provides factory values to Description
template <typename Builder>
struct ConvTraits;

// Specialization for ConvBuilder - uses template pattern matching to extract parameters
template <builder::ConvSignatureDescriptor auto SIGNATURE,
          builder::ConvAlgorithmDescriptor auto ALGORITHM,
          builder::StringLiteral VERSION>
struct ConvTraits<builder::ConvBuilder<SIGNATURE, ALGORITHM, VERSION>>
{
    using Factory = builder::ConvFactory<SIGNATURE, ALGORITHM, VERSION>;

    // Signature information (extracted directly from template parameters)
    static constexpr int spatial_dim                  = SIGNATURE.spatial_dim;
    static constexpr builder::ConvDirection direction = SIGNATURE.direction;
    static constexpr builder::GroupConvLayout layout  = SIGNATURE.layout;
    static constexpr builder::DataType data_type      = SIGNATURE.data_type;

    // Algorithm information (read from Factory static constexpr members)
    static constexpr auto block            = Factory::BLOCK;
    static constexpr auto tuning           = Factory::TUNING;
    static constexpr auto a_block_transfer = Factory::A_BLOCK_TRANSFER;
    static constexpr auto b_block_transfer = Factory::B_BLOCK_TRANSFER;
    static constexpr auto c_block_transfer = Factory::C_BLOCK_TRANSFER;
    static constexpr auto pipeline_version = Factory::PIPELINE_VERSION;
};

} // namespace ck_tile::reflect
