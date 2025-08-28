#pragma once

#include <concepts>
#include <type_traits>

#include "conv_builder_reference.hpp"
#include <ck_tile/builder/conv_algorithm.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/versions.h>

namespace ck_tile::builder {

template <ConvSignature TSignature, ConvAlgorithm TAlgorithm, auto Version>
    requires SupportedVersion<Version>
struct ConvBuilder
{
    // Input: Signature describes the mathematical funcationality of the algorithm.
    using Signature = TSignature;
    // Input: Algorithm describes the implementation of the algorithm.
    using Algorithm = TAlgorithm;
    // Input: Version of the builder, exposed for testing.
    static constexpr auto kVersion = Version;
    // Implmentation: The factory handles the builder logic.
    using builder = GroupedConvForwardXldCShuffleFactoryV3<Signature, Algorithm, Version>;
    // Output: The kernel class.
    using Instance = builder::Instance;
};

} // namespace ck_tile::builder
