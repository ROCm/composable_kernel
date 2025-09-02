#pragma once

#include <concepts>
#include <type_traits>

#include "conv_builder_reference.hpp"
#include <ck_tile/builder/conv_algorithm.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/versions.h>

namespace ck_tile::builder {

/**
 * @brief Top-level builder for creating convolution kernel instances.
 *
 * This struct serves as the main entry point for generating a convolution kernel.
 * It uses a factory pattern based on the provided signature, algorithm, and version
 * to construct the appropriate kernel instance.
 *
 * @tparam TSignature The convolution signature, which describes the mathematical functionality of
 * the algorithm (e.g., data types, layouts, direction).
 * @tparam ALGORITHM The specific convolution algorithm to be used for the implementation.
 * @tparam Version The version of the builder implementation.
 */
template <ConvSignature TSignature, ConvAlgorithm auto ALGORITHM, auto Version>
    requires SupportedVersion<Version>
struct ConvBuilder
{
    using Signature                = TSignature;
    static constexpr auto kVersion = Version;
    using factory = GroupedConvForwardXldCShuffleFactoryV3<Signature, ALGORITHM, Version>;
    // Output: The kernel class.
    using Instance = factory::Instance;
};

} // namespace ck_tile::builder
