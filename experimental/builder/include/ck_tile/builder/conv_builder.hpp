#pragma once

#include <concepts>
#include <type_traits>

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
 * @tparam SIGNATURE The convolution signature, which describes the mathematical functionality of
 * the algorithm (e.g., data types, layouts, direction).
 * @tparam ALGORITHM The specific convolution algorithm to be used for the implementation.
 * @tparam VERSION The version of the builder implementation.
 */
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires SupportedVersion<VERSION> && ValidConvSignature<SIGNATURE>
struct ConvBuilder
{
    static constexpr auto kVersion = VERSION;
    using factory = GroupedConvForwardXldCShuffleFactoryV3<SIGNATURE, ALGORITHM, VERSION>;
    // Output: The kernel class.
    using Instance = factory::Instance;
};

} // namespace ck_tile::builder
