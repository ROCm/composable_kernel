// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-time dispatcher for convolution kernel instantiation.
//
// This header provides a centralized factory dispatch mechanism that routes algorithm
// specifications to appropriate convolution kernel implementations at compile-time.
//
// ## Design Overview
//
// The dispatcher operates in two phases:
// 1. **Algorithm Identification**: Five `consteval` predicate functions (`IsXdlV3Algorithm`,
//    `IsXdlAlgorithm`, `IsWmmaAlgorithm`, `IsDlAlgorithm`, `IsLargeTensorAlgorithm`) inspect
//    the algorithm descriptor's structure to determine which kernel variant it satisfies.
//    Each predicate checks a specific set of concept constraints that define a kernel variant.
//
// 2. **Factory Routing**: The main `make_conv_instance()` function uses `if constexpr`
//    to dispatch to the appropriate factory class based on both the convolution direction
//    and the identified algorithm type. All routing decisions occur at compile-time,
//    ensuring zero runtime overhead.
//
// ## Supported Kernel Variants
//
// - **XDL V3**: Newer XDL-based pipeline using block GEMM structure. Requires fewer parameters
//   than standard XDL (e.g., uses `SpecifiesBlockGemm` instead of scheduling/prefetch configs).
//
// - **XDL**: Standard XDL-based kernel using AMD XDLops hardware instructions for matrix
//   multiply. Requires full scheduling configuration including prefetch stages and loop scheduler.
//
// - **WMMA**: Wavefront Matrix-Matrix Accumulate variant optimized for WMMA-capable hardware.
//   Requires similar configuration to XDL.
//
// - **DL**: Specialized vectorized dot-product kernel optimized for specific data layouts
//   (NHWC/KYXC/NHWK). The "DL" label just indicates this does not use XDLops instructions.
//
// - **Large Tensor**: XDL-based kernel with extended tensor support. Wraps a base XDL algorithm
//   and adds large tensor capabilities.
//
// ## Current Limitations
//
// Currently only forward convolution is supported. Backward data and backward weight convolution
// directions will fail at compile-time with informative static_assert messages.
//
// ## Usage Example
//
// ```
// auto kernel = make_conv_instance<SIGNATURE, ALGORITHM>();
// ```

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"

// Compile time diagnostics
#include "ck_tile/builder/factory/conv_algorithms.hpp"

// Include all factory implementations
#include "ck_tile/builder/factory/conv_fwd_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp"
#include "ck_tile/builder/factory/conv_tile_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_xdl_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_two_stage_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp"

namespace ck_tile::builder::factory {

// This dispatch logic is rigid and confusing for users. Further, hides most of
// the great error messages from our concepts.
//
// Requirements for a good design:
// 1. Fall through is bad: inputs should get directly to an implementation
//    if we are going to have good compiler errors.
// 2. Logic should be easy for library users to understand.
// 3. Logic should be easy to test, maintain, and extend.
//
// We should probably add explicit tags to the algorithm descriptors, at least
// for the initial implemenation.
//
// To avoid changing behavior too much during refactoring, we leave the explicit
// dispatch logic here for now, just changing it from SFINAE to consteval + if constexpr.
// There may be some subtle behavior changes, but build failure messages will be more
// clear.
//
// TODO: Make this dispatch logic much more robust and clear for users.

template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
constexpr auto make_conv_instance()
{
    using AlgoType = std::remove_const_t<decltype(ALGORITHM)>;

    // CK Tile supports common factory for each direction
    if constexpr(TileAlgorithm<AlgoType>::is_valid())
    {
        return typename ConvTileFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    else if constexpr(ConvDirectionIsForward<SIGNATURE>)
    {
        if constexpr(FwdXdlV3Algorithm<AlgoType>::is_valid())
        {
            return typename ConvFwdXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdXdlAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvFwdXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdWmmaAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvFwdWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdDlAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvFwdDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(LargeTensorAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvFwdLargeTensorFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            diagnose_fwd_algorithm_signature<AlgoType>();
        }
    }
    else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
    {
        static_assert(
            false,
            "Backward data convolution is not yet supported. ");
    }
    else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
    {
        if constexpr (BwdXdlAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvBwdWeightXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr (BwdXdlV3Algorithm<AlgoType>::is_valid())
        {
            return typename ConvBwdWeightXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr (BwdTwoStageXdlAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvBwdWeightTwoStageXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr (BwdDlAlgorithm<AlgoType>::is_valid())
        {
            return typename ConvBwdWeightDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            diagnose_bwd_weight_algorithm_signature<AlgoType>();
        }
    }
    else
    {
        static_assert(false,
                      "Invalid or unsupported convolution direction. "
                      "The SIGNATURE must specify a valid ConvDirection: FORWARD, BACKWARD_DATA, "
                      "or BACKWARD_WEIGHT.");
    }
}

} // namespace ck_tile::builder::factory
