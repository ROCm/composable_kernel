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
// 1. **Algorithm Identification**: Six `consteval` predicate functions (`IsReferenceAlgorithm`,
//    `IsXdlV3Algorithm`, `IsXdlAlgorithm`, `IsWmmaAlgorithm`, `IsDlAlgorithm`,
//    `IsLargeTensorAlgorithm`) inspect the algorithm descriptor's structure to determine which
//    kernel variant it satisfies. Each predicate checks a specific set of concept constraints
//    that define a kernel variant.
//
// 2. **Factory Routing**: The main `make_conv_instance()` function uses `if constexpr`
//    to dispatch to the appropriate factory class based on both the convolution direction
//    and the identified algorithm type. All routing decisions occur at compile-time,
//    ensuring zero runtime overhead.
//
// ## Supported Kernel Variants
//
// - **Reference**: Simple reference implementation for validation. Only requires a specialization
//   field set to ConvAlgorithmSpecialization::REFERENCE.
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
#include "ck_tile/builder/factory/conv_fwd_wmma_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp"
#include "ck_tile/builder/factory/reference_factory.hpp"
#include "ck_tile/builder/factory/conv_tile_factory.hpp"
#include "ck_tile/builder/factory/conv_depthwise_tile_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_xdl_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_two_stage_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_multi_d_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_wmma_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_two_stage_wmma_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_weight_multi_d_wmma_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_data_multi_d_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_data_multi_d_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_bwd_data_multi_d_wmma_cshuffle_v3_factory.hpp"

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

    // Reference algorithm supports all directions
    if constexpr(ReferenceAlgorithm<AlgoType>)
    {
        return typename ReferenceFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    // Depthwise tile algorithm - direct spatial pipeline, no GEMM
    else if constexpr(DepthwiseAlgorithm<AlgoType>)
    {
        return typename ConvDepthwiseTileFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    // CK Tile supports common factory for each direction
    else if constexpr(TileAlgorithm<AlgoType>)
    {
        return typename ConvTileFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
    }
    // Forward direction (supports most algorithm variants)
    else if constexpr(ConvDirectionIsForward<SIGNATURE>)
    {
        if constexpr(FwdXdlV3Algorithm<AlgoType>)
        {
            return typename ConvFwdXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdXdlAlgorithm<AlgoType>)
        {
            return typename ConvFwdXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdWmmaV3Algorithm<AlgoType>)
        {
            return typename ConvFwdWmmaV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdWmmaAlgorithm<AlgoType>)
        {
            return typename ConvFwdWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(FwdDlAlgorithm<AlgoType>)
        {
            return typename ConvFwdDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(LargeTensorAlgorithm<AlgoType>)
        {
            return typename ConvFwdLargeTensorFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            static_assert(
                false,
                "No suitable forward convolution kernel factory found for the provided ALGORITHM. "
                "The ALGORITHM must satisfy requirements for one of: Reference, Tile, XDL V3, XDL, "
                "WMMA, DL (NHWC layout), or Large Tensor variant.");
        }
    }
    // Backward data direction
    else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
    {
        if constexpr(BwdMultiDXdlAlgorithm<AlgoType>)
        {
            return typename ConvBwdDataMultiDXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdMultiDWmmaV3Algorithm<AlgoType>)
        {
            return
                typename ConvBwdDataMultiDWmmaV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdMultiDWmmaAlgorithm<AlgoType>)
        {
            return typename ConvBwdDataMultiDWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            static_assert(
                false,
                "No suitable backward data convolution kernel factory found for the provided "
                "ALGORITHM. "
                "The ALGORITHM must satisfy requirements for one of: Reference, XDL multiple d, "
                "Wmma multiple d, "
                "or WMMA multiple d v3.");
        }
    }
    // Backward weight direction (will expand with more algorithms in the future)
    else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
    {
        if constexpr(BwdXdlAlgorithm<AlgoType>)
        {
            return typename ConvBwdWeightXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdXdlV3Algorithm<AlgoType>)
        {
            return typename ConvBwdWeightXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdTwoStageXdlAlgorithm<AlgoType>)
        {
            return
                typename ConvBwdWeightTwoStageXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdDlAlgorithm<AlgoType>)
        {
            return typename ConvBwdWeightDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdMultiDXdlAlgorithm<AlgoType>)
        {
            return
                typename ConvBwdWeightMultiDXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdWmmaV3Algorithm<AlgoType>)
        {
            return typename ConvBwdWeightWmmaV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdTwoStageWmmaV3Algorithm<AlgoType>)
        {
            return typename ConvBwdWeightTwoStageWmmaV3Factory<SIGNATURE, ALGORITHM, VERSION>::
                Instance{};
        }
        else if constexpr(BwdWmmaAlgorithm<AlgoType>)
        {
            return typename ConvBwdWeightWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(BwdMultiDWmmaV3Algorithm<AlgoType>)
        {
            return typename ConvBwdWeightMultiDWmmaV3Factory<SIGNATURE, ALGORITHM, VERSION>::
                Instance{};
        }
        else
        {
            static_assert(
                false,
                "No suitable backward weight convolution kernel factory found for the provided "
                "ALGORITHM. The ALGORITHM must satisfy requirements for one of: Reference, Tile, "
                "XDL, XDL V3, Two-Stage XDL, DL, Multi-D XDL, WMMA V3, Two-Stage "
                "WMMA V3, WMMA, or Multi-D WMMA V3 variant.");
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
