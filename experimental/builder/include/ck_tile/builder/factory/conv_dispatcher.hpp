// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"

// Include all factory implementations
#include "ck_tile/builder/factory/conv_fwd_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp"

namespace ck_tile::builder::factory {

// Forward declaration of the dispatcher function
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
constexpr auto make_conv_instance();

// Implementation of the dispatcher
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
constexpr auto make_conv_instance()
{
    // Check convolution direction
    if constexpr(ConvDirectionIsForward<SIGNATURE>)
    {
        // Forward convolution dispatch
        // Check which algorithm concept the ALGORITHM satisfies
        using AlgoType = std::remove_const_t<decltype(ALGORITHM)>;

        if constexpr(DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<AlgoType>)
        {
            return typename ConvFwdXdlV3Factory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<AlgoType>)
        {
            return typename ConvFwdXdlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle<AlgoType>)
        {
            return typename ConvFwdWmmaFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<AlgoType>)
        {
            return typename ConvFwdDlFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else if constexpr(DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<AlgoType>)
        {
            return typename ConvFwdLargeTensorFactory<SIGNATURE, ALGORITHM, VERSION>::Instance{};
        }
        else
        {
            static_assert(
                false,
                "No suitable forward convolution kernel factory found for the provided ALGORITHM. "
                "The ALGORITHM must satisfy one of the following concepts: "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3, "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle, "
                "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle, "
                "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK, or "
                "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor.");
        }
    }
    else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
    {
        static_assert(
            false,
            "Backward data convolution is not yet supported. "
            "Only forward convolution (ConvDirection::FORWARD) is currently implemented.");
    }
    else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
    {
        static_assert(
            false,
            "Backward weight convolution is not yet supported. "
            "Only forward convolution (ConvDirection::FORWARD) is currently implemented.");
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
