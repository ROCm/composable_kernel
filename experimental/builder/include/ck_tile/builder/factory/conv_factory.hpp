// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A factory for instantiating CK convolution kernels.
//
// This file translates a semantic description of a convolution operation
// (`ConvSignatureDescriptor` and `ConvAlgorithmDescriptor`) into specific,
// low-level template arguments required by the underlying CK device-level
// kernel implementations. This abstraction enables more complex build
// time logic and simplifies the kernel specification.
//
// Key Components:
//
// Template Metaprogram:
//  - ConvFactory: The main factory, with specializations for different
//                 convolution directions (currently only forward).
//
// The primary entry point is the `ConvFactory` struct, which is specialized
// for different forward convolution kernel types in separate header files.

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/versions.hpp"

namespace ck_tile::builder {

// Primary template for the convolution factory.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          auto VERSION>
struct ConvFactory
{
    // This will trigger if a specialization for the given convolution direction is not found.
    // We should always catch this in an earlier validation check.
    static_assert(false, "Unsupported device operation.");
};

} // namespace ck_tile::builder

// Include all factory specializations
#include "ck_tile/builder/factory/conv_fwd_v3_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_xdl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_wmma_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_dl_factory.hpp"
#include "ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp"
