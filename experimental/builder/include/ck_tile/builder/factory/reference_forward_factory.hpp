// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/factory/helpers/conv_tensor_type.hpp"
#include <memory>

namespace ck_tile::builder::factory {

// Factory for GPU Reference Forward Convolution
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> && IsReferenceAlgorithm<decltype(ALGORITHM)>
struct ReferenceForwardFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Types                         = internal::FwdConvTensorDataTypes<SIGNATURE>;

    using InDataType  = typename Types::ADataType;
    using WeiDataType = typename Types::BDataType;
    using OutDataType = typename Types::EDataType;

    struct Instance
    {
        // Wrapper for GPU reference kernel
        static void Run(const InDataType* input,
                        const WeiDataType* weight,
                        OutDataType* output,
                        int G,
                        int N,
                        int K,
                        int C,
                        const std::vector<ck_tile::long_index_t>& input_spatial,
                        const std::vector<ck_tile::long_index_t>& filter_spatial,
                        const std::vector<ck_tile::long_index_t>& output_spatial,
                        const std::vector<ck_tile::long_index_t>& strides,
                        const std::vector<ck_tile::long_index_t>& dilations,
                        const std::vector<ck_tile::long_index_t>& left_pads)
        {
            // Call our existing GPU reference kernel
            ck_tile::naive_grouped_conv_fwd<SPATIAL_DIM, InDataType, WeiDataType, OutDataType>(
                input,
                weight,
                output,
                G,
                N,
                K,
                C,
                input_spatial,
                filter_spatial,
                output_spatial,
                strides,
                dilations,
                left_pads);
        }

        std::string GetTypeString() const
        {
            return std::string("GPU_Reference_Forward_") + std::to_string(SPATIAL_DIM) + "D";
        }

        auto MakeInvokerPointer() const
        {
            // For now, return nullptr - will implement invoker if needed
            return std::unique_ptr<void, void (*)(void*)>(nullptr, [](void*) {});
        }
    };
};

} // namespace ck_tile::builder::factory
