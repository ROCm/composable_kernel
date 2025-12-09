// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include <memory>

namespace ck_tile::builder::factory {

// Factory for GPU Reference Backward Weight Convolution
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsBackwardWeight<SIGNATURE> && IsReferenceAlgorithm<decltype(ALGORITHM)>
struct ReferenceBackwardWeightFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Types                         = internal::FwdConvTensorDataTypes<SIGNATURE>;

    using InDataType  = typename Types::ADataType;
    using WeiDataType = typename Types::BDataType;
    using OutDataType = typename Types::EDataType;

    struct Instance
    {
        static void Run(const InDataType* input,
                        WeiDataType* weight_grad,
                        const OutDataType* output_grad,
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
            ck_tile::
                naive_grouped_conv_bwd_weight<SPATIAL_DIM, InDataType, WeiDataType, OutDataType>(
                    input,
                    weight_grad,
                    output_grad,
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
            return std::string("GPU_Reference_BackwardWeight_") + std::to_string(SPATIAL_DIM) + "D";
        }

        auto MakeInvokerPointer() const
        {
            return std::unique_ptr<void, void (*)(void*)>(nullptr, [](void*) {});
        }
    };
};

} // namespace ck_tile::builder::factory
