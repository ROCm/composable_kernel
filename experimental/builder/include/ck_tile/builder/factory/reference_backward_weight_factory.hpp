// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/reference_common.hpp"
#include <memory>

namespace ck_tile::builder::factory {

// Factory for GPU Reference Backward Weight Convolution
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsBackwardWeight<SIGNATURE>
struct ReferenceBackwardWeightFactory
{
    // Validate that only PassThrough elementwise operations are specified
    static constexpr auto kValidation = (internal::ValidateReferenceSignature<SIGNATURE>(), 0);

    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Types                         = internal::FwdConvTensorDataTypes<SIGNATURE>;

    using InDataType  = typename Types::ADataType;
    using WeiDataType = typename Types::BDataType;
    using OutDataType = typename Types::EDataType;

    struct Instance
    {
        // Use common argument structure (const input, mutable weight_grad, const output_grad)
        using Argument =
            internal::ReferenceConvArgument<const InDataType*, WeiDataType*, const OutDataType*>;

        // Use common invoker with backward weight convolution lambda
        struct Invoker
        {
            float Run(const Argument* arg, const StreamConfig& stream_config = StreamConfig{})
            {
                (void)stream_config; // Unused for reference implementation

                ck_tile::naive_grouped_conv_bwd_weight<SPATIAL_DIM,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType>(arg->input_,
                                                                    arg->weight_,
                                                                    arg->output_,
                                                                    arg->G_,
                                                                    arg->N_,
                                                                    arg->K_,
                                                                    arg->C_,
                                                                    arg->input_spatial_,
                                                                    arg->filter_spatial_,
                                                                    arg->output_spatial_,
                                                                    arg->strides_,
                                                                    arg->dilations_,
                                                                    arg->left_pads_);

                return 0.0f; // Reference implementation doesn't track timing
            }
        };

        // Direct Run method (simpler interface, keeps existing API)
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

        // Old CK interface: Create argument pointer
        std::unique_ptr<Argument>
        MakeArgumentPointer(const InDataType* input,
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
                            const std::vector<ck_tile::long_index_t>& left_pads) const
        {
            return std::make_unique<Argument>(input,
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

        // Old CK interface: Create invoker pointer
        std::unique_ptr<Invoker> MakeInvokerPointer() const { return std::make_unique<Invoker>(); }
    };
};

} // namespace ck_tile::builder::factory
