// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_weight_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_data_gpu.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include <memory>

namespace ck_tile::builder::factory {

// Unified Factory for GPU Reference Convolution (all directions)
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
struct ReferenceFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;

    using Types       = internal::ConvTensorDataTypes<SIGNATURE>;
    using InDataType  = typename Types::InDataType;
    using WeiDataType = typename Types::WeiDataType;
    using OutDataType = typename Types::OutDataType;

    using Layouts   = factory::internal::ConvTensorLayouts<SIGNATURE>;
    using InLayout  = typename Layouts::InLayout;
    using WeiLayout = typename Layouts::WeiLayout;
    using OutLayout = typename Layouts::OutLayout;

    using Ops              = factory::internal::ConvElementwiseOps<SIGNATURE>;
    using InElementwiseOp  = typename Ops::InElementwiseOp;
    using WeiElementwiseOp = typename Ops::WeiElementwiseOp;
    using OutElementwiseOp = typename Ops::OutElementwiseOp;

    struct Instance
    {
        // Store template parameters for InstanceTraits reflection
        static constexpr auto kSignature = SIGNATURE;
        static constexpr auto kAlgorithm = ALGORITHM;
        static constexpr auto kVersion   = VERSION;

        /// @brief Invoke reference convolution
        ///
        /// This is the primary overload to invoke reference convolution. As the underlying
        /// function requires it, this function accepts ConvParam directly.
        template <typename InPtrType, typename WeiPtrType, typename OutPtrType>
        static void Run(InPtrType* input,
                        WeiPtrType* weight,
                        OutPtrType* output,
                        const ck::utils::conv::ConvParam& param,
                        InElementwiseOp in_op   = InElementwiseOp{},
                        WeiElementwiseOp wei_op = WeiElementwiseOp{},
                        OutElementwiseOp out_op = OutElementwiseOp{})
        {
            if constexpr(ConvDirectionIsForward<SIGNATURE>)
            {
                ck::ref::naive_conv_fwd<InLayout, WeiLayout, OutLayout>(
                    static_cast<const InDataType*>(input),
                    static_cast<const WeiDataType*>(weight),
                    static_cast<OutDataType*>(output),
                    param,
                    in_op,
                    wei_op,
                    out_op);
            }
            else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
            {
                ck::ref::naive_conv_bwd_data<InLayout, WeiLayout, OutLayout>(
                    static_cast<InDataType*>(input),
                    static_cast<const WeiDataType*>(weight),
                    static_cast<const OutDataType*>(output),
                    param,
                    in_op,
                    wei_op,
                    out_op);
            }
            else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
            {
                ck::ref::naive_conv_bwd_weight<InLayout, WeiLayout, OutLayout>(
                    static_cast<const InDataType*>(input),
                    static_cast<WeiDataType*>(weight),
                    static_cast<const OutDataType*>(output),
                    param,
                    in_op,
                    wei_op,
                    out_op);
            }
        }

        /// @brief Invoke reference convolution
        ///
        /// Convenience overload to avoid having to construct ConvParam manually.
        template <typename InPtrType, typename WeiPtrType, typename OutPtrType>
        static void Run(InPtrType* input,
                        WeiPtrType* weight,
                        OutPtrType* output,
                        int G,
                        int N,
                        int K,
                        int C,
                        const std::vector<ck::long_index_t>& input_spatial,
                        const std::vector<ck::long_index_t>& filter_spatial,
                        const std::vector<ck::long_index_t>& strides,
                        const std::vector<ck::long_index_t>& dilations,
                        const std::vector<ck::long_index_t>& left_pads,
                        const std::vector<ck::long_index_t>& right_pads)
        {
            Run(input,
                weight,
                output,
                ck::utils::conv::ConvParam(SPATIAL_DIM,
                                           G,
                                           N,
                                           K,
                                           C,
                                           filter_spatial,
                                           input_spatial,
                                           strides,
                                           dilations,
                                           left_pads,
                                           right_pads));
        }

        std::string GetTypeString() const
        {
            std::string dir_str;
            if constexpr(ConvDirectionIsForward<SIGNATURE>)
                dir_str = "Forward";
            else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
                dir_str = "BackwardData";
            else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
                dir_str = "BackwardWeight";

            return std::string("GPU_Reference_") + dir_str + "_" + std::to_string(SPATIAL_DIM) +
                   "D";
        }
    };
};

} // namespace ck_tile::builder::factory
