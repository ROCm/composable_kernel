// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <vector>

namespace ck_tile::builder::factory::internal {

// Common argument structure for reference convolution implementations
// Template parameters allow different const qualifiers for each direction
template <typename InPtrType, typename WeiPtrType, typename OutPtrType>
struct ReferenceConvArgument
{
    InPtrType input_;
    WeiPtrType weight_;
    OutPtrType output_;
    int G_, N_, K_, C_;
    std::vector<ck_tile::long_index_t> input_spatial_;
    std::vector<ck_tile::long_index_t> filter_spatial_;
    std::vector<ck_tile::long_index_t> output_spatial_;
    std::vector<ck_tile::long_index_t> strides_;
    std::vector<ck_tile::long_index_t> dilations_;
    std::vector<ck_tile::long_index_t> left_pads_;

    ReferenceConvArgument(InPtrType input,
                          WeiPtrType weight,
                          OutPtrType output,
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
        : input_(input),
          weight_(weight),
          output_(output),
          G_(G),
          N_(N),
          K_(K),
          C_(C),
          input_spatial_(input_spatial),
          filter_spatial_(filter_spatial),
          output_spatial_(output_spatial),
          strides_(strides),
          dilations_(dilations),
          left_pads_(left_pads)
    {
    }
};

// Common invoker structure for reference convolution implementations
// Takes a callable (lambda or function pointer) to execute the actual convolution
template <typename ArgumentType, typename ConvFunc>
struct ReferenceConvInvoker
{
    ConvFunc conv_func_;

    explicit ReferenceConvInvoker(ConvFunc func) : conv_func_(func) {}

    float Run(const ArgumentType* arg, const StreamConfig& stream_config = StreamConfig{})
    {
        (void)stream_config; // Unused for reference implementation

        conv_func_(arg->input_,
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

} // namespace ck_tile::builder::factory::internal
