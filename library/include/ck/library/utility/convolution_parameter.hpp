// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <numeric>
#include <iterator>
#include <vector>

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct ConvParams
{
    ConvParams();
    ConvParams(ck::index_t n_dim,
               ck::index_t n_batch,
               ck::index_t n_out_channels,
               ck::index_t n_in_channels,
               const std::vector<ck::index_t>& filters_len,
               const std::vector<ck::index_t>& input_len,
               const std::vector<ck::index_t>& strides,
               const std::vector<ck::index_t>& dilations,
               const std::vector<ck::index_t>& left_pads,
               const std::vector<ck::index_t>& right_pads);

    ck::index_t num_dim_spatial_;
    ck::index_t N_;
    ck::index_t K_;
    ck::index_t C_;

    std::vector<ck::index_t> filter_spatial_lengths_;
    std::vector<ck::index_t> input_spatial_lengths_;
    std::vector<ck::index_t> output_spatial_lengths_;

    std::vector<ck::index_t> conv_filter_strides_;
    std::vector<ck::index_t> conv_filter_dilations_;

    std::vector<ck::index_t> input_left_pads_;
    std::vector<ck::index_t> input_right_pads_;

    std::vector<ck::index_t> GetOutputSpatialLengths() const;

    std::size_t GetFlops() const;

    template <typename InDataType, typename WeiDataType, typename OutDataType>
    std::size_t GetByte() const
    {
        // sizeof(InDataType) * (N * C * <input spatial lengths product>) +
        // sizeof(WeiDataType) * (K * C * <filter spatial lengths product>) +
        // sizeof(OutDataType) * (N * K * <output spatial lengths product>);
        return sizeof(InDataType) * (N_ * C_ *
                                     std::accumulate(std::begin(input_spatial_lengths_),
                                                     std::end(input_spatial_lengths_),
                                                     static_cast<std::size_t>(1),
                                                     std::multiplies<std::size_t>())) +
               sizeof(WeiDataType) * (K_ * C_ *
                                      std::accumulate(std::begin(filter_spatial_lengths_),
                                                      std::end(filter_spatial_lengths_),
                                                      static_cast<std::size_t>(1),
                                                      std::multiplies<std::size_t>())) +
               sizeof(OutDataType) * (N_ * K_ *
                                      std::accumulate(std::begin(output_spatial_lengths_),
                                                      std::end(output_spatial_lengths_),
                                                      static_cast<std::size_t>(1),
                                                      std::multiplies<std::size_t>()));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::tensor_operation::device::ConvParams& p);
