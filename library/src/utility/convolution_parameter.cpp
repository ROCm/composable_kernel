// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

ConvParams::ConvParams(ck::index_t n_dim,
                       ck::index_t n_batch,
                       ck::index_t n_out_channels,
                       ck::index_t n_in_channels,
                       const std::vector<ck::index_t>& filters_len,
                       const std::vector<ck::index_t>& input_len,
                       const std::vector<ck::index_t>& strides,
                       const std::vector<ck::index_t>& dilations,
                       const std::vector<ck::index_t>& left_pads,
                       const std::vector<ck::index_t>& right_pads)
    : num_dim_spatial_(n_dim),
      N_(n_batch),
      K_(n_out_channels),
      C_(n_in_channels),
      filter_spatial_lengths_(filters_len),
      input_spatial_lengths_(input_len),
      output_spatial_lengths_(num_dim_spatial_),
      conv_filter_strides_(strides),
      conv_filter_dilations_(dilations),
      input_left_pads_(left_pads),
      input_right_pads_(right_pads)
{
    if(static_cast<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
    {
        throw(
            std::runtime_error("ConvParams::ConvParams: "
                               "parameter size is different from number of declared dimensions!"));
    }

    for(ck::index_t i = 0; i < num_dim_spatial_; ++i)
    {
        // XEff = (X - 1) * conv_dilation_w + 1;
        // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
        const ck::index_t x_eff = (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

        output_spatial_lengths_[i] =
            (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                conv_filter_strides_[i] +
            1;
    }
}

ConvParams::ConvParams()
    : ConvParams::ConvParams(2, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1})
{
}

std::vector<ck::index_t> ConvParams::GetOutputSpatialLengths() const
{
    return output_spatial_lengths_;
}

std::size_t ConvParams::GetFlops() const
{
    // 2 * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    return static_cast<std::size_t>(2) * N_ * K_ * C_ *
           std::accumulate(std::begin(output_spatial_lengths_),
                           std::end(output_spatial_lengths_),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>()) *
           std::accumulate(std::begin(filter_spatial_lengths_),
                           std::end(filter_spatial_lengths_),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>());
}

} // namespace device
} // namespace tensor_operation
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::tensor_operation::device::ConvParams& p)
{
    os << "ConvParams {"
       << "\nnum_dim_spatial: " << p.num_dim_spatial_ << "\nN: " << p.N_ << "\nK: " << p.K_
       << "\nC: " << p.C_ << "\nfilter_spatial_lengths: " << p.filter_spatial_lengths_
       << "\ninput_spatial_lengths: " << p.input_spatial_lengths_
       << "\nconv_filter_strides: " << p.conv_filter_strides_
       << "\nconv_filter_dilations: " << p.conv_filter_dilations_
       << "\ninput_left_pads: " << p.input_left_pads_
       << "\ninput_right_pads: " << p.input_right_pads_;

    return os;
}
