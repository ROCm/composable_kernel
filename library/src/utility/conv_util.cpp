// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/utility/conv_util.hpp"

namespace ck {
namespace utils {
namespace conv {

/**
 * @brief      Calculate number of FLOPs for Convolution
 *
 * @param[in]  N                       Batch size.
 * @param[in]  C                       Number of input channels.
 * @param[in]  K                       Number of output channels.
 * @param[in]  filter_spatial_lengths  Filter spatial dimensions lengths.
 * @param[in]  output_spatial_lengths  Convolution output spatial dimensions
 *                                     lengths.
 *
 * @return     The number of flops.
 */
std::size_t get_flops(ck::index_t N,
                      ck::index_t C,
                      ck::index_t K,
                      const std::vector<ck::index_t>& filter_spatial_lengths,
                      const std::vector<ck::index_t>& output_spatial_lengths)
{
    // 2 * N * K * <output spatial lengths product> * C * <filter spatial lengths product>
    return static_cast<std::size_t>(2) * N * K *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>()) *
           C *
           std::accumulate(std::begin(filter_spatial_lengths),
                           std::end(filter_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>());
}

ConvParams::ConvParams()
    : num_dim_spatial_(2),
      N_(128),
      K_(256),
      C_(192),
      filter_spatial_lengths_(2, 3),
      input_spatial_lengths_(2, 71),
      conv_filter_strides_(2, 2),
      conv_filter_dilations_(2, 1),
      input_left_pads_(2, 1),
      input_right_pads_(2, 1)
{
}

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
      conv_filter_strides_(strides),
      conv_filter_dilations_(dilations),
      input_left_pads_(left_pads),
      input_right_pads_(right_pads)
{
    if(ck::type_convert<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
    {
        throw(
            std::runtime_error("ConvParams::GetOutputSpatialLengths: "
                               "parameter size is different from number of declared dimensions!"));
    }
}

std::vector<ck::index_t> ConvParams::GetOutputSpatialLengths() const
{
    if(ck::type_convert<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
       ck::type_convert<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
    {
        throw(
            std::runtime_error("ConvParams::GetOutputSpatialLengths: "
                               "parameter size is different from number of declared dimensions!"));
    }

    std::vector<ck::index_t> out_spatial_len(num_dim_spatial_, 0);
    for(ck::index_t i = 0; i < num_dim_spatial_; ++i)
    {
        // XEff = (X - 1) * conv_dilation_w + 1;
        // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
        const ck::index_t idx_eff =
            (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;
        out_spatial_len[i] =
            (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - idx_eff) /
                conv_filter_strides_[i] +
            1;
    }
    return out_spatial_len;
}

ConvParams parse_conv_params(int num_dim_spatial, int arg_idx, char* const argv[])
{
    ck::utils::conv::ConvParams params;

    params.num_dim_spatial_ = num_dim_spatial;
    params.N_               = std::stoi(argv[arg_idx++]);
    params.K_               = std::stoi(argv[arg_idx++]);
    params.C_               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads_[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

HostTensorDescriptor get_output_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                       int num_dim_spatial)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWK{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWK{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWK{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor get_filters_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                        int num_dim_spatial)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KZYXC{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KYXC{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KXC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor get_input_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                      int num_dim_spatial)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWC{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWC{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

} // namespace conv
} // namespace utils
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::utils::conv::ConvParams& p)
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
