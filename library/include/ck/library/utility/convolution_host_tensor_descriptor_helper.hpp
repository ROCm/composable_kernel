// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

#include "ck/library/utility/convolution_parameter.hpp"

namespace ck {
namespace utils {
namespace conv {

template <typename InLayout>
HostTensorDescriptor
get_input_host_tensor_descriptor(const ck::tensor_operation::device::ConvParams& param)
{
    if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NHWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
    {
        std::vector<std::size_t> nhwc_lengths{static_cast<std::size_t>(param.N_),
                                              static_cast<std::size_t>(param.C_)};

        nhwc_lengths.insert(nhwc_lengths.begin() + 1,
                            param.input_spatial_lengths_.begin(),
                            param.input_spatial_lengths_.end());

        return HostTensorDescriptor(nhwc_lengths);
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCHW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NCDHW>)
    {
        std::vector<std::size_t> nchw_lengths{static_cast<std::size_t>(param.N_),
                                              static_cast<std::size_t>(param.C_)};

        nchw_lengths.insert(nchw_lengths.end(),
                            param.input_spatial_lengths_.begin(),
                            param.input_spatial_lengths_.end());

        return HostTensorDescriptor(nchw_lengths);
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout");
    }
}

template <typename WeiLayout>
HostTensorDescriptor
get_weight_host_tensor_descriptor(const ck::tensor_operation::device::ConvParams& param)
{
    if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        std::vector<std::size_t> kyxc_lengths{static_cast<std::size_t>(param.K_),
                                              static_cast<std::size_t>(param.C_)};

        kyxc_lengths.insert(kyxc_lengths.begin() + 1,
                            param.filter_spatial_lengths_.begin(),
                            param.filter_spatial_lengths_.end());

        return HostTensorDescriptor(kyxc_lengths);
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCYX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KCZYX>)
    {
        std::vector<std::size_t> kcyx_lengths{static_cast<std::size_t>(param.K_),
                                              static_cast<std::size_t>(param.C_)};

        kcyx_lengths.insert(kcyx_lengths.end(),
                            param.filter_spatial_lengths_.begin(),
                            param.filter_spatial_lengths_.end());

        return HostTensorDescriptor(kcyx_lengths);
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout");
    }
}

template <typename OutLayout>
HostTensorDescriptor
get_output_host_tensor_descriptor(const ck::tensor_operation::device::ConvParams& param)
{
    if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        std::vector<std::size_t> nhwk_lengths{static_cast<std::size_t>(param.N_),
                                              static_cast<std::size_t>(param.K_)};

        nhwk_lengths.insert(nhwk_lengths.begin() + 1,
                            param.output_spatial_lengths_.begin(),
                            param.output_spatial_lengths_.end());

        return HostTensorDescriptor(nhwk_lengths);
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKHW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NKDHW>)
    {
        std::vector<std::size_t> nkhw_lengths{static_cast<std::size_t>(param.N_),
                                              static_cast<std::size_t>(param.K_)};

        nkhw_lengths.insert(nkhw_lengths.end(),
                            param.output_spatial_lengths_.begin(),
                            param.output_spatial_lengths_.end());

        return HostTensorDescriptor(nkhw_lengths);
    }
    else
    {
        throw std::runtime_error("wrong! unsupported layout");
    }
}

} // namespace conv
} // namespace utils
} // namespace ck
