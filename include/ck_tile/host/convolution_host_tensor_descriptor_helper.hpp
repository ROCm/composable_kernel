// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/core/utility/type.hpp"

namespace ck_tile {
namespace conv {
namespace detail {

template <typename OldLayout>
CK_TILE_HOST std::vector<std::size_t> get_layout_transpose_gnchw_to_old()
{
    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWC> ||
                 ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KXC> ||
                 ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KZYXC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    // separate from legacy code above
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCW> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCX> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKW>)
    {
        return {0, 1, 2, 3};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCHW> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCYX> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKHW>)
    {
        return {0, 1, 2, 3, 4};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCDHW> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCZYX> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKDHW>)
    {
        return {0, 1, 2, 3, 4, 5};
    }
    if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNWC> ||
                 ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKXC> ||
                 ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNHWC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKYXC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNDHWC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKZYXC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KXGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWGK>)
    {
        return {2, 0, 3, 1};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KYXGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWGK>)
    {
        return {3, 0, 4, 1, 2};
    }
    else if constexpr(ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KZYXGC> ||
                      ck_tile::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWGK>)
    {
        return {4, 0, 5, 1, 2, 3};
    }
    else
    {
        printf("%s\n", __func__);
        throw std::runtime_error("wrong! unsupported layout");
    }
}

} // namespace detail

// make tensor descriptor for packed input tensor, and order the dimension in the order of GNCHW
// regardless of physical layout
template <typename InLayout>
CK_TILE_HOST HostTensorDescriptor
make_input_host_tensor_descriptor_g_n_c_wis_packed(const ck_tile::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NWC> ||
                 ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NHWC> ||
                 ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NDHWC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCW> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCHW> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNWC> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNHWC> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNDHWC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NWGC> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NHWGC> ||
                      ck_tile::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NDHWGC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", InLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<InLayout>());
}

// make tensor descriptor for packed weight tensor, and order the dimension in the order of GKCYX
// regardless of physical layout
template <typename WeiLayout>
CK_TILE_HOST HostTensorDescriptor
make_weight_host_tensor_descriptor_g_k_c_xs_packed(const ck_tile::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXC> ||
                 ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                 ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCX> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCYX> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCZYX>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKXC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKYXC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKZYXC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXGC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXGC> ||
                      ck_tile::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXGC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", WeiLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<WeiLayout>());
}

// make tensor descriptor for packed output tensor, and order the dimension in the order of GNKHW
// regardless of physical layout
template <typename OutLayout>
CK_TILE_HOST HostTensorDescriptor
make_output_host_tensor_descriptor_g_n_k_wos_packed(const ck_tile::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NWK> ||
                 ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NHWK> ||
                 ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NDHWK>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKW> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKHW> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNWK> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNHWK> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNDHWK>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NWGK> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NHWGK> ||
                      ck_tile::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NDHWGK>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", OutLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<OutLayout>());
}

} // namespace conv
} // namespace ck_tile
