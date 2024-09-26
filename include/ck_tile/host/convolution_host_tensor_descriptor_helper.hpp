// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
namespace conv {
namespace detail {

template <typename OldLayout>
CK_TILE_HOST std::vector<std::size_t> get_layout_transpose_gnchw_to_old()
{
    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWC> ||
                 std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KXC> ||
                 std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KZYXC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    // separate from legacy code above
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCW> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCX> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKW>)
    {
        return {0, 1, 2, 3};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCHW> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCYX> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKHW>)
    {
        return {0, 1, 2, 3, 4};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNCDHW> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKCZYX> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNKDHW>)
    {
        return {0, 1, 2, 3, 4, 5};
    }
    if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNWC> ||
                 std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKXC> ||
                 std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNHWC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKYXC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNDHWC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GKZYXC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::GNDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KXGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NWGK>)
    {
        return {2, 0, 3, 1};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KYXGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NHWGK>)
    {
        return {3, 0, 4, 1, 2};
    }
    else if constexpr(std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::KZYXGC> ||
                      std::is_same_v<OldLayout, ck_tile::tensor_layout::convolution::NDHWGK>)
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
    if constexpr(std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NWC> ||
                 std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NHWC> ||
                 std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NDHWC>)
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
    else if constexpr(std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCW> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCHW> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNCDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNWC> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNHWC> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::GNDHWC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NWGC> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NHWGC> ||
                      std::is_same_v<InLayout, ck_tile::tensor_layout::convolution::NDHWGC>)
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
    if constexpr(std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXC> ||
                 std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                 std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXC>)
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
    else if constexpr(std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXC>)
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
    else if constexpr(std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCX> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCYX> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKCZYX>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKXC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKYXC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::GKZYXC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KXGC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KYXGC> ||
                      std::is_same_v<WeiLayout, ck_tile::tensor_layout::convolution::KZYXGC>)
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
    if constexpr(std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NWK> ||
                 std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NHWK> ||
                 std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NDHWK>)
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
    else if constexpr(std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKW> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKHW> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNKDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNWK> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNHWK> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::GNDHWK>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NWGK> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NHWGK> ||
                      std::is_same_v<OutLayout, ck_tile::tensor_layout::convolution::NDHWGK>)
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
