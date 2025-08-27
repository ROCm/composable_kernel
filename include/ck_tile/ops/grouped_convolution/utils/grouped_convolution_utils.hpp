// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/convolution_parameter.hpp"

namespace ck_tile {

/// @brief The Grouped Conv kernel host arguments.
///
/// @par Overview
///      This structure is passed to Grouped Convolution Kernels when creating kernel
///      arguments object. It contain all necessary information required to
///      build proper kernel argument and launch kernel on GPU.
template <typename InPtr, typename WeiPtr, typename OutPtr>
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     InPtr in_ptr_,
                                     WeiPtr wei_ptr_,
                                     const std::vector<const void*> ds_ptr_,
                                     OutPtr out_ptr_,
                                     index_t k_batch_)
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          ds_ptr(ds_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_)
    {
    }

    InPtr in_ptr;
    WeiPtr wei_ptr;
    const std::vector<const void*> ds_ptr;
    OutPtr out_ptr;
    index_t k_batch;
};

using GroupedConvFwdHostArgs       = GroupedConvHostArgs<const void*, const void*, void*>;
using GroupedConvBwdWeightHostArgs = GroupedConvHostArgs<const void*, void*, const void*>;

template <index_t NDimSpatial_,
          ConvolutionSpecialization ConvSpecialization_,
          typename InLayout_,
          typename WeiLayout_,
          typename DsLayout_,
          typename OutLayout_>
struct GroupedConvTraits
{
    private:
    static constexpr auto generate_implicit_gemm_layout()
    {
        return generate_tuple([](auto) { return ck_tile::tensor_layout::gemm::RowMajor{}; },
                              number<DsLayout_::size()>{});
    }

    public:
    static constexpr index_t NumGroupsToMerge                     = 1;
    static constexpr index_t NDimSpatial                          = NDimSpatial_;
    static constexpr ConvolutionSpecialization ConvSpecialization = ConvSpecialization_;
    using InLayout                                                = InLayout_;
    using WeiLayout                                               = WeiLayout_;
    using DsLayout                                                = DsLayout_;
    using OutLayout                                               = OutLayout_;
    using GroupedConvImplicitGemmTraits                           = TileGemmTraits<true,
                                                                                   true,
                                                                                   true,
                                                                                   ck_tile::tensor_layout::gemm::RowMajor,
                                                                                   ck_tile::tensor_layout::gemm::ColumnMajor,
                                                                                   ck_tile::tensor_layout::gemm::RowMajor>;
    static constexpr index_t NumDTensor                           = DsLayout::size();
    using ImplicitGemmDsLayout = decltype(generate_implicit_gemm_layout());
};

} // namespace ck_tile
