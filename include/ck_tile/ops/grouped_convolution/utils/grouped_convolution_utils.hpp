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
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     const void* in_ptr_,
                                     const void* wei_ptr_,
                                     void* out_ptr_,
                                     index_t k_batch_)
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* in_ptr;
    const void* wei_ptr;
    void* out_ptr;
    index_t k_batch;
};

using GroupedConvImplicitGemmTraits = TileGemmTraits<true,
                                                     true,
                                                     true,
                                                     ck_tile::tensor_layout::gemm::RowMajor,
                                                     ck_tile::tensor_layout::gemm::ColumnMajor,
                                                     ck_tile::tensor_layout::gemm::RowMajor>;

} // namespace ck_tile
