// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_type.hpp"

namespace ck_tile::builder::factory {

// Factory for CK Tile depthwise grouped convolution kernels.
// Instantiates GroupedConvolutionForwardKernel with DepthwiseConvFwdPipeline.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
struct ConvDepthwiseTileFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Types                         = internal::TileConvTensorTypes<SIGNATURE.data_type>;

    static constexpr auto DW = ALGORITHM.depthwise_params;

    using InDataType  = typename Types::ADataType;
    using WeiDataType = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using OutDataType = typename Types::EDataType;

    using DwTraits = ck_tile::DepthwiseConvFwdTraits<InDataType,
                                                     WeiDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     DW.block_size,
                                                     DW.tile_h,
                                                     DW.tile_w,
                                                     DW.filter_h,
                                                     DW.filter_w,
                                                     DW.stride_h,
                                                     DW.stride_w,
                                                     DW.dilation_h,
                                                     DW.dilation_w,
                                                     DW.pad_h,
                                                     DW.pad_w,
                                                     DW.nbatch,
                                                     DW.subtile_h,
                                                     DW.subtile_w,
                                                     DW.in_vec,
                                                     DW.out_vec>;

    using DwPipeline = ck_tile::DepthwiseConvFwdPipeline<DwTraits>;

    using ConvTraitsType = ck_tile::GroupedConvTraits<SPATIAL_DIM,
                                                      ck_tile::ConvolutionSpecialization::Default,
                                                      void,
                                                      void,
                                                      ck_tile::tuple<>,
                                                      void,
                                                      DW.in_vec,
                                                      DW.in_vec,
                                                      DW.out_vec,
                                                      1,
                                                      false,
                                                      false,
                                                      DwTraits>;

    struct DepthwiseNullEpilogue
    {
        using DsLayout      = ck_tile::tuple<>;
        using DsDataType    = ck_tile::tuple<>;
        using ODataType     = OutDataType;
        using AccDataType   = float;
        using CDElementwise = ck_tile::element_wise::PassThrough;
    };

    using Instance = ck_tile::
        GroupedConvolutionForwardKernel<ConvTraitsType, void, DwPipeline, DepthwiseNullEpilogue>;
};

} // namespace ck_tile::builder::factory
