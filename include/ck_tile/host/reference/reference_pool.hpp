// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename InDataType,
          typename ComputeDataType,
          typename OutDataType,
          typename ReduceOp,
          typename InputShape,
          typename OutputShape,
          typename InputStrides,
          typename OutputStrides,
          typename WindowSpatialLengths,
          typename WindowStrides,
          typename WindowDilations,
          typename InputLeftPads,
          typename InputRightPads>
CK_TILE_HOST void reference_pool2d(const HostTensor<InDataType>& input,
                                   HostTensor<OutDataType>& output,
                                   InputShape input_shape,
                                   OutputShape output_shape,
                                   InputStrides /* input_strides */,
                                   OutputStrides /* output_strides */,
                                   WindowSpatialLengths window_spatial_lengths,
                                   WindowStrides window_strides,
                                   WindowDilations window_dilations,
                                   InputLeftPads input_left_pads,
                                   InputRightPads /* input_right_pads */,
                                   ReduceOp reduce_op)
{
    const ck_tile::index_t N = input_shape.at(ck_tile::number<0>{});
    const ck_tile::index_t H = input_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t W = input_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t C = input_shape.at(ck_tile::number<3>{});

    const ck_tile::index_t Ho = output_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t Wo = output_shape.at(ck_tile::number<2>{});

    const ck_tile::index_t Y = window_spatial_lengths.at(ck_tile::number<0>{});
    const ck_tile::index_t X = window_spatial_lengths.at(ck_tile::number<1>{});

    const ck_tile::index_t Sy = window_strides.at(ck_tile::number<0>{});
    const ck_tile::index_t Sx = window_strides.at(ck_tile::number<1>{});

    const ck_tile::index_t Dy = window_dilations.at(ck_tile::number<0>{});
    const ck_tile::index_t Dx = window_dilations.at(ck_tile::number<1>{});

    const ck_tile::index_t LeftPy = input_left_pads.at(ck_tile::number<0>{});
    const ck_tile::index_t LeftPx = input_left_pads.at(ck_tile::number<1>{});
    // Right padding is handled implicitly by bounds checking in the implementation

    auto f = [&](auto n, auto ho, auto wo, auto c) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        for(ck_tile::index_t y = 0; y < Y; ++y)
        {
            // Calculate input height index with stride, dilation, and padding
            ck_tile::index_t hi = ho * Sy + y * Dy - LeftPy;

            for(ck_tile::index_t x = 0; x < X; ++x)
            {
                // Calculate input width index with stride, dilation, and padding
                ck_tile::index_t wi = wo * Sx + x * Dx - LeftPx;

                if(hi >= 0 && hi < H && wi >= 0 && wi < W)
                {
                    const ComputeDataType v_in = type_convert<ComputeDataType>(input(n, hi, wi, c));
                    v_acc                      = reduce_op(v_acc, v_in);
                }
                // For positions outside bounds, we implicitly use identity value
            }
        }

        output(n, ho, wo, c) = ck_tile::type_convert<OutDataType>(v_acc);
    };

    // Parallelize over all output dimensions
    make_ParallelTensorFunctor(f, N, Ho, Wo, C)(std::thread::hardware_concurrency());
}

template <typename InDataType,
          typename ComputeDataType,
          typename OutDataType,
          typename ReduceOp,
          typename InputShape,
          typename OutputShape,
          typename InputStrides,
          typename OutputStrides,
          typename WindowSpatialLengths,
          typename WindowStrides,
          typename WindowDilations,
          typename InputLeftPads,
          typename InputRightPads>
CK_TILE_HOST void reference_pool3d(const HostTensor<InDataType>& input,
                                   HostTensor<OutDataType>& output,
                                   InputShape input_shape,
                                   OutputShape output_shape,
                                   InputStrides /* input_strides */,
                                   OutputStrides /* output_strides */,
                                   WindowSpatialLengths window_spatial_lengths,
                                   WindowStrides window_strides,
                                   WindowDilations window_dilations,
                                   InputLeftPads input_left_pads,
                                   InputRightPads /* input_right_pads */,
                                   ReduceOp reduce_op)
{
    const ck_tile::index_t N = input_shape.at(ck_tile::number<0>{});
    const ck_tile::index_t D = input_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t H = input_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t W = input_shape.at(ck_tile::number<3>{});
    const ck_tile::index_t C = input_shape.at(ck_tile::number<4>{});

    const ck_tile::index_t Do = output_shape.at(ck_tile::number<1>{});
    const ck_tile::index_t Ho = output_shape.at(ck_tile::number<2>{});
    const ck_tile::index_t Wo = output_shape.at(ck_tile::number<3>{});

    const ck_tile::index_t Z = window_spatial_lengths.at(ck_tile::number<0>{});
    const ck_tile::index_t Y = window_spatial_lengths.at(ck_tile::number<1>{});
    const ck_tile::index_t X = window_spatial_lengths.at(ck_tile::number<2>{});

    const ck_tile::index_t Sz = window_strides.at(ck_tile::number<0>{});
    const ck_tile::index_t Sy = window_strides.at(ck_tile::number<1>{});
    const ck_tile::index_t Sx = window_strides.at(ck_tile::number<2>{});

    const ck_tile::index_t Dz = window_dilations.at(ck_tile::number<0>{});
    const ck_tile::index_t Dy = window_dilations.at(ck_tile::number<1>{});
    const ck_tile::index_t Dx = window_dilations.at(ck_tile::number<2>{});

    const ck_tile::index_t LeftPz = input_left_pads.at(ck_tile::number<0>{});
    const ck_tile::index_t LeftPy = input_left_pads.at(ck_tile::number<1>{});
    const ck_tile::index_t LeftPx = input_left_pads.at(ck_tile::number<2>{});
    // Right padding is handled implicitly by bounds checking in the implementation

    auto f = [&](auto n, auto do_, auto ho, auto wo, auto c) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        for(ck_tile::index_t z = 0; z < Z; ++z)
        {
            // Calculate input depth index with stride, dilation, and padding
            ck_tile::index_t di = do_ * Sz + z * Dz - LeftPz;

            for(ck_tile::index_t y = 0; y < Y; ++y)
            {
                // Calculate input height index with stride, dilation, and padding
                ck_tile::index_t hi = ho * Sy + y * Dy - LeftPy;

                for(ck_tile::index_t x = 0; x < X; ++x)
                {
                    // Calculate input width index with stride, dilation, and padding
                    ck_tile::index_t wi = wo * Sx + x * Dx - LeftPx;

                    if(di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W)
                    {
                        const ComputeDataType v_in = type_convert<ComputeDataType>(input(n, di, hi, wi, c));
                        v_acc                      = reduce_op(v_acc, v_in);
                    }
                    // For positions outside bounds, we implicitly use identity value
                }
            }
        }

        output(n, do_, ho, wo, c) = ck_tile::type_convert<OutDataType>(v_acc);
    };

    // Parallelize over all output dimensions
    make_ParallelTensorFunctor(f, N, Do, Ho, Wo, C)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
