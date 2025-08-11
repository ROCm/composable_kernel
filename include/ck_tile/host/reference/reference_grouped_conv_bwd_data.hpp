// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <ck_tile::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
CK_TILE_HOST void reference_grouped_conv_bwd_data(HostTensor<InDataType>& input,
                                                  const HostTensor<WeiDataType>& weight,
                                                  const HostTensor<OutDataType>& output,
                                                  std::vector<ck_tile::long_index_t> conv_strides,
                                                  std::vector<ck_tile::long_index_t> conv_dilations,
                                                  std::vector<ck_tile::long_index_t> in_left_pads,
                                                  std::vector<ck_tile::long_index_t>)
{
    if(!(input.get_num_of_dimension() == NDimSpatial + 3 &&
         weight.get_num_of_dimension() == NDimSpatial + 3 &&
         output.get_num_of_dimension() == NDimSpatial + 3))
    {

        printf("%lu %lu %lu",
               input.get_num_of_dimension(),
               weight.get_num_of_dimension(),
               output.get_num_of_dimension());

        throw std::runtime_error("wrong! inconsistent dimension");
    }

    if constexpr(NDimSpatial == 1)
    {
        auto func = [&](auto g, auto n, auto c, auto wi) {
            std::size_t K = weight.get_lengths()[1];
            std::size_t X = weight.get_lengths()[3];

            std::size_t Wo = output.get_lengths()[3];
            float v_acc    = 0;

            for(std::size_t x = 0; x < X; ++x)
            {
                auto w_tmp = static_cast<ck_tile::long_index_t>(wi) +
                             static_cast<ck_tile::long_index_t>(in_left_pads[0]) -
                             static_cast<ck_tile::long_index_t>(x * conv_dilations[0]);

                if(w_tmp % conv_strides[0] == 0)
                {
                    auto wo = static_cast<ck_tile::long_index_t>(w_tmp) /
                              static_cast<ck_tile::long_index_t>(conv_strides[0]);

                    if(wo >= 0 && ck_tile::type_convert<std::size_t>(wo) < Wo)
                    {
                        for(std::size_t k = 0; k < K; ++k)
                        {
                            OutDataType v_out = output(g, n, k, wo);
                            WeiDataType v_wei = weight(g, k, c, x);
                            v_acc += ck_tile::type_convert<float>(v_out) *
                                     ck_tile::type_convert<float>(v_wei);
                        }
                    }
                }
            }
            InDataType v_acc_converted = ck_tile::type_convert<InDataType>(v_acc);
            input(g, n, c, wi)         = v_acc_converted;
        };

        make_ParallelTensorFunctor(func,
                                   input.get_lengths()[0],
                                   input.get_lengths()[1],
                                   input.get_lengths()[2],
                                   input.get_lengths()[3])(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 2)
    {
        auto func = [&](auto g, auto n, auto c, auto hi, auto wi) {
            std::size_t K = weight.get_lengths()[1];
            std::size_t Y = weight.get_lengths()[3];
            std::size_t X = weight.get_lengths()[4];

            std::size_t Ho = output.get_lengths()[3];
            std::size_t Wo = output.get_lengths()[4];

            float v_acc = 0;

            for(std::size_t y = 0; y < Y; ++y)
            {
                auto h_tmp = static_cast<ck_tile::long_index_t>(hi) +
                             static_cast<ck_tile::long_index_t>(in_left_pads[0]) -
                             static_cast<ck_tile::long_index_t>(y * conv_dilations[0]);
                if(h_tmp % conv_strides[0] == 0)
                {
                    auto ho = static_cast<ck_tile::long_index_t>(h_tmp) /
                              static_cast<ck_tile::long_index_t>(conv_strides[0]);
                    if(ho >= 0 && ck_tile::type_convert<std::size_t>(ho) < Ho)
                    {
                        for(std::size_t x = 0; x < X; ++x)
                        {
                            auto w_tmp = static_cast<ck_tile::long_index_t>(wi) +
                                         static_cast<ck_tile::long_index_t>(in_left_pads[1]) -
                                         static_cast<ck_tile::long_index_t>(x * conv_dilations[1]);
                            if(w_tmp % conv_strides[1] == 0)
                            {
                                auto wo = static_cast<ck_tile::long_index_t>(w_tmp) /
                                          static_cast<ck_tile::long_index_t>(conv_strides[1]);

                                if(wo >= 0 && ck_tile::type_convert<std::size_t>(wo) < Wo)
                                {
                                    for(std::size_t k = 0; k < K; ++k)
                                    {
                                        OutDataType v_out = output(g, n, k, ho, wo);
                                        WeiDataType v_wei = weight(g, k, c, y, x);
                                        v_acc += ck_tile::type_convert<float>(v_out) *
                                                 ck_tile::type_convert<float>(v_wei);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            InDataType v_acc_converted = ck_tile::type_convert<InDataType>(v_acc);
            input(g, n, c, hi, wi)     = v_acc_converted;
        };

        make_ParallelTensorFunctor(func,
                                   input.get_lengths()[0],
                                   input.get_lengths()[1],
                                   input.get_lengths()[2],
                                   input.get_lengths()[3],
                                   input.get_lengths()[4])(std::thread::hardware_concurrency());
    }
    // else if constexpr(NDimSpatial == 3)
    // {
    //     auto func = [&](auto g, auto k, auto c, auto z, auto y, auto x) {
    //         float v_acc = 0;

    //         for(std::size_t n = 0; n < output.get_lengths()[1]; ++n)
    //         {
    //             for(std::size_t do_ = 0; do_ < output.get_lengths()[3]; ++do_)
    //             {
    //                 auto di = static_cast<ck_tile::long_index_t>(do_ * conv_strides[0]) +
    //                           static_cast<ck_tile::long_index_t>(z * conv_dilations[0]) -
    //                           static_cast<ck_tile::long_index_t>(in_left_pads[0]);
    //                 for(std::size_t ho = 0; ho < output.get_lengths()[4]; ++ho)
    //                 {
    //                     auto hi = static_cast<ck_tile::long_index_t>(ho * conv_strides[1]) +
    //                               static_cast<ck_tile::long_index_t>(y * conv_dilations[1]) -
    //                               static_cast<ck_tile::long_index_t>(in_left_pads[1]);
    //                     for(std::size_t wo = 0; wo < output.get_lengths()[5]; ++wo)
    //                     {
    //                         auto wi = static_cast<ck_tile::long_index_t>(wo * conv_strides[2]) +
    //                                   static_cast<ck_tile::long_index_t>(x * conv_dilations[2]) -
    //                                   static_cast<ck_tile::long_index_t>(in_left_pads[2]);
    //                         if(di >= 0 &&
    //                            ck_tile::type_convert<std::size_t>(di) < input.get_lengths()[3] &&
    //                            hi >= 0 &&
    //                            ck_tile::type_convert<std::size_t>(hi) < input.get_lengths()[4] &&
    //                            wi >= 0 &&
    //                            ck_tile::type_convert<std::size_t>(wi) < input.get_lengths()[5])
    //                         {
    //                             InDataType v_in   = input(g, n, c, di, hi, wi);
    //                             OutDataType v_out = output(g, n, k, do_, ho, wo);

    //                             v_acc += ck_tile::type_convert<float>(v_out) *
    //                                      ck_tile::type_convert<float>(v_in);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         WeiDataType v_acc_converted = ck_tile::type_convert<WeiDataType>(v_acc);
    //         weight(g, k, c, z, y, x)    = v_acc_converted;
    //     };

    //     make_ParallelTensorFunctor(func,
    //                                weight.get_lengths()[0],
    //                                weight.get_lengths()[1],
    //                                weight.get_lengths()[2],
    //                                weight.get_lengths()[3],
    //                                weight.get_lengths()[4],
    //                                weight.get_lengths()[5])(std::thread::hardware_concurrency());
    // }
    else
    {
        throw std::runtime_error(
            "Ref_conv_bwd_data: number of dimensions must be between 1 and 3.");
    }
}
} // namespace ck_tile
