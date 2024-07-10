// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include <cassert>
#include <thread>

namespace ck_tile {

namespace detail {

}

template <typename ComputeDataType, typename DataType>
CK_TILE_HOST void reference_rotary_position_embedding(const HostTensor<DataType>& input_bhsd,
                                                      const HostTensor<DataType>& cos_sd,
                                                      const HostTensor<DataType>& sin_sd,
                                                      bool interleaved,
                                                      HostTensor<DataType>& output_bhsd)
{
    assert(cos_sd.get_num_of_dimension() == 2 && sin_sd.get_num_of_dimension() == 2);
    assert(cos_sd.get_length(0) == sin_sd.get_length(0) &&
           cos_sd.get_length(1) == sin_sd.get_length(1));

    const index_t rotary_dim = cos_sd.get_length(1) * 2;
    assert(static_cast<std::size_t>(rotary_dim) <= input_bhsd.get_length(3));

    output_bhsd.ForEach([&](auto& self, auto i) {
        const index_t i_d = i[3];
        if(rotary_dim <= i_d)
        {
            self(i) = input_bhsd(i);
            return;
        }

        const index_t i_s = i[2];

        const ComputeDataType cos = type_convert<ComputeDataType>(
            interleaved ? cos_sd(i_s, i_d / 2) : cos_sd(i_s, i_d % rotary_dim));
        const ComputeDataType sin = type_convert<ComputeDataType>(
            interleaved ? sin_sd(i_s, i_d / 2) : sin_sd(i_s, i_d % rotary_dim));
        const ComputeDataType half_rotated_input = [&] {
            const index_t i_b = i[0];
            const index_t i_h = i[1];

            const index_t hdim      = input_bhsd.get_length(3);
            const index_t half_hdim = hdim / 2;

            if(interleaved)
            {
                const index_t pos = (i_d < half_hdim ? (i_d * 2 + 1) : (i_d - half_hdim) * 2);
                const ComputeDataType sign = (i_d < half_hdim ? 1 : -1);
                return sign * type_convert<ComputeDataType>(input_bhsd(i_b, i_h, i_s, pos));
            }
            else
            {
                const index_t pos          = (i_d + half_hdim) % hdim;
                const ComputeDataType sign = (pos < half_hdim ? 1 : -1);
                return sign * type_convert<ComputeDataType>(input_bhsd(i_b, i_h, i_s, pos));
            }
        }();
        ComputeDataType result =
            type_convert<ComputeDataType>(input_bhsd(i)) * cos + half_rotated_input * sin;

        self(i) = type_convert<DataType>(result);
    });
}

} // namespace ck_tile
