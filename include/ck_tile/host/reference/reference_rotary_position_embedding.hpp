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

template <typename DataType, typename ComputeDataType>
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
    assert(rotary_dim <= input_bhsd.get_length(3));
    (void)rotary_dim;
    (void)input_bhsd;
    (void)sin_sd;
    (void)cos_sd;
    (void)interleaved;
    (void)output_bhsd;
}

} // namespace ck_tile
