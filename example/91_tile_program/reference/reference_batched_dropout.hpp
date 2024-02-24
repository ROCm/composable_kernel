// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <optional>
#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include <fstream>

template <typename DataType, typename RandValOutputDataType>
void reference_batched_dropout(Tensor<DataType>& in_out_b_m_n,
                               const Tensor<RandValOutputDataType>& randval_b_m_n,
                               const uint8_t& p_undrop_in_uint8_t,
                               const float scale)
{
    const int N = in_out_b_m_n.mDesc.GetLengths()[2];
    auto f      = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            float tmp                 = ck::type_convert<float>(in_out_b_m_n(batch, m, n)) * scale;
            in_out_b_m_n(batch, m, n) = randval_b_m_n(batch, m, n) <= p_undrop_in_uint8_t
                                                 ? ck::type_convert<DataType>(tmp)
                                                 : float(0);
        }
    };

    make_ParallelTensorFunctor(
        f, randval_b_m_n.mDesc.GetLengths()[0], randval_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}
