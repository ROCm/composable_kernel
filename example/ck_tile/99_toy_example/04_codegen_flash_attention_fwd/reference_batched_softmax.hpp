// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

template <typename ADataType, typename AccDataType, typename BDataType>
void reference_batched_softmax(const ck_tile::HostTensor<ADataType>& a_b_m_n,
                               ck_tile::HostTensor<BDataType>& b_b_m_n)
{
    const int N = a_b_m_n.mDesc.get_lengths()[2];

    auto f = [&](auto batch, auto m) {
        AccDataType v_max = std::numeric_limits<ADataType>::lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_b_m_n(batch, m, n);

            v_max = v_max < v_a ? v_a : v_max;
        }

        AccDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_b_m_n(batch, m, n);

            v_exp_sum += ck_tile::exp(v_a - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_b_m_n(batch, m, n);

            b_b_m_n(batch, m, n) = ck_tile::exp(v_a - v_max) / v_exp_sum;
        }
    };

    ck_tile::make_ParallelTensorFunctor(
        f, b_b_m_n.mDesc.get_lengths()[0], b_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
