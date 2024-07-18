// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType, typename AccDataType, typename BDataType>
void reference_batched_softmax(const Tensor<ADataType>& a_b_m_n, Tensor<BDataType>& b_b_m_n)
{
    const int N = a_b_m_n.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        AccDataType v_max = ck::NumericLimits<ADataType>::Lowest();

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

            v_exp_sum += ck::math::exp(v_a - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_b_m_n(batch, m, n);

            b_b_m_n(batch, m, n) = ck::math::exp(v_a - v_max) / v_exp_sum;
        }
    };

    make_ParallelTensorFunctor(f, b_b_m_n.mDesc.GetLengths()[0], b_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}
