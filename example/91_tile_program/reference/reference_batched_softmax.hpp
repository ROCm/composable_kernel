// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <optional>
#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType, typename CompDataType, typename BDataType>
void reference_batched_softmax(
    const Tensor<ADataType>& a_b_m_n,
    Tensor<BDataType>& b_b_m_n,
    std::optional<std::reference_wrapper<Tensor<CompDataType>>> lse_b_m = std::nullopt)
{
    const int N = a_b_m_n.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        CompDataType v_max = ck::NumericLimits<CompDataType>::Lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck::type_convert<CompDataType>(a_b_m_n(batch, m, n));

            v_max = v_max < v_a ? v_a : v_max;
        }

        CompDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck::type_convert<CompDataType>(a_b_m_n(batch, m, n));

            v_exp_sum += ck::math::exp(v_a - v_max);
        }

        // if sum is zero(masked), or nan/inf(other computation error), don't do divide
        CompDataType inv_sum = (v_exp_sum == 0.f || v_exp_sum != v_exp_sum) ? 1.f : 1.f / v_exp_sum;

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck::type_convert<CompDataType>(a_b_m_n(batch, m, n));

            b_b_m_n(batch, m, n) =
                ck::type_convert<BDataType>(ck::math::exp(v_a - v_max) * inv_sum);
        }
        // lse
        if(lse_b_m)
        {
            lse_b_m->get()(batch, m) = v_max + ck::math::log(v_exp_sum);
        }
    };

    make_ParallelTensorFunctor(f, b_b_m_n.mDesc.GetLengths()[0], b_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}
