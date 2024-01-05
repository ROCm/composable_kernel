// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType, typename AccDataType, typename BDataType>
void reference_reduce(const Tensor<ADataType>& a_m_n, Tensor<BDataType>& b_m)
{
    auto f = [&](auto m) {
        const int N = a_m_n.mDesc.GetLengths()[1];

        AccDataType v_acc = 0;

        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_acc += v_a;
        }

        b_m(m) = ck::type_convert<BDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, b_m.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}
