// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_gemm(const Tensor<ADataType>& a_m_k,
                    const Tensor<BDataType>& b_n_k,
                    Tensor<CDataType>& c_m_n)
{
    const int N = b_n_k.mDesc.GetLengths()[0];
    const int K = b_n_k.mDesc.GetLengths()[1];

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_m_k(m, k);
                BDataType v_b = b_n_k(n, k);

                v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck::type_convert<CDataType>(v_acc);
        }
    };

    make_ParallelTensorFunctor(f, c_m_n.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}
