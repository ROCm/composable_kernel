// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_batched_gemm(const ck_tile::HostTensor<ADataType>& a_b_m_k,
                            const ck_tile::HostTensor<BDataType>& b_b_n_k,
                            ck_tile::HostTensor<CDataType>& c_b_m_n)
{
    const int N = b_b_n_k.mDesc.get_lengths()[1];
    const int K = b_b_n_k.mDesc.get_lengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_b_m_k(batch, m, k);
                BDataType v_b = b_b_n_k(batch, n, k);

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(v_acc);
        }
    };

    ck_tile::make_ParallelTensorFunctor(
        f, c_b_m_n.mDesc.get_lengths()[0], c_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
