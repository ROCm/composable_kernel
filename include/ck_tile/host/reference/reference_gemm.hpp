// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/common.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm(const HostTensor<ADataType>& a_m_k,
                                 const HostTensor<BDataType>& b_n_k,
                                 HostTensor<CDataType>& c_m_n,
                                 const AElementOp& a_element_op     = {},
                                 const BElementOp& b_element_op     = {},
                                 const ACCElementOp& acc_element_op = {})
{
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                      ? a_m_k.mDesc.get_lengths()[1]
                      : a_m_k.mDesc.get_lengths()[0];
    const int M = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                      ? a_m_k.mDesc.get_lengths()[0]
                      : a_m_k.mDesc.get_lengths()[1];

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                                    ? a_element_op(a_m_k(m, k))
                                    : a_element_op(a_m_k(k, m));
                BDataType v_b = b_element_op(b_n_k(n, k));

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, M)(std::thread::hardware_concurrency());
}
} // namespace ck_tile
