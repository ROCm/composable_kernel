// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename AccDataType,
          typename ATensorView,
          typename BTensorView,
          typename CTensorView,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_batched_gemm(const ATensorView& a_b_m_k,
                                         const BTensorView& b_b_n_k,
                                         CTensorView& c_b_m_n,
                                         const AElementOp& a_element_op     = {},
                                         const BElementOp& b_element_op     = {},
                                         const ACCElementOp& acc_element_op = {})
{
    using ADataType = typename ATensorView::value_type;
    using BDataType = typename BTensorView::value_type;
    using CDataType = typename CTensorView::value_type;

    const int N = b_b_n_k.get_length(1);
    const int K = b_b_n_k.get_length(2);

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.get_length(0), c_b_m_n.get_length(1))(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
