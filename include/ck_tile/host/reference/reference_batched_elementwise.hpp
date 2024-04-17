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
          typename AElementOp      = ck_tile::identity,
          typename BElementOp      = ck_tile::identity,
          typename BinaryElementOp = ck_tile::plus<AccDataType>>
CK_TILE_HOST void reference_batched_elementwise(const ATensorView& a_b_m_n,
                                                const BTensorView& b_b_m_n,
                                                CTensorView& c_b_m_n,
                                                const AElementOp& a_element_op           = {},
                                                const BElementOp& b_element_op           = {},
                                                const BinaryElementOp& binary_element_op = {})
{
    using ADataType = typename ATensorView::value_type;
    using BDataType = typename BTensorView::value_type;
    using CDataType = typename CTensorView::value_type;

    const ck_tile::index_t N = c_b_m_n.get_length(2);

    const bool broadcast_a_dim_b = (a_b_m_n.get_length(0) == 1);
    const bool broadcast_a_dim_m = (a_b_m_n.get_length(1) == 1);
    const bool broadcast_a_dim_n = (a_b_m_n.get_length(2) == 1);

    const bool broadcast_b_dim_b = (b_b_m_n.get_length(0) == 1);
    const bool broadcast_b_dim_m = (b_b_m_n.get_length(1) == 1);
    const bool broadcast_b_dim_n = (b_b_m_n.get_length(2) == 1);

    auto f = [&](auto batch, auto m) {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            AccDataType v_a{};
            {
                ck_tile::index_t i_b = (broadcast_a_dim_b ? 0 : batch);
                ck_tile::index_t i_m = (broadcast_a_dim_m ? 0 : m);
                ck_tile::index_t i_n = (broadcast_a_dim_n ? 0 : n);

                v_a = ck_tile::type_convert<AccDataType>(a_element_op(a_b_m_n(i_b, i_m, i_n)));
            }

            AccDataType v_b{};
            {
                ck_tile::index_t i_b = (broadcast_b_dim_b ? 0 : batch);
                ck_tile::index_t i_m = (broadcast_b_dim_m ? 0 : m);
                ck_tile::index_t i_n = (broadcast_b_dim_n ? 0 : n);

                v_b = ck_tile::type_convert<AccDataType>(b_element_op(b_b_m_n(i_b, i_m, i_n)));
            }

            c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(binary_element_op(v_a, v_b));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.get_length(0), c_b_m_n.get_length(1))(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
