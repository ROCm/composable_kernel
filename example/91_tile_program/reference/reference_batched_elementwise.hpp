// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp      = ck::identity,
          typename BElementOp      = ck::identity,
          typename BinaryElementOp = ck::math::plus<AccDataType>>
void reference_batched_elementwise(const Tensor<ADataType>& a_b_m_n,
                                   const Tensor<BDataType>& b_b_m_n,
                                   Tensor<CDataType>& c_b_m_n,
                                   const AElementOp& a_element_op           = {},
                                   const BElementOp& b_element_op           = {},
                                   const BinaryElementOp& binary_element_op = {})
{
    const ck::index_t N = c_b_m_n.mDesc.GetLengths()[2];

    const bool broadcast_a_dim_b = (a_b_m_n.GetLengths()[0] == 1);
    const bool broadcast_a_dim_m = (a_b_m_n.GetLengths()[1] == 1);
    const bool broadcast_a_dim_n = (a_b_m_n.GetLengths()[2] == 1);

    const bool broadcast_b_dim_b = (b_b_m_n.GetLengths()[0] == 1);
    const bool broadcast_b_dim_m = (b_b_m_n.GetLengths()[1] == 1);
    const bool broadcast_b_dim_n = (b_b_m_n.GetLengths()[2] == 1);

    auto f = [&](auto batch, auto m) {
        for(ck::index_t n = 0; n < N; ++n)
        {
            AccDataType v_a{};
            {
                ck::index_t i_b = (broadcast_a_dim_b ? 0 : batch);
                ck::index_t i_m = (broadcast_a_dim_m ? 0 : m);
                ck::index_t i_n = (broadcast_a_dim_n ? 0 : n);

                v_a = ck::type_convert<AccDataType>(a_element_op(a_b_m_n(i_b, i_m, i_n)));
            }

            AccDataType v_b{};
            {
                ck::index_t i_b = (broadcast_b_dim_b ? 0 : batch);
                ck::index_t i_m = (broadcast_b_dim_m ? 0 : m);
                ck::index_t i_n = (broadcast_b_dim_n ? 0 : n);

                v_b = ck::type_convert<AccDataType>(b_element_op(b_b_m_n(i_b, i_m, i_n)));
            }

            c_b_m_n(batch, m, n) = ck::type_convert<CDataType>(binary_element_op(v_a, v_b));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}
