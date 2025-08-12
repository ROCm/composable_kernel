// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename WeightType, typename IndexType>
void reference_basic_gemm_softmax_grouped_topk(const ck_tile::HostTensor<ADataType>& a_m_k,
                          const ck_tile::HostTensor<BDataType>& b_n_k,
                          ck_tile::HostTensor<WeightType>& y_values,
                          ck_tile::HostTensor<IndexType>& y_indices,
                          ck_tile::index_t topk)
{
    const int M = a_m_k.mDesc.get_lengths()[0];
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = b_n_k.mDesc.get_lengths()[1];
    ck_tile::HostTensor<AccDataType> c_m_n({M, N}, {N, 1});

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_m_k(m, k);
                BDataType v_b = b_n_k(n, k);
                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<AccDataType>(v_acc);
        }
        // reference softmax
        AccDataType v_max = std::numeric_limits<ADataType>::lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            v_max = v_max < v_c ? v_c : v_max;
        }

        AccDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            v_exp_sum += ck_tile::exp(v_c - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            c_m_n(m, n) = ck_tile::exp(v_c - v_max) / v_exp_sum;
        }
    };

    ck_tile::make_ParallelTensorFunctor(f, c_m_n.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());

    reference_topk(c_m_n, y_values, y_indices, topk);
    // reference_grouped_topk(c_m_n, y_values, y_indices, topk, num_expert_group, topk_group, dim, largest, sorted);
}

template <typename ADataType, typename BDataType, typename AccDataType>
auto reference_basic_gemm_softmax(const ck_tile::HostTensor<ADataType>& a_m_k,
                          const ck_tile::HostTensor<BDataType>& b_n_k)
{
    const int M = a_m_k.mDesc.get_lengths()[0];
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = b_n_k.mDesc.get_lengths()[1];
    ck_tile::HostTensor<AccDataType> c_m_n({M, N}, {N, 1});

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_m_k(m, k);
                BDataType v_b = b_n_k(n, k);
                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<AccDataType>(v_acc);
        }
        // reference softmax
        AccDataType v_max = std::numeric_limits<ADataType>::lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            v_max = v_max < v_c ? v_c : v_max;
        }

        AccDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            v_exp_sum += ck_tile::exp(v_c - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const AccDataType v_c = c_m_n(m, n);
            c_m_n(m, n) = ck_tile::exp(v_c - v_max) / v_exp_sum;
        }
    };

    ck_tile::make_ParallelTensorFunctor(f, c_m_n.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());

    // reference_topk(c_m_n, y_values, y_indices, topk);
    // reference_grouped_topk(c_m_n, y_values, y_indices, topk, num_expert_group, topk_group, dim, largest, sorted);
    return c_m_n;
}

template <typename ADataType, typename BDataType, typename AccDataType>
auto reference_basic_gemm(const ck_tile::HostTensor<ADataType>& a_m_k,
                          const ck_tile::HostTensor<BDataType>& b_n_k)
{
    const int M = a_m_k.mDesc.get_lengths()[0];
    const int N = b_n_k.mDesc.get_lengths()[0];
    const int K = b_n_k.mDesc.get_lengths()[1];
    ck_tile::HostTensor<AccDataType> c_m_n({M, N}, {N, 1});

    auto f = [&](auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_m_k(m, k);
                BDataType v_b = b_n_k(n, k);
                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_m_n(m, n) = ck_tile::type_convert<AccDataType>(v_acc);
        }
    };

    ck_tile::make_ParallelTensorFunctor(f, c_m_n.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());

    // reference_topk(c_m_n, y_values, y_indices, topk);
    // reference_grouped_topk(c_m_n, y_values, y_indices, topk, num_expert_group, topk_group, dim, largest, sorted);
    return c_m_n;
}
