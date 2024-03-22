// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp,
          typename BElementOp,
          typename ACCElementOp>
void reference_batched_gemm(const Tensor<ADataType>& a_b_m_k,
                            const Tensor<BDataType>& b_b_n_k,
                            Tensor<CDataType>& c_b_m_n,
                            const AElementOp& a_element_op,
                            const BElementOp& b_element_op,
                            const ACCElementOp& acc_element_op)
{
    const int N = b_b_n_k.mDesc.GetLengths()[1];
    const int K = b_b_n_k.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_batched_gemm(const Tensor<ADataType>& a_b_m_k,
                            const Tensor<BDataType>& b_b_n_k,
                            Tensor<CDataType>& c_b_m_n)
{
    reference_batched_gemm<ADataType, BDataType, AccDataType, CDataType>(
        a_b_m_k,
        b_b_n_k,
        c_b_m_n,
        [](const ADataType& x) { return x; },
        [](const BDataType& x) { return x; },
        [](const AccDataType& x) { return x; });
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp,
          typename BElementOp,
          typename ACCElementOp>
void reference_batched_tilemasked_sddmm(const Tensor<ADataType>& a_b_m_k,
                             const Tensor<BDataType>& b_b_n_k,
                             Tensor<CDataType>& c_b_m_n,
                             const Tensor<ck::index_t>& tile_mask,
                             ck::index_t tile_h,
                             ck::index_t tile_w,
                             const AElementOp& a_element_op,
                             const BElementOp& b_element_op,
                             const ACCElementOp& acc_element_op)
{
    const int N = b_b_n_k.mDesc.GetLengths()[1];
    const int K = b_b_n_k.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            if (tile_mask(static_cast<ck::index_t>(m / tile_h), static_cast<ck::index_t>(n / tile_w)) == 0) {
                c_b_m_n(batch, m, n) = 0;
                continue;
            }
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
    // make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])();
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_batched_tilemasked_sddmm(const Tensor<ADataType>& a_b_m_k,
                             const Tensor<BDataType>& b_b_n_k,
                             Tensor<CDataType>& c_b_m_n,
                             const Tensor<ck::index_t>& tile_mask,
                             ck::index_t tile_h,
                             ck::index_t tile_w)
{
    reference_batched_tilemasked_sddmm<ADataType, BDataType, AccDataType, CDataType>(
        a_b_m_k,
        b_b_n_k,
        c_b_m_n,
        tile_mask,
        tile_h,
        tile_w,
        [](const ADataType& x) { return x; },
        [](const BDataType& x) { return x; },
        [](const AccDataType& x) { return x; });
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp,
          typename BElementOp,
          typename ACCElementOp>
void reference_batched_tilemasked_spmm(const Tensor<ADataType>& a_b_m_k,
                                  const Tensor<BDataType>& b_b_n_k,
                                  Tensor<CDataType>& c_b_m_n,
                                  const Tensor<ck::index_t>& tile_mask,
                                  ck::index_t tile_h,
                                  ck::index_t tile_w,
                                  const AElementOp& a_element_op,
                                  const BElementOp& b_element_op,
                                  const ACCElementOp& acc_element_op)
{
    const int N = b_b_n_k.mDesc.GetLengths()[1];
    const int K = b_b_n_k.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                if (tile_mask(static_cast<ck::index_t>(m / tile_h), static_cast<ck::index_t>(k / tile_w)) == 0) {
                    continue;
                }
                ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
void reference_batched_tilemasked_spmm(const Tensor<ADataType>& a_b_m_k,
                                  const Tensor<BDataType>& b_b_n_k,
                                  Tensor<CDataType>& c_b_m_n,
                                  const Tensor<ck::index_t>& tile_mask,
                                  ck::index_t tile_h,
                                  ck::index_t tile_w)
{
    reference_batched_tilemasked_spmm<ADataType, BDataType, AccDataType, CDataType>(
        a_b_m_k,
        b_b_n_k,
        c_b_m_n,
        tile_mask,
        tile_h,
        tile_w,
        [](const ADataType& x) { return x; },
        [](const BDataType& x) { return x; },
        [](const AccDataType& x) { return x; });
}
