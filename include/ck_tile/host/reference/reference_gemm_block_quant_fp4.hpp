// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename ADataType,
          typename CDataType,
          typename AElementOp = ck_tile::identity,
          typename CElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_mxfp4_act(const HostTensor<ADataType>& a_m_n,
                                           HostTensor<CDataType>& c_m_n,
                                           float alpha                    = 1.702,
                                           float limit                    = 7.0,
                                           const AElementOp& a_element_op = {},
                                           const CElementOp& c_element_op = {})
{

    const std::size_t M = c_m_n.get_length(0);
    const std::size_t N = c_m_n.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        using ComputeType = float;
        auto a_gelu       = ck_tile::type_convert<ComputeType>(a_element_op(a_m_n(m, 2 * n)));
        auto a_linear     = ck_tile::type_convert<ComputeType>(a_element_op(a_m_n(m, 2 * n + 1)));

        a_gelu   = a_gelu > limit ? limit : a_gelu;
        a_linear = ck_tile::clamp(a_linear, -limit, limit);

        auto alpha_gelu = alpha * a_gelu;
        auto value      = a_gelu / (1 + exp(-alpha_gelu));
        auto out        = value * (a_linear + 1);

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(c_element_op(out));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_block_quant_fp4(const HostTensor<ADataType>& a_m_k,
                                                 const HostTensor<BDataType>& b_k_n,
                                                 HostTensor<CDataType>& c_m_n,
                                                 const HostTensor<AccDataType>& bias,
                                                 const HostTensor<uint8_t>& b_scale_m_k,
                                                 const AElementOp& a_element_op     = {},
                                                 const BElementOp& b_element_op     = {},
                                                 const ACCElementOp& acc_element_op = {})
{
    const std::size_t M        = a_m_k.get_length(0);
    const std::size_t N        = b_k_n.get_length(1);
    const std::size_t K        = a_m_k.get_length(1);
    ck_tile::index_t blocksize = 32;

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc   = 0;
        AccDataType pasual  = 0;
        AccDataType b_value = bias(n);
        for(std::size_t k = 0; k < (K / 2); k++)
        {
            using ComputeType = float;
            auto b_scale      = type_convert<int32_t>(b_scale_m_k((2 * k) / blocksize, n)) - 127;
            ComputeType v_a_0, v_a_1;
            ComputeType v_b_0, v_b_1;

            v_a_0 = ck_tile::type_convert<ComputeType>((a_element_op(a_m_k(m, 2 * k))));
            v_a_1 = ck_tile::type_convert<ComputeType>((a_element_op(a_m_k(m, 2 * k + 1))));

            if constexpr(std::is_same_v<BDataType, uint8_t>)
            {
                auto b_pack      = type_convert<pk_fp4_t>(b_element_op(b_k_n(k, n)));
                auto b_scale_fp4 = type_convert<float>(std::pow(2.0f, b_scale));

                auto b_f4_lo = type_convert<pk_fp4_t>(b_pack.unpack(number<0>{}));
                auto b_f4_hi = type_convert<pk_fp4_t>(b_pack.unpack(number<1>{}));

                v_b_0 = type_convert<ComputeType>(b_f4_lo) * b_scale_fp4;
                v_b_1 = type_convert<ComputeType>(b_f4_hi) * b_scale_fp4;
            }
            pasual = v_a_0 * v_b_0 + v_a_1 * v_b_1;
            v_acc += pasual;
        }
        v_acc       = v_acc + b_value;
        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
