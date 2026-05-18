// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ck_tile {

template <typename ADataType,
          typename QDataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename QuantGroupSize,
          bool aquant,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_quant(const HostTensor<ADataType>& a_m_k,
                                       const HostTensor<QDataType>& q,
                                       const HostTensor<BDataType>& b_k_n,
                                       HostTensor<CDataType>& c_m_n,
                                       const AElementOp& a_element_op     = {},
                                       const BElementOp& b_element_op     = {},
                                       const ACCElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;

        constexpr std::size_t kGroupK = QuantGroupSize::kK;

        // ---- A loader: dequant A(m,k) into AccDataType ----
        auto load_a = [&](std::size_t k) -> AccDataType {
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = a_element_op(a_m_k(m, k));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                return (k & 1) ? fp32_val.hi : fp32_val.lo;
            }
            else
            {
                return ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(m, k)));
            }
        };

        // ---- B loader: dequant B(k,n) into AccDataType ----
        auto load_b = [&](std::size_t k) -> AccDataType {
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = b_element_op(b_k_n(k, n));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                return (k & 1) ? fp32_val.hi : fp32_val.lo;
            }
            else if constexpr(std::is_same_v<BDataType, fp8_t>)
            {
                return fp8_to_float_raw(b_element_op(b_k_n(k, n)));
            }
            else
            {
                return ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(k, n)));
            }
        };

        // ---- scale loader for a given K-group index ----
        auto load_scale = [&](ck_tile::index_t k_group) -> float {
            const ck_tile::index_t outer_dim = aquant ? (m / QuantGroupSize::kM) : k_group;
            const ck_tile::index_t inner_dim = aquant ? k_group : (n / QuantGroupSize::kN);

            if constexpr(std::is_same_v<QDataType, float>)
            {
                return q(outer_dim, inner_dim);
            }
            else if constexpr(std::is_same_v<QDataType, ck_tile::fp8_t>)
            {
                return fp8_to_float_raw(q(outer_dim, inner_dim));
            }
            else // QDataType == bf8_t by static_assert above
            {
                return bf8_to_float_raw(q(outer_dim, inner_dim));
            }
        };

        // ---- Loop over K by groups (full and tail) ----
        for(std::size_t k_begin = 0; k_begin < K; k_begin += kGroupK)
        {
            const std::size_t k_end = std::min<std::size_t>(k_begin + kGroupK, K);

            AccDataType v_block_acc = 0;

            // unscaled accumulation within this K-group
            for(std::size_t k = k_begin; k < k_end; ++k)
            {
                const AccDataType v_a = load_a(k);
                const AccDataType v_b = load_b(k);
                v_block_acc += v_a * v_b;
            }

            const ck_tile::index_t k_group = static_cast<ck_tile::index_t>(k_begin / kGroupK);
            const float scale              = load_scale(k_group);

            v_acc += v_block_acc * scale;
        }

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
    std::cout << std::endl;
}

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_abquant(const HostTensor<ADataType>& a_m_k,
                                         const HostTensor<AQDataType>& a_q,
                                         const HostTensor<BDataType>& b_k_n,
                                         const HostTensor<BQDataType>& b_q,
                                         HostTensor<CDataType>& c_m_n,
                                         const AElementOp& a_element_op     = {},
                                         const BElementOp& b_element_op     = {},
                                         const ACCElementOp& acc_element_op = {})
{
    constexpr auto A_TENSOR_M_DIM = 0;
    constexpr auto A_TENSOR_K_DIM = 1;
    constexpr auto B_TENSOR_K_DIM = 0;
    constexpr auto B_TENSOR_N_DIM = 1;

    const std::size_t M = a_m_k.get_length(A_TENSOR_M_DIM);
    const std::size_t N = b_k_n.get_length(B_TENSOR_N_DIM);
    const std::size_t K = a_m_k.get_length(A_TENSOR_K_DIM);

    // Pre-convert A/B tensors to AccData type
    // This prevents doing slow reconversions for each row/column
    HostTensor<AccDataType> a_acc(a_m_k.mDesc);
    HostTensor<AccDataType> b_acc(b_k_n.mDesc);

    a_acc.ForEach([&](auto& self, auto index) {
        if constexpr(std::is_same_v<ADataType, pk_int4_t> || std::is_same_v<ADataType, pk_fp4_t>)
        {
            const ADataType pk_val  = a_element_op(a_m_k(index));
            const fp32x2_t fp32_val = pk_val.to_fp32x2();
            self(index)             = (index[A_TENSOR_K_DIM] & 1) ? fp32_val.hi : fp32_val.lo;
        }
        else
        {
            self(index) = ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(index)));
        }
    });

    b_acc.ForEach([&](auto& self, auto index) {
        if constexpr(std::is_same_v<BDataType, pk_int4_t> || std::is_same_v<BDataType, pk_fp4_t>)
        {
            const BDataType pk_val  = b_element_op(b_k_n(index));
            const fp32x2_t fp32_val = pk_val.to_fp32x2();
            self(index)             = (index[B_TENSOR_K_DIM] & 1) ? fp32_val.hi : fp32_val.lo;
        }
        else if constexpr(std::is_same_v<BDataType, fp8_t>)
        {
            self(index) = fp8_to_float_raw(b_element_op(b_k_n(index)));
        }
        else
        {
            self(index) = ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(index)));
        }
    });

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;

        constexpr std::size_t kGroupK = BQuantGroupSize::kK;

        // ---- a scale loader for a given K-group index ----
        auto load_scale_a = [&](ck_tile::index_t k_group) -> float {
            const ck_tile::index_t outer_dim = m / AQuantGroupSize::kM;
            const ck_tile::index_t inner_dim = k_group;

            if constexpr(std::is_same_v<AQDataType, float>)
            {
                return a_q(outer_dim, inner_dim);
            }
            else if constexpr(std::is_same_v<AQDataType, ck_tile::fp8_t>)
            {
                return fp8_to_float_raw(a_q(outer_dim, inner_dim));
            }
            else // QDataType == bf8_t by static_assert above
            {
                return bf8_to_float_raw(a_q(outer_dim, inner_dim));
            }
        };
        // ---- b scale loader for a given K-group index ----
        auto load_scale_b = [&](ck_tile::index_t k_group) -> float {
            const ck_tile::index_t outer_dim = k_group;
            const ck_tile::index_t inner_dim = n / BQuantGroupSize::kN;

            if constexpr(std::is_same_v<BQDataType, float>)
            {
                return b_q(outer_dim, inner_dim);
            }
            else if constexpr(std::is_same_v<BQDataType, ck_tile::fp8_t>)
            {
                return fp8_to_float_raw(b_q(outer_dim, inner_dim));
            }
            else // QDataType == bf8_t by static_assert above
            {
                return bf8_to_float_raw(b_q(outer_dim, inner_dim));
            }
        };
        // ---- Loop over K by groups (full and tail) ----
        for(std::size_t k_begin = 0; k_begin < K; k_begin += kGroupK)
        {
            const std::size_t k_end = std::min<std::size_t>(k_begin + kGroupK, K);

            AccDataType v_block_acc = 0;

            // unscaled accumulation within this K-group
            for(std::size_t k = k_begin; k < k_end; ++k)
            {
                const AccDataType v_a = a_acc(m, k);
                const AccDataType v_b = b_acc(k, n);
                v_block_acc += v_a * v_b;
            }

            const ck_tile::index_t k_group = static_cast<ck_tile::index_t>(k_begin / kGroupK);
            const float scale_a            = load_scale_a(k_group);
            const float scale_b            = load_scale_b(k_group);

            v_acc += v_block_acc * scale_a * scale_b;
        }

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_rowcol_quant(const HostTensor<ADataType>& a_m_k,
                                              const HostTensor<AQDataType>& aq_m_1,
                                              const HostTensor<BDataType>& b_k_n,
                                              const HostTensor<BQDataType>& bq_1_n,
                                              HostTensor<CDataType>& c_m_n,
                                              const AElementOp& a_element_op     = {},
                                              const BElementOp& b_element_op     = {},
                                              const ACCElementOp& acc_element_op = {})
{
    static_assert(std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t>);
    static_assert(std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t>);
    static_assert(std::is_same_v<AccDataType, float>);
    static_assert(std::is_same_v<CDataType, float> || std::is_same_v<CDataType, ck_tile::half_t>);
    static_assert(std::is_same_v<AQDataType, float> && std::is_same_v<BQDataType, float>);
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        // Init accumulator
        AccDataType v_acc = 0;
        // Get row scale for A and column scale for B
        float a_scale = aq_m_1(m, 0);
        float b_scale = bq_1_n(0, n);

        // Compute the dot product
        for(std::size_t k = 0; k < K; ++k)
        {
            AccDataType v_a;
            AccDataType v_b;

            // Process A data
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = a_element_op(a_m_k(m, k));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t_signed_conversion(pk_val);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(m, k)));
            }

            // Process B data
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = b_element_op(b_k_n(k, n));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t_signed_conversion(pk_val);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(k, n)));
            }

            v_acc += v_a * v_b;
        }

        v_acc = v_acc * a_scale * b_scale;

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm_tensor_quant(const HostTensor<ADataType>& a_m_k,
                                              const HostTensor<AQDataType>& aq_1_1,
                                              const HostTensor<BDataType>& b_k_n,
                                              const HostTensor<BQDataType>& bq_1_1,
                                              HostTensor<CDataType>& c_m_n,
                                              const AElementOp& a_element_op     = {},
                                              const BElementOp& b_element_op     = {},
                                              const ACCElementOp& acc_element_op = {})
{
    static_assert(std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t>);
    static_assert(std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t>);
    static_assert(std::is_same_v<AccDataType, float>);
    static_assert(std::is_same_v<CDataType, float> || std::is_same_v<CDataType, ck_tile::half_t>);
    static_assert(std::is_same_v<AQDataType, float> && std::is_same_v<BQDataType, float>);
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        // Init accumulator
        AccDataType v_acc = 0;
        // Get scale for A and scale for B
        const AccDataType a_scale = ck_tile::type_convert<AccDataType>(aq_1_1(0, 0));
        const AccDataType b_scale = ck_tile::type_convert<AccDataType>(bq_1_1(0, 0));

        // Compute the dot product
        for(std::size_t k = 0; k < K; ++k)
        {
            AccDataType v_a = ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(m, k)));
            AccDataType v_b = ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(k, n)));

            v_acc += v_a * v_b;
        }

        v_acc = v_acc * a_scale * b_scale;

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename QDataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename QuantGroupSize,
          typename BLayout,
          bool aquant,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_mx_gemm_bquant(const HostTensor<ADataType>& a_m_k,
                                           const HostTensor<QDataType>& q,
                                           const HostTensor<BDataType>& b_k_n,
                                           HostTensor<CDataType>& c_m_n,
                                           const AElementOp& a_element_op     = {},
                                           const BElementOp& b_element_op     = {},
                                           const ACCElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;
        using ComputeType = float;
        ComputeType v_a;
        ComputeType v_b;

        auto load_b = [&](std::size_t k) -> AccDataType {
            if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
            {
                const auto b_pack = type_convert<pk_fp4_t>(b_element_op(b_k_n(k, n)));
                if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    return (n & 1) ? type_convert<ComputeType>(b_pack.unpack(number<1>{}))
                                   : type_convert<ComputeType>(b_pack.unpack(number<0>{}));
                }
                else
                {
                    return (k & 1) ? type_convert<ComputeType>(b_pack.unpack(number<1>{}))
                                   : type_convert<ComputeType>(b_pack.unpack(number<0>{}));
                }
            }
            else
            {
                return ck_tile::type_convert<ComputeType>(b_element_op(b_k_n(k, n)));
            }
        };

        for(std::size_t k = 0; k < K; k++)
        {
            const auto b_scale = type_convert<float>(q(k / QuantGroupSize::kK, n));
            v_a                = ck_tile::type_convert<ComputeType>(a_element_op(a_m_k(m, k)));
            v_b                = load_b(k) * b_scale;
            v_acc += v_a * v_b;
        }
        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
    std::cout << std::endl;
}

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void
reference_gemm(const HostTensor<if_select_t<ADataType_, tf32_t, float, ADataType_>>& a_m_k,
               const HostTensor<if_select_t<BDataType_, tf32_t, float, BDataType_>>& b_k_n,
               HostTensor<CDataType>& c_m_n,
               const AElementOp& a_element_op     = {},
               const BElementOp& b_element_op     = {},
               const ACCElementOp& acc_element_op = {})
{
    if constexpr(std::is_same_v<ADataType_, tf32_t> || std::is_same_v<BDataType_, tf32_t>)
        static_assert(std::is_same_v<ADataType_, BDataType_>,
                      "ADataType and BDataType must be the same");
    using ADataTypeCompute = ADataType_;
    using ADataTypeBuf     = if_select_t<ADataType_, tf32_t, float, ADataType_>;
    using BDataTypeBuf     = if_select_t<BDataType_, tf32_t, float, BDataType_>;

    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    const bool is_gfx950 = (ck_tile::get_device_name() == "gfx950");

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;

        for(std::size_t k = 0; k < K; ++k)
        {
            AccDataType v_a;
            AccDataType v_b;
            if constexpr(std::is_same_v<ADataTypeBuf, pk_fp4_t>)
            {
                // HostTensor automatically handles packed indexing: a_m_k(m,k) divides offset by
                // PackedSize So a_m_k(m,0) and a_m_k(m,1) return the same packed byte
                const pk_fp4_t pk_val   = a_m_k(m, k);
                const fp32x2_t fp32_val = pk_val.to_fp32x2(1.0f);
                const float unpacked    = (k % 2 == 1) ? fp32_val.hi : fp32_val.lo;
                v_a = ck_tile::type_convert<AccDataType>(a_element_op(unpacked));
            }
            else if constexpr(std::is_same_v<ADataTypeBuf, pk_int4_t>)
            {
                // HostTensor automatically handles packed indexing
                const pk_int4_t pk_val  = a_m_k(m, k);
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                const float unpacked    = (k % 2 == 1) ? fp32_val.hi : fp32_val.lo;
                v_a = ck_tile::type_convert<AccDataType>(a_element_op(unpacked));
            }
            else if constexpr(std::is_same_v<ADataTypeBuf, pk_fp4_t>)
            {
                const pk_fp4_t pk_val   = a_element_op(a_m_k(m, k));
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(pk_val, 1.f);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(m, k)));
            }
            if constexpr(std::is_same_v<BDataTypeBuf, pk_fp4_t>)
            {
                // HostTensor automatically handles packed indexing
                const pk_fp4_t pk_val   = b_k_n(k, n);
                const fp32x2_t fp32_val = pk_val.to_fp32x2(1.0f);
                const float unpacked    = (k % 2 == 1) ? fp32_val.hi : fp32_val.lo;
                v_b = ck_tile::type_convert<AccDataType>(b_element_op(unpacked));
            }
            else if constexpr(std::is_same_v<BDataTypeBuf, pk_int4_t>)
            {
                // HostTensor automatically handles packed indexing
                const pk_int4_t pk_val  = b_k_n(k, n);
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                const float unpacked    = (k % 2 == 1) ? fp32_val.hi : fp32_val.lo;
                v_b = ck_tile::type_convert<AccDataType>(b_element_op(unpacked));
            }
            else if constexpr(std::is_same_v<BDataTypeBuf, pk_fp4_t>)
            {
                const pk_fp4_t pk_val   = b_element_op(b_k_n(k, n));
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(pk_val, 1.f);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(k, n)));
            }

            if constexpr(std::is_same_v<ADataTypeCompute, tf32_t>)
            {
                if(is_gfx950)
                {
                    // gfx950: use 3x bf16 emulation
                    bf16_t v_a_bf16_big   = ck_tile::type_convert<bf16_t>(v_a);
                    bf16_t v_a_bf16_small = ck_tile::type_convert<bf16_t>(
                        v_a - type_convert<AccDataType>(v_a_bf16_big));
                    bf16_t v_b_bf16_big   = ck_tile::type_convert<bf16_t>(v_b);
                    bf16_t v_b_bf16_small = ck_tile::type_convert<bf16_t>(
                        v_b - type_convert<AccDataType>(v_b_bf16_big));

                    v_acc += ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                                 ck_tile::type_convert<AccDataType>(v_b_bf16_small) +
                             ck_tile::type_convert<AccDataType>(v_a_bf16_small) *
                                 ck_tile::type_convert<AccDataType>(v_b_bf16_big) +
                             ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                                 ck_tile::type_convert<AccDataType>(v_b_bf16_big);
                }
                else
                {
                    // Other architectures: tf32 not supported or handled via fp32 fallback
                    v_acc += v_a * v_b;
                }
            }
            else
            {
                v_acc += v_a * v_b;
            }
        }

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp,
          typename BElementOp,
          typename CDElementOp,
          typename ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>,
          typename BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>,
          typename DDataType = remove_cvref_t<std::tuple_element_t<0, DsDataType>>>
CK_TILE_HOST void
reference_gemm_multiple_abd(const std::array<HostTensor<ADataType>, AsDataType::size()>& as_m_k,
                            const std::array<HostTensor<BDataType>, BsDataType::size()>& bs_k_n,
                            const std::array<HostTensor<DDataType>, DsDataType::size()>& ds_m_n,
                            HostTensor<ADataType>& a_m_k,
                            HostTensor<BDataType>& b_k_n,
                            HostTensor<CDataType>& c_m_n,
                            const AElementOp& a_element_op    = {},
                            const BElementOp& b_element_op    = {},
                            const CDElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto as_m_k_tuple =
        generate_tie([&](auto idx) -> auto& { return as_m_k[idx]; }, number<AsDataType::size()>{});

    auto bs_k_n_tuple =
        generate_tie([&](auto idx) -> auto& { return bs_k_n[idx]; }, number<BsDataType::size()>{});

    auto ds_m_n_tuple =
        generate_tie([&](auto idx) -> auto& { return ds_m_n[idx]; }, number<DsDataType::size()>{});

    // Apply elementwise function to A
    auto a_elementwise_fn = [&](auto i, auto j) {
        ck_tile::apply([&](auto&&... t) { a_element_op(a_m_k(i, j), t(i, j)...); }, as_m_k_tuple);
    };

    make_ParallelTensorFunctor(a_elementwise_fn, M, K)(std::thread::hardware_concurrency());

    // Apply elementwise function to B
    auto b_elementwise_fn = [&](auto i, auto j) {
        ck_tile::apply([&](auto&&... t) { b_element_op(b_k_n(i, j), t(i, j)...); }, bs_k_n_tuple);
    };

    make_ParallelTensorFunctor(b_elementwise_fn, K, N)(std::thread::hardware_concurrency());

    auto f_mk_kn_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;
        for(std::size_t k = 0; k < K; ++k)
        {
            ADataType v_a = a_m_k(m, k);
            BDataType v_b = b_k_n(k, n);
            v_acc +=
                ck_tile::type_convert<AccDataType>(v_a) * ck_tile::type_convert<AccDataType>(v_b);
        }

        CDataType v_c = 0;

        ck_tile::apply(
            [&](auto&&... t) {
                acc_element_op(v_c,
                               ck_tile::type_convert<float>(v_acc),
                               ck_tile::type_convert<float>(t(m, n))...);
            },
            ds_m_n_tuple);

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(v_c);
    };

    make_ParallelTensorFunctor(f_mk_kn_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename BDataType,
          typename AScaleDataType,
          typename BScaleDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_mx_gemm(const HostTensor<ADataType>& a_m_k,
                                    const HostTensor<BDataType>& b_k_n,
                                    HostTensor<CDataType>& c_m_n,
                                    const HostTensor<AScaleDataType>& scale_a,
                                    const HostTensor<BScaleDataType>& scale_b,
                                    const AElementOp&   = {},
                                    const BElementOp&   = {},
                                    const ACCElementOp& = {})
{
    static_assert(std::is_same_v<AElementOp, ck_tile::identity>);
    static_assert(std::is_same_v<BElementOp, ck_tile::identity>);
    static_assert(std::is_same_v<ACCElementOp, ck_tile::identity>);

    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    const std::size_t ScaleBlockSize = K / scale_a.get_length(1);

    HostTensor<AccDataType> a_m_k_scaled({std::size_t(M), std::size_t(K)},
                                         {std::size_t(K), std::size_t(1)});
    HostTensor<AccDataType> b_k_n_scaled({std::size_t(K), std::size_t(N)},
                                         {std::size_t(1), std::size_t(K)});

    for(std::size_t m = 0; m < M; ++m)
    {
        for(std::size_t k = 0; k < K; ++k)
        {
            if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
            {
                auto a_f4x2         = a_m_k(m, k);
                std::size_t raw_off = m * a_m_k.get_stride(0) + k * a_m_k.get_stride(1);
                auto a_val =
                    (raw_off % 2 == 0)
                        ? ck_tile::type_convert<AccDataType>(a_f4x2.template unpack<>(number<0>{}))
                        : ck_tile::type_convert<AccDataType>(a_f4x2.template unpack<>(number<1>{}));
                auto a_scale = ck_tile::type_convert<AccDataType>(scale_a(m, k / ScaleBlockSize));
                a_m_k_scaled(m, k) = a_val * a_scale;
            }
            else if constexpr(std::is_same_v<ADataType, pk_fp6x16_t>)
            {
                std::size_t raw_off = m * a_m_k.get_stride(0) + k * a_m_k.get_stride(1);
                std::size_t idx     = raw_off % pk_fp6x16_t::packed_size;
                auto a_scale = ck_tile::type_convert<AccDataType>(scale_a(m, k / ScaleBlockSize));
                a_m_k_scaled(m, k) =
                    pk_fp6x16_t::fp6_e2m3_to_float(a_m_k(m, k).unpack(idx)) * a_scale;
            }
            else
            {
                a_m_k_scaled(m, k) =
                    ck_tile::type_convert<AccDataType>((a_m_k(m, k))) *
                    ck_tile::type_convert<AccDataType>(scale_a(m, k / ScaleBlockSize));
            }
        }
    }

    for(std::size_t n = 0; n < N; n++)
    {
        for(std::size_t k = 0; k < K; k++)
        {
            if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
            {
                auto b_f4x2         = b_k_n(k, n);
                std::size_t raw_off = k * b_k_n.get_stride(0) + n * b_k_n.get_stride(1);
                auto b_val =
                    (raw_off % 2 == 0)
                        ? ck_tile::type_convert<AccDataType>(b_f4x2.template unpack<>(number<0>{}))
                        : ck_tile::type_convert<AccDataType>(b_f4x2.template unpack<>(number<1>{}));
                auto b_scale = ck_tile::type_convert<AccDataType>(scale_b(k / ScaleBlockSize, n));
                b_k_n_scaled(k, n) = b_val * b_scale;
            }
            else if constexpr(std::is_same_v<BDataType, pk_fp6x16_t>)
            {
                std::size_t raw_off = k * b_k_n.get_stride(0) + n * b_k_n.get_stride(1);
                std::size_t idx     = raw_off % pk_fp6x16_t::packed_size;
                auto b_scale = ck_tile::type_convert<AccDataType>(scale_b(k / ScaleBlockSize, n));
                b_k_n_scaled(k, n) =
                    pk_fp6x16_t::fp6_e2m3_to_float(b_k_n(k, n).unpack(idx)) * b_scale;
            }
            else
            {
                b_k_n_scaled(k, n) =
                    ck_tile::type_convert<AccDataType>((b_k_n(k, n))) *
                    ck_tile::type_convert<AccDataType>(scale_b(k / ScaleBlockSize, n));
            }
        }
    }

    // call reference gemm
    reference_gemm<AccDataType, AccDataType, AccDataType, CDataType>(
        a_m_k_scaled, b_k_n_scaled, c_m_n);
}

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ACCElementOp,
          typename DDataType = remove_cvref_t<std::tuple_element_t<0, DsDataType>>>
CK_TILE_HOST void
reference_gemm_multiple_d(const HostTensor<ADataType>& a_m_k,
                          const HostTensor<BDataType>& b_k_n,
                          const std::array<HostTensor<DDataType>, DsDataType::size()>& ds_m_n,
                          HostTensor<CDataType>& c_m_n,
                          const ACCElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mk_kn_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;
        for(std::size_t k = 0; k < K; ++k)
        {
            ADataType v_a = a_m_k(m, k);
            BDataType v_b = b_k_n(k, n);
            v_acc +=
                ck_tile::type_convert<AccDataType>(v_a) * ck_tile::type_convert<AccDataType>(v_b);
        }

        CDataType v_c = 0;
        if constexpr(DsDataType::size() == 0)
        {
            acc_element_op(v_c, ck_tile::type_convert<float>(v_acc));
        }
        else if constexpr(DsDataType::size() == 1)
        {
            acc_element_op(v_c,
                           ck_tile::type_convert<float>(v_acc),
                           ck_tile::type_convert<float>(ds_m_n[0](m, n)));
        }
        else if constexpr(DsDataType::size() == 2)
        {
            acc_element_op(v_c,
                           ck_tile::type_convert<float>(v_acc),
                           ck_tile::type_convert<float>(ds_m_n[0](m, n)),
                           ck_tile::type_convert<float>(ds_m_n[1](m, n)));
        }
        c_m_n(m, n) = ck_tile::type_convert<CDataType>(v_c);
    };

    make_ParallelTensorFunctor(f_mk_kn_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void naive_gemm_kernel(if_select_t<ADataType_, tf32_t, float, ADataType_>* A,
                                  if_select_t<BDataType_, tf32_t, float, BDataType_>* B,
                                  CDataType* C,
                                  ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t strideA,
                                  ck_tile::index_t strideB,
                                  ck_tile::index_t strideC)
{
    if constexpr(std::is_same_v<ADataType_, tf32_t> || std::is_same_v<BDataType_, tf32_t>)
        static_assert(std::is_same_v<ADataType_, BDataType_>,
                      "ADataType and BDataType must be the same");
    using ADataTypeCompute = ADataType_;
    // ADataTypeBuf: buffer/storage type (fp32 when tf32)
    using ADataTypeBuf = if_select_t<ADataType_, tf32_t, float, ADataType_>;
    using BDataTypeBuf = if_select_t<BDataType_, tf32_t, float, BDataType_>;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N; // Compute row index
    int col = idx % N; // Compute column index

    if(row < M && col < N)
    {
        AccDataType acc = 0.0;
        for(int k = 0; k < K; ++k)
        {
            constexpr index_t packed_size_a = ck_tile::numeric_traits<ADataTypeBuf>::PackedSize;
            constexpr index_t packed_size_b = ck_tile::numeric_traits<BDataTypeBuf>::PackedSize;
            // Adjust indexing based on matrix layout
            int a_index = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                              ? row * strideA + k
                              : k * strideA + row;
            int b_index = (std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
                              ? col * strideB + k
                              : k * strideB + col;

            AccDataType v_a;
            AccDataType v_b;
            if constexpr(std::is_same_v<ADataTypeBuf, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(A[a_index / packed_size_a]);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else if constexpr(std::is_same_v<ADataTypeBuf, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(A[a_index / packed_size_a], 1.0f);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(A[a_index]);
            }
            if constexpr(std::is_same_v<BDataTypeBuf, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(B[b_index / packed_size_b]);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else if constexpr(std::is_same_v<BDataTypeBuf, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(B[b_index / packed_size_b], 1.0f);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(B[b_index]);
            }

            if constexpr(std::is_same_v<ADataTypeCompute, tf32_t>)
            {
#ifdef CK_GFX950_SUPPORT
                // gfx950: use 3x bf16 emulation
                bf16_t v_a_bf16_big = ck_tile::type_convert<bf16_t>(v_a);
                bf16_t v_a_bf16_small =
                    ck_tile::type_convert<bf16_t>(v_a - type_convert<AccDataType>(v_a_bf16_big));
                bf16_t v_b_bf16_big = ck_tile::type_convert<bf16_t>(v_b);
                bf16_t v_b_bf16_small =
                    ck_tile::type_convert<bf16_t>(v_b - type_convert<AccDataType>(v_b_bf16_big));

                acc += ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                           ck_tile::type_convert<AccDataType>(v_b_bf16_small) +
                       ck_tile::type_convert<AccDataType>(v_a_bf16_small) *
                           ck_tile::type_convert<AccDataType>(v_b_bf16_big) +
                       ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                           ck_tile::type_convert<AccDataType>(v_b_bf16_big);
#else
                // Other architectures: use fp32 fallback
                acc += v_a * v_b;
#endif
            }
            else
            {
                acc += v_a * v_b;
            }
        }

        int c_index = (std::is_same_v<LayoutC, tensor_layout::gemm::RowMajor>)
                          ? row * strideC + col
                          : col * strideC + row;
        C[c_index]  = ck_tile::type_convert<CDataType>(acc);
    }
}

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void blockwise_gemm_kernel(if_select_t<ADataType_, tf32_t, float, ADataType_>* A,
                                      if_select_t<BDataType_, tf32_t, float, BDataType_>* B,
                                      CDataType* C,
                                      ck_tile::index_t M,
                                      ck_tile::index_t N,
                                      ck_tile::index_t K,
                                      ck_tile::index_t strideA,
                                      ck_tile::index_t strideB,
                                      ck_tile::index_t strideC,
                                      ck_tile::index_t scale_granularity_m,
                                      ck_tile::index_t scale_granularity_n,
                                      ck_tile::index_t scale_granularity_k,
                                      float* scale_A_ptr,
                                      float* scale_B_ptr)
{
    if constexpr(std::is_same_v<ADataType_, tf32_t> || std::is_same_v<BDataType_, tf32_t>)
        static_assert(std::is_same_v<ADataType_, BDataType_>,
                      "ADataType and BDataType must be the same");
    using ADataTypeCompute = ADataType_;
    // ADataTypeBuf: buffer/storage type (fp32 when tf32)
    using ADataTypeBuf = if_select_t<ADataType_, tf32_t, float, ADataType_>;
    using BDataTypeBuf = if_select_t<BDataType_, tf32_t, float, BDataType_>;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N; // Compute row index
    int col = idx % N; // Compute column index

    if(row < M && col < N)
    {
        AccDataType acc = 0.0, acc_temp = 0.0;

        index_t scale_A_stride = (M + scale_granularity_m - 1) / scale_granularity_m;
        index_t scale_B_stride = (N + scale_granularity_n - 1) / scale_granularity_n;

        float scale_A = 0;
        float scale_B = 0;

        for(int k = 0; k < K; ++k)
        {
            if(k % scale_granularity_k == 0)
            {
                // update acc
                acc += acc_temp * scale_A * scale_B;
                acc_temp = 0.0;
                // update scale factors
                scale_A = scale_A_ptr[(row / scale_granularity_m) +
                                      (k / scale_granularity_k) * scale_A_stride];
                scale_B = scale_B_ptr[(col / scale_granularity_n) +
                                      (k / scale_granularity_k) * scale_B_stride];
            }

            constexpr index_t packed_size_a = ck_tile::numeric_traits<ADataTypeBuf>::PackedSize;
            constexpr index_t packed_size_b = ck_tile::numeric_traits<BDataTypeBuf>::PackedSize;
            // Adjust indexing based on matrix layout
            int a_index = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                              ? row * strideA + k
                              : k * strideA + row;
            int b_index = (std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
                              ? col * strideB + k
                              : k * strideB + col;

            AccDataType v_a;
            AccDataType v_b;
            if constexpr(std::is_same_v<ADataTypeBuf, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(A[a_index / packed_size_a]);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else if constexpr(std::is_same_v<ADataTypeBuf, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(A[a_index / packed_size_a], 1.0f);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(A[a_index]);
            }

            if constexpr(std::is_same_v<BDataTypeBuf, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(B[b_index / packed_size_b]);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else if constexpr(std::is_same_v<BDataTypeBuf, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(B[b_index / packed_size_b], 1.0f);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(B[b_index]);
            }

            if constexpr(std::is_same_v<ADataTypeCompute, tf32_t>)
            {
#ifdef CK_GFX950_SUPPORT
                // gfx950: use 3x bf16 emulation
                bf16_t v_a_bf16_big = ck_tile::type_convert<bf16_t>(v_a);
                bf16_t v_a_bf16_small =
                    ck_tile::type_convert<bf16_t>(v_a - type_convert<AccDataType>(v_a_bf16_big));
                bf16_t v_b_bf16_big = ck_tile::type_convert<bf16_t>(v_b);
                bf16_t v_b_bf16_small =
                    ck_tile::type_convert<bf16_t>(v_b - type_convert<AccDataType>(v_b_bf16_big));

                acc_temp += ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                                ck_tile::type_convert<AccDataType>(v_b_bf16_small) +
                            ck_tile::type_convert<AccDataType>(v_a_bf16_small) *
                                ck_tile::type_convert<AccDataType>(v_b_bf16_big) +
                            ck_tile::type_convert<AccDataType>(v_a_bf16_big) *
                                ck_tile::type_convert<AccDataType>(v_b_bf16_big);
#else
                // Other architectures: use fp32 fallback
                acc_temp += v_a * v_b;
#endif
            }
            else
            {
                acc_temp += v_a * v_b;
            }
        }
        // final accumulation
        acc += acc_temp * scale_A * scale_B;

        int c_index = (std::is_same_v<LayoutC, tensor_layout::gemm::RowMajor>)
                          ? row * strideC + col
                          : col * strideC + row;
        C[c_index]  = ck_tile::type_convert<CDataType>(acc);
    }
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void reference_gemm_gpu(if_select_t<ADataType, tf32_t, float, ADataType>* a_ptr,
                        if_select_t<BDataType, tf32_t, float, BDataType>* b_ptr,
                        CDataType* c_ptr,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_a,
                        index_t stride_b,
                        index_t stride_c)
{
    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    naive_gemm_kernel<ADataType, BDataType, AccDataType, CDataType, LayoutA, LayoutB, LayoutC>
        <<<numBlocks, numThreadsPerBlock>>>(
            a_ptr, b_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c);

    return;
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void reference_blockwise_gemm_gpu(if_select_t<ADataType, tf32_t, float, ADataType>* a_ptr,
                                  if_select_t<BDataType, tf32_t, float, BDataType>* b_ptr,
                                  CDataType* c_ptr,
                                  index_t M,
                                  index_t N,
                                  index_t K,
                                  index_t stride_a,
                                  index_t stride_b,
                                  index_t stride_c,
                                  index_t scale_granularity_m,
                                  index_t scale_granularity_n,
                                  index_t scale_granularity_k,
                                  float* scale_A_ptr,
                                  float* scale_B_ptr)
{
    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    blockwise_gemm_kernel<ADataType, BDataType, AccDataType, CDataType, LayoutA, LayoutB, LayoutC>
        <<<numBlocks, numThreadsPerBlock>>>(a_ptr,
                                            b_ptr,
                                            c_ptr,
                                            M,
                                            N,
                                            K,
                                            stride_a,
                                            stride_b,
                                            stride_c,
                                            scale_granularity_m,
                                            scale_granularity_n,
                                            scale_granularity_k,
                                            scale_A_ptr,
                                            scale_B_ptr);

    return;
}

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void reference_batched_gemm_gpu(if_select_t<ADataType_, tf32_t, float, ADataType_>* a_ptr,
                                if_select_t<BDataType_, tf32_t, float, BDataType_>* b_ptr,
                                CDataType* c_ptr,
                                index_t M,
                                index_t N,
                                index_t K,
                                index_t stride_a,
                                index_t stride_b,
                                index_t stride_c,
                                index_t batch_stride_A,
                                index_t batch_stride_B,
                                index_t batch_stride_C,
                                index_t batch_count)
{
    using ADataTypeBuf = if_select_t<ADataType_, tf32_t, float, ADataType_>;
    using BDataTypeBuf = if_select_t<BDataType_, tf32_t, float, BDataType_>;

    using ADataTypeCompute = ADataType_;
    using BDataTypeCompute = BDataType_;

    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for(index_t batch_id = 0; batch_id < batch_count; ++batch_id)
    {
        ADataTypeBuf* d_ATemp = a_ptr + batch_id * batch_stride_A;
        BDataTypeBuf* d_BTemp = b_ptr + batch_id * batch_stride_B;
        CDataType* d_CTemp    = c_ptr + batch_id * batch_stride_C;
        naive_gemm_kernel<ADataTypeCompute,
                          BDataTypeCompute,
                          AccDataType,
                          CDataType,
                          LayoutA,
                          LayoutB,
                          LayoutC><<<numBlocks, numThreadsPerBlock>>>(
            d_ATemp, d_BTemp, d_CTemp, M, N, K, stride_a, stride_b, stride_c);
    }

    return;
}

} // namespace ck_tile
