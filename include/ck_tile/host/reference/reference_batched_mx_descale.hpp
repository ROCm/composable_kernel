// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename InDataType,
          typename ScaleDataType,
          typename OutDataType,
          typename ComputeDataType>
CK_TILE_HOST HostTensor<OutDataType>
reference_batched_mx_descale(const HostTensor<InDataType>& a_b_m_k,
                             const HostTensor<ScaleDataType>& scales_b_m_ks,
                             const std::size_t scale_granularity)
{
    const std::size_t B = a_b_m_k.get_length(0);
    const std::size_t M = a_b_m_k.get_length(1);
    const std::size_t K = a_b_m_k.get_length(2);

    HostTensor<ComputeDataType> a_b_m_k_scaled(a_b_m_k.get_lengths());

    auto f = [&](auto batch) {
        constexpr index_t packed_size = ck_tile::numeric_traits<InDataType>::PackedSize;

        for(std::size_t m = 0; m < M; ++m)
        {
            for(std::size_t k = 0; k < K; k += packed_size)
            {
                const auto scale = ck_tile::type_convert<ComputeDataType>(
                    scales_b_m_ks(batch, m, k / scale_granularity));

                if constexpr(std::is_same_v<InDataType, pk_fp4_t>)
                {
                    auto a_f4x2  = a_b_m_k(batch, m, k);
                    auto a_f4_lo = ck_tile::type_convert<ComputeDataType>(
                        a_f4x2.template unpack<>(number<0>{}));
                    auto a_f4_hi = ck_tile::type_convert<ComputeDataType>(
                        a_f4x2.template unpack<>(number<1>{}));

                    a_b_m_k_scaled(batch, m, k)     = a_f4_lo * scale;
                    a_b_m_k_scaled(batch, m, k + 1) = a_f4_hi * scale;
                }
                else
                {
                    a_b_m_k_scaled(batch, m, k) =
                        ck_tile::type_convert<ComputeDataType>(a_b_m_k(batch, m, k)) * scale;
                }
            }
        }
    };
    make_ParallelTensorFunctor(f, B)(std::thread::hardware_concurrency());

    return a_b_m_k_scaled;
}

} // namespace ck_tile
