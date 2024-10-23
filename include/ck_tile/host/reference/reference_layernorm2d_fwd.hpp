// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename MeanDataType,
          typename InvStdDataType>
void reference_layernorm2d_fwd(const HostTensor<XDataType>& x_m_n,
                               const HostTensor<GammaDataType>& gamma_n,
                               const HostTensor<BetaDataType>& beta_n,
                               HostTensor<YDataType>& y_m_n,
                               HostTensor<MeanDataType>& mean_m,
                               HostTensor<InvStdDataType>& invStd_m,
                               ComputeDataType epsilon)
{
    auto layernorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        int count                = 0;
        ComputeDataType mean     = 0;
        ComputeDataType variance = 0;
        ComputeDataType divisor  = 0;

        for(int n = 0; n < N; ++n)
        {
            ++count;
            ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType delta = x - mean;
            mean += delta / count;
            ComputeDataType delta2 = x - mean;
            variance += delta * delta2;
        }

        // actual variance
        variance = variance / count;
        divisor  = ck_tile::type_convert<ComputeDataType>(1) / ck_tile::sqrt(variance + epsilon);

        if constexpr(!std::is_same_v<MeanDataType, ck_tile::null_type>)
            mean_m(m) = ck_tile::type_convert<MeanDataType>(mean);

        if constexpr(!std::is_same_v<InvStdDataType, ck_tile::null_type>)
            invStd_m(m) = ck_tile::type_convert<InvStdDataType>(divisor);

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
            ComputeDataType beta  = ck_tile::type_convert<ComputeDataType>(beta_n(n));
            auto y                = (x - mean) * divisor;
            y                     = y * gamma + beta;

            y_m_n(m, n) = ck_tile::type_convert<YDataType>(y);
        }
    };

    make_ParallelTensorFunctor(layernorm2d_fwd_func,
                               mean_m.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}
} // namespace ck_tile
