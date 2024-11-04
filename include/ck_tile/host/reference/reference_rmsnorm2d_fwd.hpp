// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename XDataType,
          typename GammaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename InvRmsDataType>
void reference_rmsnorm2d_fwd(const HostTensor<XDataType>& x_m_n,
                             const HostTensor<GammaDataType>& gamma_n,
                             HostTensor<YDataType>& y_m_n,
                             HostTensor<InvRmsDataType>& invRms_m,
                             ComputeDataType epsilon)
{
    auto rmsnorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        ComputeDataType mean_square = 0;
        ComputeDataType divisor     = 0;

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            mean_square += x * x;
        }

        mean_square = mean_square / N;
        divisor = ck_tile::type_convert<ComputeDataType>(1) / ck_tile::sqrt(mean_square + epsilon);

        if constexpr(!std::is_same_v<InvRmsDataType, ck_tile::null_type>)
            invRms_m(m) = ck_tile::type_convert<InvRmsDataType>(divisor);

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
            auto y                = x * divisor * gamma;
            y_m_n(m, n)           = ck_tile::type_convert<YDataType>(y);
        }
    };

    make_ParallelTensorFunctor(rmsnorm2d_fwd_func, invRms_m.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
