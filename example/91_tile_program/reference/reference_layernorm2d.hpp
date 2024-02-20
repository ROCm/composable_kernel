// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"


template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename MeanDataType,
          typename InvStdDataType>
void reference_layernorm2d_fwd(const Tensor<XDataType>& x_m_n,
                               const Tensor<GammaDataType>& gamma_n,
                               const Tensor<BetaDataType>& beta_n,
                               Tensor<YDataType>& y_m_n,
                               Tensor<MeanDataType>& mean_m,
                               Tensor<InvStdDataType>& invStd_m,
                               ComputeDataType epsilon)
{
    auto layernorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.GetLengths()[1];

        int count                = 0;
        ComputeDataType mean     = 0;
        ComputeDataType variance = 0;
        ComputeDataType divisor  = 0;

        for(int n = 0; n < N; ++n)
        {
            ++count;
            ComputeDataType x     = ck::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType delta = x - mean;
            mean += delta / count;
            ComputeDataType delta2 = x - mean;
            variance += delta * delta2;
        }

        // actual variance
        variance = variance / count;
        divisor  = ck::type_convert<ComputeDataType>(1) / ck::math::sqrt(variance + epsilon);

        mean_m(m)   = ck::type_convert<MeanDataType>(mean);
        invStd_m(m) = ck::type_convert<InvStdDataType>(divisor);

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType gamma = ck::type_convert<ComputeDataType>(gamma_n(n));
            ComputeDataType beta  = ck::type_convert<ComputeDataType>(beta_n(n));
            auto y                = (x - mean) * divisor;
            y                     = y * gamma + beta;

            y_m_n(m, n) = ck::type_convert<YDataType>(y);
        }
    };

    make_ParallelTensorFunctor(layernorm2d_fwd_func,
                               mean_m.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}
