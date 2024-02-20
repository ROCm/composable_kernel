// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename XDataType, typename ComputeDataType, typename MeanDataType, typename VarDataType>
void reference_variance(const Tensor<XDataType>& x_m_n,
                        Tensor<MeanDataType>& mean_m,
                        Tensor<VarDataType>& var_m)
{
    auto welford_func = [&](auto m) {
        const int N = x_m_n.mDesc.GetLengths()[1];

        int count                = 0;
        ComputeDataType mean     = 0;
        ComputeDataType variance = 0;

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

        mean_m(m) = ck::type_convert<MeanDataType>(mean);
        var_m(m)  = ck::type_convert<VarDataType>(variance);
    };

    make_ParallelTensorFunctor(welford_func,
                               mean_m.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}
