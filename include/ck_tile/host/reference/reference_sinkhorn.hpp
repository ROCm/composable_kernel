// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ComputeDataType>
void sinkhorn_knopp_ref_single_iter(HostTensor<ComputeDataType>& c_n_n,
                                    HostTensor<ComputeDataType>& acc_n)
{
    const index_t input_n = acc_n.get_length(0);

    // Sum and scale rowwise
    for(index_t i = 0; i < input_n; ++i)
    {
        acc_n(i) = 0;
        for(index_t j = 0; j < input_n; ++j)
        {
            acc_n(i) += c_n_n(i, j);
        }
        for(index_t j = 0; j < input_n; ++j)
        {
            c_n_n(i, j) /= acc_n(i);
        }
    }

    // Repeat columnwise
    for(index_t i = 0; i < input_n; ++i)
    {
        acc_n(i) = 0;
        for(index_t j = 0; j < input_n; ++j)
        {
            acc_n(i) += c_n_n(j, i);
        }
        for(index_t j = 0; j < input_n; ++j)
        {
            c_n_n(j, i) /= acc_n(i);
        }
    }
}

template <typename XDataType, typename ComputeDataType, typename YDataType>
void sinkhorn_knopp_ref(const HostTensor<XDataType>& x_n_n,
                        HostTensor<YDataType>& y_n_n,
                        const int n_iter)
{
    const index_t input_n = x_n_n.get_length(0);
    HostTensor<ComputeDataType> c_n_n({input_n, input_n}, {1, input_n});
    HostTensor<ComputeDataType> acc_n({input_n}, {1});

    // First apply exp to make input nonnegative
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            c_n_n(i, j) = exp(type_convert<ComputeDataType>(x_n_n(i, j)));
        }
    }

    // Iterate normalization on rows and columns
    for(auto it = 0; it < n_iter; ++it)
    {
        sinkhorn_knopp_ref_single_iter(c_n_n, c_n_n);
    }

    // Copy and cast to output type
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            y_n_n(i, j) = type_convert<YDataType>(c_n_n(i, j));
        }
    }
}

} // namespace ck_tile
