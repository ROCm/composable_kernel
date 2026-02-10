// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename ComputeDataType, typename YDataType>
void sinkhorn_knopp_naive_ref(const HostTensor<XDataType>& x_n_n,
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

    // Copy and cast to output type
    for(index_t i = 0; i < input_n; ++i)
    {
        for(index_t j = 0; j < input_n; ++j)
        {
            y_n_n(i, j) = type_convert<YDataType>(c_n_n(i, j));
        }
    }
}

// Log-space implementation for Sinkhorn-Knopp
template <typename XDataType, typename ComputeDataType, typename YDataType>
void sinkhorn_knopp_lse_ref(const HostTensor<XDataType>& x_n_n,
                            HostTensor<YDataType>& y_n_n,
                            const int n_iter)
{
    const index_t input_n = x_n_n.get_length(0);
    HostTensor<ComputeDataType> c_n_n({input_n, input_n}, {1, input_n});
    HostTensor<ComputeDataType> log_u({input_n}, {1});
    HostTensor<ComputeDataType> log_v({input_n}, {1});

    ck_tile::FillConstant<ComputeDataType>{0}(log_u);
    ck_tile::FillConstant<ComputeDataType>{0}(log_v);

    for(auto it = 0; it < n_iter; ++it)
    {
        for(auto i = 0; i < input_n; ++i)
        {
            // For each row:
            // 1. Add the corresponding column scaling to each value and compute the max
            ComputeDataType max_value = 0.0;
            for(auto j = 0; j < input_n; ++j)
            {
                c_n_n(i, j) = type_convert<ComputeDataType>(x_n_n(i, j)) + log_v(j);
                if(c_n_n(i, j) > max_value)
                {
                    max_value = c_n_n(i, j);
                }
            }

            // 2. exponentiate and compute the sum of the row
            ComputeDataType sumexp = 0.0;
            for(auto j = 0; j < input_n; ++j)
            {
                sumexp += ck_tile::exp(c_n_n(i, j) - max_value);
            }

            // 3. Update the row scale factors
            log_u(i) = -(max_value + ck_tile::log(sumexp));
        }

        for(auto j = 0; j < input_n; ++j)
        {
            // For each column:
            // 1. Add the corresponding row scaling to each value and compute the max
            ComputeDataType max_value = 0.0;
            for(auto i = 0; i < input_n; ++i)
            {
                c_n_n(i, j) = type_convert<ComputeDataType>(x_n_n(i, j)) + log_u(i);
                if(c_n_n(i, j) > max_value)
                {
                    max_value = c_n_n(i, j);
                }
            }

            // 2. exponentiate and compute the sum of the row
            ComputeDataType sumexp = 0.0;
            for(auto i = 0; i < input_n; ++i)
            {
                sumexp += ck_tile::exp(c_n_n(i, j) - max_value);
            }

            // 3. Update the row scale factors
            log_v(j) = -(max_value + ck_tile::log(sumexp));
        }
    }

    // Apply the final scaling factors and exponentiate
    for(auto i = 0; i < input_n; ++i)
    {
        for(auto j = 0; j < input_n; ++j)
        {
            c_n_n(i, j) =
                ck_tile::exp(type_convert<ComputeDataType>(x_n_n(i, j)) + log_u(i) + log_v(j));
        }
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
