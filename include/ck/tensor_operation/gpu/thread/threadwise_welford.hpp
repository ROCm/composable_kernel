// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

// Assume
//  1) XDesc is known at compile-time
//  2) MeanVarDesc is known at compile-time
//  3) XBuffer is static buffer
//  4) MeanBuffer is static buffer
//  5) VarBuffer is static buffer
template <typename T, typename XThreadDesc_M_K, typename MeanVarThreadDesc_M>
struct ThreadwiseWelford
{
    static constexpr auto x_thread_desc_m_k      = XThreadDesc_M_K{};
    static constexpr auto mean_var_thread_desc_m = MeanVarThreadDesc_M{};

    static constexpr auto thread_x_length_m        = x_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto thread_x_length_k        = x_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto thread_mean_var_length_m = mean_var_thread_desc_m.GetLength(Number<0>{});

    static_assert(thread_x_length_m == thread_mean_var_length_m,
                  "lengths of source and mean/var buffer must match!");

    static_assert(thread_x_length_k > 0, "lengths of k must greater than 0!");

    __device__ constexpr ThreadwiseWelford() : cur_count_(0), max_count_(0) {}

    __device__ inline void Update(T& mean, T& var, T x)
    {
        using ck::math::isnan;

        if(cur_count_ < max_count_)
        {
            if(isnan(x))
            {
                mean = x;
                var  = x;
            }
            else
            {
                ++cur_count_;
                T delta = x - mean;
                mean += delta / cur_count_;
                T delta2 = x - mean;
                var += delta * delta2;
            }
        }
    }

    template <typename XBufferType, typename MeanBufferType, typename VarBufferType>
    __device__ void
    Run(const XBufferType& x_buf_m_k, MeanBufferType& mean_buf_m, VarBufferType& var_buf_m)
    {
        // FIXME - Better naming for var_buf_m
        static_for<0, thread_x_length_m, 1>{}([&](auto iM) {
            constexpr index_t out_offset = mean_var_thread_desc_m.CalculateOffset(make_tuple(iM));

            // TODO - tail case
            static_for<0, thread_x_length_k, 1>{}([&](auto iK) {
                constexpr auto in_offset = x_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                Update(mean_buf_m(Number<out_offset>{}),
                       var_buf_m(Number<out_offset>{}),
                       x_buf_m_k[Number<in_offset>{}]);
            });
        });
    };

    int cur_count_;
    int max_count_;
};

} // namespace ck
