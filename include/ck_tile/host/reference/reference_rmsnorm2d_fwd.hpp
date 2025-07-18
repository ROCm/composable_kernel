// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Note: for simplicity, each functor only care about single M
struct reference_rmsnorm2d_default_epilogue
{
    template <typename OutDataType, typename AccDataType>
    void operator()(int m, HostTensor<OutDataType>& o, const HostTensor<AccDataType>& acc)
    {
        const int N = acc.mDesc.get_lengths()[1];
        for(int n = 0; n < N; ++n)
        {
            o(m, n) = ck_tile::type_convert<OutDataType>(acc(m, n));
        }
    }

    template <typename OutDataType, typename AccDataType>
    auto operator()(int m, const HostTensor<AccDataType>& acc)
    {
        HostTensor<OutDataType> o(acc.get_lengths(), acc.get_strides());
        operator()(m, o, acc);
        return o;
    }
};

template <typename XDataType,
          typename GammaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename InvRmsDataType,
          typename UnquantYDataType,
          typename Epilogue = reference_rmsnorm2d_default_epilogue>
void reference_rmsnorm2d_fwd(const HostTensor<XDataType>& x_m_n,
                             const HostTensor<GammaDataType>& gamma_n,
                             HostTensor<YDataType>& y_m_n,
                             HostTensor<InvRmsDataType>& invRms_m,
                             HostTensor<UnquantYDataType>& unquant_y_m_n,
                             ComputeDataType epsilon,
                             Epilogue epilogue_functor = {})
{
    constexpr int elements_per_thread = 5;
    constexpr int warp_size = 64;

    auto rmsnorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        const int num_threads = N / elements_per_thread;
        const int num_warps   = (num_threads + warp_size - 1) / warp_size;

        // Step 1: per-thread local partial sum
        std::vector<ComputeDataType> thread_partial_sum(num_threads, 0);

        for(int tid = 0; tid < num_threads; ++tid)
        {
            for(int i = 0; i < elements_per_thread; ++i)
            {
                int n = tid * elements_per_thread + i;
                if(n < N)
                {
                    ComputeDataType x = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
                    thread_partial_sum[tid] += x * x;
                }
            }
        }

        // Step 2: intra-warp tree reduction
        std::vector<ComputeDataType> warp_partial_sum(num_warps, 0);
        for(int w = 0; w < num_warps; ++w)
        {
            ComputeDataType warp_sum = 0;
            for(int t = 0; t < warp_size; ++t)
            {
                int tid = w * warp_size + t;
                if(tid < num_threads)
                    warp_sum += thread_partial_sum[tid];
            }
            warp_partial_sum[w] = warp_sum;
        }

        // Step 3: cross-warp reduction
        // ComputeDataType total_sum = 0;
        // for(int w = 0; w < num_warps; ++w)
        //     total_sum += warp_partial_sum[w];
        // Step 3: cross-warp tree reduction
        ComputeDataType total_sum = 0;
        {
            std::vector<ComputeDataType> buffer = warp_partial_sum; // copy for reduction
            int size = static_cast<int>(buffer.size());
            while(size > 1)
            {
                int half = size / 2;
                for(int i = 0; i < half; ++i)
                {
                    buffer[i] += buffer[i + half];
                }
                if(size % 2 == 1) // handle odd case
                {
                    buffer[0] += buffer[size - 1];
                    size = half + 1;
                }
                else
                {
                    size = half;
                }
            }
            total_sum = buffer[0];
        }


        ComputeDataType mean_square = total_sum / N;
        ComputeDataType divisor     = ck_tile::type_convert<ComputeDataType>(1) /
                                      ck_tile::sqrt(mean_square + epsilon);

        if constexpr(!std::is_same_v<InvRmsDataType, ck_tile::null_type>)
            invRms_m(m) = ck_tile::type_convert<InvRmsDataType>(divisor);

        // Compute y = x * gamma * inv_rms
        HostTensor<ComputeDataType> acc(x_m_n.get_lengths(), x_m_n.get_strides());
        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            YDataType tmp = ck_tile::type_convert<YDataType>(x*divisor);
            ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
            ComputeDataType tmp1 = ck_tile::type_convert<ComputeDataType>(tmp) * gamma;
            acc(m, n)             = tmp1;
        }

        if constexpr(!std::is_same_v<UnquantYDataType, ck_tile::null_type>)
        {
            epilogue_functor(m, unquant_y_m_n, y_m_n, acc);
        }
        else
        {
            epilogue_functor(m, y_m_n, acc);
        }
    };

    make_ParallelTensorFunctor(rmsnorm2d_fwd_func, invRms_m.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());
}

} // namespace ck_tile
