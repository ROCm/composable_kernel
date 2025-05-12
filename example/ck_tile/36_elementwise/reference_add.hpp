// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename YDataType>
CK_TILE_HOST void reference_add(const HostTensor<XDataType>& xa,
                                const HostTensor<XDataType>& xb,
                                HostTensor<YDataType>& y)
{
    // auto f = [&](auto m) {
    //     const int N = xa_m_n.mDesc.get_lengths()[1];

    //     for(int n = 0; n < N; ++n)
    //     {
    //         y_m_n(m, n) = ck_tile::type_convert<YDataType>(xa_m_n(m, n)) +
    //                       ck_tile::type_convert<YDataType>(xb_m_n(m, n));
    //     }
    // };

    // make_ParallelTensorFunctor(f,
    //                            y_m_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
    y.ForEach([&](auto& self, auto i) {
        self(i) = ck_tile::type_convert<YDataType>(xa(i)) +
                  ck_tile::type_convert<YDataType>(xb(i));
    });
}


template <typename XDataType, typename YDataType>
    CK_TILE_HOST void reference_add_3D(const HostTensor<XDataType>& xa_b_m_n,
                                    const HostTensor<XDataType>& xb_b_m_n,
                                    HostTensor<YDataType>& y_b_m_n)
    {
        auto f = [&](auto bm_idx) {
            // Calculate batch and m indices
            // const int B = xa_b_m_n.mDesc.get_lengths()[0];
            const int M = xa_b_m_n.mDesc.get_lengths()[1];
            const int N = xa_b_m_n.mDesc.get_lengths()[2];
     
            // Convert flat bm_idx to separate b and m indices
            const int b = bm_idx / M;
            const int m = bm_idx % M;
     
            // Process each element in the N dimension
            for(int n = 0; n < N; ++n)
            {
                y_b_m_n(b, m, n) = ck_tile::type_convert<YDataType>(xa_b_m_n(b, m, n)) +
                                  ck_tile::type_convert<YDataType>(xb_b_m_n(b, m, n));
            }
     
            
        };
     
        // Get total elements to process in the B and M dimensions
        const int total_bm = y_b_m_n.mDesc.get_lengths()[0] * y_b_m_n.mDesc.get_lengths()[1];
        // Parallelize computation across the flattened B×M space
        make_ParallelTensorFunctor(f, total_bm)(std::thread::hardware_concurrency());
     
        
    }

} // namespace ck_tile
