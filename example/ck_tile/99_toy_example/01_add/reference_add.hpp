// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename YDataType>
CK_TILE_HOST void
reference_add(const HostTensor<XDataType>& xa_m_n, const HostTensor<XDataType>& xb_m_n, HostTensor<YDataType>& y_m_n)
{
    auto f = [&](auto m) {
        const int N = xa_m_n.mDesc.get_lengths()[1];

        for(int n = 0; n < N; ++n)
        {
            y_m_n(m, n) = ck_tile::type_convert<YDataType>(xa_m_n(m, n)) + ck_tile::type_convert<YDataType>(xb_m_n(m, n));
        }
    };

    make_ParallelTensorFunctor(f, y_m_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}

} // namespace ck_tile
