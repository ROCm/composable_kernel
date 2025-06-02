// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {
template <typename ADataType, typename BDataType, typename ComputeDataType, typename ElementOp>
CK_TILE_HOST void reference_unary_elementwise(const HostTensor<ADataType>& a,
                                              HostTensor<BDataType>& b,
                                              ElementOp element_op)
{
    // TODO: imeplement gpu version reference function
    auto f = [&](auto i) {
        auto v_a   = type_convert<ComputeDataType>(a.mData[i]);
        auto v_b   = element_op(v_a);
        b.mData[i] = ck_tile::type_convert<BDataType>(v_b);
    };

    make_ParallelTensorFunctor(f, b.get_element_space_size())(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ElementOp>
CK_TILE_HOST void reference_binary_elementwise(const HostTensor<ADataType>& a,
                                               const HostTensor<BDataType>& b,
                                               HostTensor<CDataType>& c,
                                               ElementOp element_op)
{
    // TODO: imeplement gpu version reference function
    auto f = [&](auto i) {
        auto v_a   = type_convert<ComputeDataType>(a.mData[i]);
        auto v_b   = type_convert<ComputeDataType>(b.mData[i]);
        auto v_c   = element_op(v_a, v_b);
        c.mData[i] = ck_tile::type_convert<CDataType>(v_c);
    };

    make_ParallelTensorFunctor(f, c.get_element_space_size())(std::thread::hardware_concurrency());
}

template <typename ADataType, typename BDataType>
void reference_transpose_elementwise(const HostTensor<ADataType>& a,
                                     HostTensor<BDataType>& b)
{
    ck_tile::index_t M = static_cast<ck_tile::index_t>(a.mDesc.get_lengths()[0]);
    ck_tile::index_t N = static_cast<ck_tile::index_t>(a.mDesc.get_lengths()[1]);

    // Ensure the b tensor is sized correctly for N x M
    if (static_cast<ck_tile::index_t>(b.mDesc.get_lengths()[0]) != N || static_cast<ck_tile::index_t>(b.mDesc.get_lengths()[1]) != M) {
        throw std::runtime_error("Output tensor b has incorrect dimensions for transpose.");
    }

    auto f = [&](auto i, auto j) {
        auto v_a   = a(i,j);
        b(j,i) = ck_tile::type_convert<BDataType>(v_a);
    };

    make_ParallelTensorFunctor(f, M, N)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
