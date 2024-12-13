// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"

namespace ck_tile {

struct Problem
{
    CK_TILE_HOST Problem() = default;
    CK_TILE_HOST Problem(
        index_t M_, index_t N_, index_t K_, index_t stride_A_, index_t stride_B_, index_t stride_C_)
        : M(M_), N(N_), K(K_), stride_A(stride_A_), stride_B(stride_B_), stride_C(stride_C_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
};

struct GemmHostArgs : public Problem
{
    CK_TILE_HOST GemmHostArgs() = default;
    CK_TILE_HOST GemmHostArgs(const void* a_ptr_,
                              const void* b_ptr_,
                              void* c_ptr_,
                              index_t k_batch_,
                              index_t M_,
                              index_t N_,
                              index_t K_,
                              index_t stride_A_,
                              index_t stride_B_,
                              index_t stride_C_)
        : Problem(M_, N_, K_, stride_A_, stride_B_, stride_C_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    index_t k_batch;
};

struct BatchedGemmHostArgs : public ck_tile::GemmHostArgs
{
    CK_TILE_HOST BatchedGemmHostArgs() = default;
    CK_TILE_HOST BatchedGemmHostArgs(const void* a_ptr_,
                                     const void* b_ptr_,
                                     void* c_ptr_,
                                     ck_tile::index_t k_batch_,
                                     ck_tile::index_t M_,
                                     ck_tile::index_t N_,
                                     ck_tile::index_t K_,
                                     ck_tile::index_t stride_A_,
                                     ck_tile::index_t stride_B_,
                                     ck_tile::index_t stride_C_,
                                     ck_tile::index_t batch_stride_A_,
                                     ck_tile::index_t batch_stride_B_,
                                     ck_tile::index_t batch_stride_C_,
                                     ck_tile::index_t batch_count_)
        : GemmHostArgs(
              a_ptr_, b_ptr_, c_ptr_, k_batch_, M_, N_, K_, stride_A_, stride_B_, stride_C_),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          batch_stride_C(batch_stride_C_),
          batch_count(batch_count_)
    {
    }

    ck_tile::index_t batch_stride_A;
    ck_tile::index_t batch_stride_B;
    ck_tile::index_t batch_stride_C;
    ck_tile::index_t batch_count;
};

} // namespace ck_tile
