// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_basic.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using FP32 = float;
using FP16 = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t M_Tile_,
          ck_tile::index_t N_Tile_,
          ck_tile::index_t K_Tile_,
          ck_tile::index_t M_Warp_,
          ck_tile::index_t N_Warp_,
          ck_tile::index_t K_Warp_,
          ck_tile::index_t M_Warp_Tile_,
          ck_tile::index_t N_Warp_Tile_,
          ck_tile::index_t K_Warp_Tile_,
          bool kPadM_,
          bool kPadN_,
          bool kPadK_>
using trait_ = gemm_traits_<ADataType_,
                            BDataType_,
                            AccDataType_,
                            CDataType_,
                            ALayout_,
                            BLayout_,
                            CLayout_,
                            M_Tile_,
                            N_Tile_,
                            K_Tile_,
                            M_Warp_,
                            N_Warp_,
                            K_Warp_,
                            M_Warp_Tile_,
                            N_Warp_Tile_,
                            K_Warp_Tile_,
                            kPadM_,
                            kPadN_,
                            kPadK_>;

float gemm(const gemm_traits& t, const ck_tile::GemmHostArgs& a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound RR
                std::cout << "fp16 comp\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Row,
                                    Row,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound RR
                std::cout << "fp16 mem\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Row,
                                    Row,
                                    Row,
                                    128,
                                    32,
                                    64,
                                    4,
                                    1,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound RC
                std::cout << "fp16 comp RC\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Row,
                                    Col,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound RC
                std::cout << "fp16 mem RC\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Row,
                                    Col,
                                    Row,
                                    128,
                                    32,
                                    64,
                                    4,
                                    1,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound CR
                std::cout << "fp16 comp CR\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Col,
                                    Row,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound CR
                std::cout << "fp16 mem CR\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Col,
                                    Row,
                                    Row,
                                    128,
                                    128,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound CC
                std::cout << "fp16 comp CC\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Col,
                                    Col,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound CC
                std::cout << "fp16 mem CC\n";
                return gemm_<trait_<FP16,
                                    FP16,
                                    FP32,
                                    FP16,
                                    Col,
                                    Col,
                                    Row,
                                    128,
                                    128,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else
        {
            throw std::runtime_error("Wrong! ColumnMajor layout not supported for C Matrix!\n");
        }
    }
    else if(t.data_type.compare("bf16") == 0)
    {
        if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound RR
                std::cout << "bf16 comp\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Row,
                                    Row,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound RR
                std::cout << "bf16 mem\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Row,
                                    Row,
                                    Row,
                                    128,
                                    32,
                                    64,
                                    4,
                                    1,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound RC
                std::cout << "bf16 comp RC\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Row,
                                    Col,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound RC
                std::cout << "bf16 mem RC\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Row,
                                    Col,
                                    Row,
                                    128,
                                    32,
                                    64,
                                    4,
                                    1,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound CR
                std::cout << "bf16 comp CR\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Col,
                                    Row,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound CR
                std::cout << "bf16 mem CR\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Col,
                                    Row,
                                    Row,
                                    128,
                                    128,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // universal gemm compute bound CC
                std::cout << "bf16 comp CC\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Col,
                                    Col,
                                    Row,
                                    256,
                                    256,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    16,
                                    false,
                                    false,
                                    false>>(a, s);
            }
            else
            {
                // universal gemm memory bound CC
                std::cout << "bf16 mem CC\n";
                return gemm_<trait_<BF16,
                                    BF16,
                                    FP32,
                                    BF16,
                                    Col,
                                    Col,
                                    Row,
                                    128,
                                    128,
                                    32,
                                    2,
                                    2,
                                    1,
                                    32,
                                    32,
                                    8,
                                    false,
                                    false,
                                    false>>(a, s);
            }
        }
        else
        {
            throw std::runtime_error("Wrong! ColumnMajor layout not supported for C Matrix!\n");
        }
    }
    else
    {
        throw std::runtime_error("Wrong! DataTypes not supported!\n");
    }

    return 1.0f;
}
