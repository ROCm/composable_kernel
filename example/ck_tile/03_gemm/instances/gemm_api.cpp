// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using FP32 = float;
using FP16 = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using FP8  = ck_tile::fp8_t;
using BF8  = ck_tile::bf8_t;

float gemm(const gemm_traits& t, const ck_tile::GemmHostArgs& a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Row,     Row,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Row,     Row,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Row,     Col,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Col,     Row,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Col,     Row,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     FP16,      FP16,        FP32,      FP16,     Col,     Col,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
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
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Row,     Row,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Row,     Row,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Row,     Col,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Row,     Col,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Col,     Row,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Col,     Row,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Col,     Col,     Row,    256,    256,     32,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<     BF16,      BF16,        FP32,      BF16,     Col,     Col,     Row,    128,    128,     32,      2,      2,      1,          32,          32,           8, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else
        {
            throw std::runtime_error("Wrong! ColumnMajor layout not supported for C Matrix!\n");
        }
    }
    else if(t.data_type.compare("fp8") == 0)
    {
        if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Row,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Row,     Row,     Row,    128,    128,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Row,     Col,     Row,    128,    128,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Col,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Col,     Row,     Row,    128,    128,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(!t.is_a_rowmajor && !t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,       FP8,     Col,     Col,     Row,    128,    128,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
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
}
