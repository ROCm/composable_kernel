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
using INT4 = ck_tile::pk_int4_t;

float gemm(const gemm_traits& t, const ck_tile::GemmHostArgs& a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16_fp16_fp16") == 0)
    {
        if(t.IsRRRLayout())
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
        else if(t.IsRCRLayout())
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
        else if(t.IsCRRLayout())
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
        else if(t.IsCCRLayout())
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
    else if(t.data_type.compare("bf16_bf16_bf16") == 0)
    {
        if(t.IsRRRLayout())
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
        else if(t.IsRCRLayout())
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
        else if(t.IsCRRLayout())
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
        else if(t.IsCCRLayout())
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
    else if(t.data_type.compare("fp8_fp8_fp16") == 0)
    {
        if(t.IsRRRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Row,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            if(a.M < 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Row,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsRCRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsCRRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Col,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Col,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsCCRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      FP8,       FP8,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else
        {
            throw std::runtime_error("Wrong! ColumnMajor layout not supported for C Matrix!\n");
        }
    }
    else if(t.data_type.compare("bf8_bf8_fp16") == 0)
    {
        if(t.IsRRRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Row,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            if(a.M < 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Row,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsRCRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsCRRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Col,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Col,     Row,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else if(t.IsCCRLayout())
        {
            if(a.M > 512)
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
            else
            {
                // clang-format off
                //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
                return gemm_<gemm_traits_<      BF8,       BF8,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
                // clang-format on
            }
        }
        else
        {
            throw std::runtime_error("Wrong! ColumnMajor layout not supported for C Matrix!\n");
        }
    }
    else if(t.data_type.compare("fp16_pk_int4_t_fp16") == 0)
    {
        if(t.IsRCRLayout())
        {
            // Currently only CompV3 pipeline supports pk_int4_t
            // clang-format off
            //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
            return gemm_<gemm_traits_<     FP16,      INT4,        FP32,      FP16,     Row,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
            // clang-format on
        }
        else if(t.IsCCRLayout())
        {
            // Currently only CompV3 pipeline supports pk_int4_t
            // clang-format off
            //                        ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,  PadM,  PadN,  PadK
            return gemm_<gemm_traits_<     FP16,      INT4,        FP32,      FP16,     Col,     Col,     Row,    256,    256,     64,      2,      2,      1,          32,          32,          16, false, false, false>>(a, s);
            // clang-format on
        }
        else
        {
            throw std::runtime_error("Wrong! Layouts not supported!\n");
        }
    }
    else
    {
        throw std::runtime_error("Wrong! DataTypes not supported!\n");
    }
    return 1.0f;
}
