// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using FP32 = float;
using FP16 = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;

float gemm(const gemm_traits& t, const ck_tile::GemmHostArgs& a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        if(t.is_a_rowmajor && t.is_b_rowmajor && t.is_c_rowmajor)
        {
            if(a.M > 512)
            {
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<FP16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
                return gemm_<gemm_traits_<BF16,
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
}
