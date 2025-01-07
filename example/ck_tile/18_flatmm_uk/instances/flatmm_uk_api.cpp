// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "flatmm_uk.hpp"
#include "flatmm_uk_api.hpp"
#include "ck_tile/ops/flatmm_uk.hpp"
#include <iostream>

template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

// do not the define of this tepmlate function inside the _api.cpp, otherwise will block make -j
template <typename Ts_>
float flatmm_uk_(const ck_tile::stream_config& s_, flatmm_uk_args_ a_)
{
    printf("[FF] ======= fused_moegemm_() ======= \n \tget moe arg in a_ <flatmm_uk_args>, get "
           "config in Ts_\n");
    using t_traits = ck_tile::FusedMoeGemmTraits<true, true, 1 /*atomic*/>;
    using t_shape  = ck_tile::
        FlatmmShape<typename Ts_::BlockTile, typename Ts_::WavePerBlock, typename Ts_::WaveTile>;
    printf("[FF] --- flatmm_uk_(): <FlatmmShape> --- \n");
    printf("[FF] t_shape::BlockSize = %d\n", static_cast<uint32_t>(t_shape::BlockSize));
    printf("[FF] t_shape::NumWaves = %d\n", static_cast<uint32_t>(t_shape::NumWaves));
    printf("[FF] --------- \n");
    printf("[FF] t_shape::Block_M = %d\n", static_cast<uint32_t>(t_shape::Block_M));
    printf("[FF] t_shape::Block_N = %d\n", static_cast<uint32_t>(t_shape::Block_N));
    printf("[FF] t_shape::Block_K = %d\n", static_cast<uint32_t>(t_shape::Block_K));
    printf("[FF] t_shape::WavePerBlock_M = %d\n", static_cast<uint32_t>(t_shape::WavePerBlock_M));
    printf("[FF] t_shape::WavePerBlock_N = %d\n", static_cast<uint32_t>(t_shape::WavePerBlock_N));
    printf("[FF] t_shape::WavePerBlock_K = %d\n", static_cast<uint32_t>(t_shape::WavePerBlock_K));
    printf("[FF] t_shape::Wave_M = %d\n", static_cast<uint32_t>(t_shape::Wave_M));
    printf("[FF] t_shape::Wave_N = %d\n", static_cast<uint32_t>(t_shape::Wave_N));
    printf("[FF] t_shape::Wave_K = %d\n", static_cast<uint32_t>(t_shape::Wave_K));
    printf("[FF] t_shape::ThreadPerBlock_M = %d\n",
           static_cast<uint32_t>(t_shape::ThreadPerBlock_M));
    printf("[FF] t_shape::ThreadPerBlock_N = %d\n",
           static_cast<uint32_t>(t_shape::ThreadPerBlock_N));
    printf("[FF] t_shape::ThreadPerBlock_K = %d\n",
           static_cast<uint32_t>(t_shape::ThreadPerBlock_K));
    printf("[FF] t_shape::Repeat_M = %d\n", static_cast<uint32_t>(t_shape::Repeat_M));
    printf("[FF] t_shape::Repeat_N = %d\n", static_cast<uint32_t>(t_shape::Repeat_N));
    printf("[FF] t_shape::Repeat_K = %d\n", static_cast<uint32_t>(t_shape::Repeat_K));
    printf("[FF] t_shape::Block_Mr = %d\n", static_cast<uint32_t>(t_shape::Block_Mr));
    printf("[FF] t_shape::Block_Nr = %d\n", static_cast<uint32_t>(t_shape::Block_Nr));
    printf("[FF] t_shape::Block_Kr = %d\n", static_cast<uint32_t>(t_shape::Block_Kr));
    printf("[FF] t_shape::Block_W  = %d\n", static_cast<uint32_t>(t_shape::Block_W));
    printf("[FF] --------- \n");
    using t_problem =
        ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                             typename Ts_::BDataType,
                                             typename Ts_::DDataType,
                                             typename Ts_::AccDataType,
                                             typename Ts_::ODataType,
                                             typename Ts_::AScaleDataType,
                                             typename Ts_::GScaleDataType,
                                             typename Ts_::DScaleDataType,
                                             typename Ts_::YSmoothScaleDataType,
                                             typename Ts_::TopkWeightDataType,
                                             typename Ts_::IndexDataType,
                                             ck_tile::element_wise::FastGeluAsm, // TODO: hardcoded
                                             t_shape,
                                             t_traits>;

    using t_pipeline    = ck_tile::GemmPipeline_FlatmmUk<t_problem>;
    using t_kernel      = ck_tile::FlatmmUkKernel<t_pipeline, void>;

    const dim3 grids                       = t_kernel::GridSize(a_);
    constexpr dim3 blocks                  = t_kernel::BlockSize();
    printf("[FF] grids  = [%d, %d, %d]\n", grids.x, grids.y, grids.z);
    printf("[FF] blocks = [%d, %d, %d]\n", blocks.x, blocks.y, blocks.z);

    static int printed = 0;

    auto kargs = t_kernel::MakeKargs(a_);
    constexpr ck_tile::index_t kBlockPerCu = 1;
    t_kernel kernel{};
    auto lambda_kenrel = ck_tile::make_kernel<blocks.x, kBlockPerCu>(kernel, grids, blocks, 0, kargs);

    if(s_.log_level_ > 0 && printed == 10)
    {
        // std::cout << ", " << t_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(s_, lambda_kenrel);

    t_traits traits;
    t_shape shape;
    t_problem problem;
    t_pipeline pipeline;
    (void)a_;
    (void)s_;
    (void)traits;
    (void)shape;
    (void)problem;
    (void)pipeline;
    (void)kernel;
    (void)lambda_kenrel;
    return 0;
}

float flatmm_uk(flatmm_uk_traits t, flatmm_uk_args a, const ck_tile::stream_config& s)
{
    // auto s_ = ck_tile::stream_config{s.stream_id_, false, s.log_level_, 0, 1};
    auto s_ = s;

    auto t_ = flatmm_uk_traits_{t.prec_i,
                                t.prec_w,
                                t.prec_o,
                                t.prec_st,
                                t.prec_sw,
                                t.prec_sq,
                                t.prec_kw,
                                t.block_m,
                                t.gate_only,
                                t.fused_quant};
    auto a_ = flatmm_uk_args_{
        a.a_ptr,     // const void* a_ptr;
        a.b_ptr,     // const void* a_ptr;
        a.c_ptr,     // void* o_ptr;
        a.sa_ptr,    // void* o_ptr;
        a.sb_ptr,    // void* o_ptr;
        a.d_ptr,     // void* o_ptr;
        a.d_f16_ptr, // void* o_ptr;
        a.dbg_int_ptr,
        a.dbg_fp8_ptr,
        a.dbg_f16_ptr,
        a.dbg_fp32_ptr,
        a.hidden_size,       // index_t hidden_size;
        a.intermediate_size, // index_t intermediate_size;
        a.num_tokens,        // index_t num_tokens;
        a.num_experts,       // index_t num_experts;
        a.topk,              // index_t topk;
        a.stride_token       // index_t stride_token;
    };

    float r = -1;

#if 0
    if(t_.prec_i == "bf16" && t_.prec_w == "bf16" && t_.prec_o == "bf16" && t_.prec_st == "fp32" &&
       t_.prec_sw == "fp32" && t_.prec_sq == "fp32" && t_.prec_kw == "fp32" && t_.block_m == 32 &&
       t_.gate_only == 1)
    {
        using t_ = fmoe_<ck_tile::bf16_t,
                         ck_tile::bf16_t,
                         ck_tile::bf16_t,
                         float,
                         float,
                         float,
                         float,
                         S<32, 512, 128, 128>,
                         S<1, 4, 1>,
                         S<16, 16, 32>,
                         1,
                         0>;
        r        = flatmm_uk_<t_>(s_, a_);
    }
    else if(t_.prec_i == "fp16" && t_.prec_w == "fp16" && t_.prec_o == "fp16" &&
            t_.prec_st == "fp32" && t_.prec_sw == "fp32" && t_.prec_sq == "fp32" &&
            t_.prec_kw == "fp32" && t_.block_m == 32 && t_.gate_only == 1)
    {
        using t_ = fmoe_<ck_tile::fp16_t,
                         ck_tile::fp16_t,
                         ck_tile::fp16_t,
                         float,
                         float,
                         float,
                         float,
                         S<32, 512, 128, 128>,
                         S<1, 4, 1>,
                         S<16, 16, 32>,
                         1,
                         0>;
        r        = flatmm_uk_<t_>(s_, a_);
    }
    else if(t_.prec_i == "fp8" && t_.prec_w == "fp8" && t_.prec_o == "bf16" &&
            t_.prec_st == "fp32" && t_.prec_sw == "fp32" && t_.prec_sq == "fp32" &&
            t_.prec_kw == "fp32" && t_.block_m == 32 && t_.gate_only == 1)
    {
        using t_ = fmoe_<ck_tile::fp8_t,
                         ck_tile::fp8_t,
                         ck_tile::bf16_t,
                         float,
                         float,
                         float,
                         float,
                         S<32, 128, 256, 128>,  // tile
                         S<1, 4, 1>,            // block
                         S<16, 16, 32>,         // mfma
                         1,
                         0>;
        r        = flatmm_uk_<t_>(s_, a_);
    }
    else 
#endif
    
    if(t_.prec_i == "fp8" && t_.prec_w == "fp8" && t_.prec_o == "fp16" &&
            t_.prec_st == "fp32" && t_.prec_sw == "fp32" && t_.prec_sq == "fp32" &&
            t_.prec_kw == "fp32" && t_.block_m == 32 && t_.gate_only == 1)
    {
        using t_ = fmoe_<ck_tile::fp8_t,
                         ck_tile::fp8_t,
                         ck_tile::fp16_t,
                         float,
                         float,
                         float,
                         float,
                         S<32, 128, 256, 128>,  // tile
                         S<1, 4, 1>,            // block
                         S<16, 16, 32>          // mfma
                         >;
        r        = flatmm_uk_<t_>(s_, a_);
    }

    // keep unsupported case return negative
    if(r < 0)
        return -1;

    return r;
}
