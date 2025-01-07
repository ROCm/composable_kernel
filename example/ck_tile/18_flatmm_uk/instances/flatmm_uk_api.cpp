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
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly, Ts_::FusedQuant == 1, 1 /*atomic*/>;
    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                                typename Ts_::WarpPerBlock_0,
                                                typename Ts_::WarpTile_0,
                                                typename Ts_::BlockTile_1,
                                                typename Ts_::WarpPerBlock_0,
                                                typename Ts_::WarpTile_0>;
    printf("[FF] --- fused_moegemm_(): <FusedMoeGemmShape> --- \n");
    printf("[FF] f_shape::BlockSize = %d\n", static_cast<uint32_t>(f_shape::BlockSize));
    printf("[FF] f_shape::NumWarps = %d\n", static_cast<uint32_t>(f_shape::NumWarps));
    printf("[FF] --------- \n");
    printf("[FF] f_shape::Block_M0 = %d\n", static_cast<uint32_t>(f_shape::Block_M0));
    printf("[FF] f_shape::Block_N0 = %d\n", static_cast<uint32_t>(f_shape::Block_N0));
    printf("[FF] f_shape::Block_K0 = %d\n", static_cast<uint32_t>(f_shape::Block_K0));
    printf("[FF] f_shape::WarpPerBlock_M0 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_M0));
    printf("[FF] f_shape::WarpPerBlock_N0 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_N0));
    printf("[FF] f_shape::WarpPerBlock_K0 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_K0));
    printf("[FF] f_shape::Warp_M0 = %d\n", static_cast<uint32_t>(f_shape::Warp_M0));
    printf("[FF] f_shape::Warp_N0 = %d\n", static_cast<uint32_t>(f_shape::Warp_N0));
    printf("[FF] f_shape::Warp_K0 = %d\n", static_cast<uint32_t>(f_shape::Warp_K0));
    printf("[FF] f_shape::ThreadPerBlock_M0 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_M0));
    printf("[FF] f_shape::ThreadPerBlock_N0 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_N0));
    printf("[FF] f_shape::ThreadPerBlock_K0 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_K0));
    printf("[FF] f_shape::Repeat_M0 = %d\n", static_cast<uint32_t>(f_shape::Repeat_M0));
    printf("[FF] f_shape::Repeat_N0 = %d\n", static_cast<uint32_t>(f_shape::Repeat_N0));
    printf("[FF] f_shape::Repeat_K0 = %d\n", static_cast<uint32_t>(f_shape::Repeat_K0));
    printf("[FF] f_shape::Block_W0  = %d\n", static_cast<uint32_t>(f_shape::Block_W0));
    printf("[FF] f_shape::Block_Nr0 = %d\n", static_cast<uint32_t>(f_shape::Block_Nr0));
    printf("[FF] f_shape::Block_Kr0 = %d\n", static_cast<uint32_t>(f_shape::Block_Kr0));
    printf("[FF] --------- \n");
    printf("[FF] f_shape::Block_M1 = %d\n", static_cast<uint32_t>(f_shape::Block_M1));
    printf("[FF] f_shape::Block_N1 = %d\n", static_cast<uint32_t>(f_shape::Block_N1));
    printf("[FF] f_shape::Block_K1 = %d\n", static_cast<uint32_t>(f_shape::Block_K1));
    printf("[FF] f_shape::WarpPerBlock_M1 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_M1));
    printf("[FF] f_shape::WarpPerBlock_N1 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_N1));
    printf("[FF] f_shape::WarpPerBlock_K1 = %d\n", static_cast<uint32_t>(f_shape::WarpPerBlock_K1));
    printf("[FF] f_shape::Warp_M1 = %d\n", static_cast<uint32_t>(f_shape::Warp_M1));
    printf("[FF] f_shape::Warp_N1 = %d\n", static_cast<uint32_t>(f_shape::Warp_N1));
    printf("[FF] f_shape::Warp_K1 = %d\n", static_cast<uint32_t>(f_shape::Warp_K1));
    printf("[FF] f_shape::ThreadPerBlock_M1 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_M1));
    printf("[FF] f_shape::ThreadPerBlock_N1 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_N1));
    printf("[FF] f_shape::ThreadPerBlock_K1 = %d\n",
           static_cast<uint32_t>(f_shape::ThreadPerBlock_K1));
    printf("[FF] f_shape::Repeat_M1 = %d\n", static_cast<uint32_t>(f_shape::Repeat_M1));
    printf("[FF] f_shape::Repeat_N1 = %d\n", static_cast<uint32_t>(f_shape::Repeat_N1));
    printf("[FF] f_shape::Repeat_K1 = %d\n", static_cast<uint32_t>(f_shape::Repeat_K1));
    printf("[FF] f_shape::Block_W1  = %d\n", static_cast<uint32_t>(f_shape::Block_W1));
    printf("[FF] f_shape::Block_Nr1 = %d\n", static_cast<uint32_t>(f_shape::Block_Nr1));
    printf("[FF] f_shape::Block_Kr1 = %d\n", static_cast<uint32_t>(f_shape::Block_Kr1));
    using f_problem =
        ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                             typename Ts_::GDataType,
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
                                             f_shape,
                                             f_traits>;

    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmEx<f_problem>;
    using f_pipeline    = ck_tile::GemmPipeline_FlatmmUk<f_problem>;
    using f_kernel      = ck_tile::FlatmmUkKernel<f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a_);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;
    printf("[FF] grids = [%d, %d, %d]\n", grids.x, grids.y, grids.z);
    printf("[FF] blocks = [%d, %d, %d]\n", blocks.x, blocks.y, blocks.z);

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a_);
    f_kernel kernel{};
    auto lambda_kenrel =
        ck_tile::make_kernel<blocks.x, kBlockPerCu>(kernel, grids, blocks, 0, kargs);

    if(s_.log_level_ > 0 && printed == 10)
    {
        // std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s_, lambda_kenrel
        // ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs)
    );
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
        a.a_ptr, // const void* a_ptr;
        a.b_ptr, // const void* a_ptr;
        a.c_ptr, // void* o_ptr;
        a.d_ptr, // void* o_ptr;
        a.dbg_int_ptr,
        a.dbg_bf16_ptr,
        a.dbg_fp32_ptr,
        a.hidden_size,       // index_t hidden_size;
        a.intermediate_size, // index_t intermediate_size;
        a.num_tokens,        // index_t num_tokens;
        a.num_experts,       // index_t num_experts;
        a.topk,              // index_t topk;
        a.stride_token       // index_t stride_token;
    };

    float r = -1;

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

    // keep unsupported case return negative
    if(r < 0)
        return -1;

    return r;
}
