// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;

TEST(FmhaProblemTest, BuildsForwardProblemFromInvocation)
{
    fmha_fwd_traits traits{};
    traits.hdim_q              = 128;
    traits.hdim_v              = 128;
    traits.data_type           = "fp16";
    traits.is_group_mode       = false;
    traits.is_v_rowmajor       = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type           = mask_enum::no_mask;
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = false;
    traits.has_dropout         = false;
    traits.qscale_type         = quant_scale_enum::no_scale;

    fmha_fwd_args args{};
    args.batch        = 2;
    args.seqlen_q     = 128;
    args.seqlen_k     = 256;
    args.max_seqlen_q = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 8;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    EXPECT_TRUE(problem.is_valid());
    EXPECT_EQ(problem.api_family, FmhaApiFamily::Fwd);
    EXPECT_EQ(problem.requested_family, FmhaKernelFamily::Fwd);
    EXPECT_EQ(problem.data_type, "fp16");
    EXPECT_EQ(problem.hdim_q, 128);
    EXPECT_EQ(problem.hdim_v, 128);
    EXPECT_EQ(problem.batch, 2);
    EXPECT_EQ(problem.seqlen_q, 128);
    EXPECT_EQ(problem.seqlen_k, 256);
    EXPECT_EQ(problem.nhead_q, 16);
    EXPECT_EQ(problem.nhead_k, 8);
}

TEST(FmhaProblemTest, BuilderCreatesValidProblem)
{
    auto problem = FmhaProblemBuilder()
                       .api_family(FmhaApiFamily::Fwd)
                       .kernel_family(FmhaKernelFamily::Fwd)
                       .gfx_arch("gfx950")
                       .data_type("fp16")
                       .dims(128, 128, 2, 256, 512)
                       .nheads(16, 4)
                       .mask_type(static_cast<int>(mask_enum::mask_bottom_right))
                       .bias_type(static_cast<int>(bias_enum::elementwise_bias))
                       .lse(true)
                       .dropout(false)
                       .v_rowmajor(true)
                       .group_mode(false)
                       .window(128, 0)
                       .build();

    EXPECT_TRUE(problem.is_valid());
    EXPECT_EQ(problem.gfx_arch, "gfx950");
    EXPECT_EQ(problem.data_type, "fp16");
    EXPECT_EQ(problem.nhead_q, 16);
    EXPECT_EQ(problem.nhead_k, 4);
    EXPECT_EQ(problem.mask_type, static_cast<int>(mask_enum::mask_bottom_right));
    EXPECT_EQ(problem.bias_type, static_cast<int>(bias_enum::elementwise_bias));
    EXPECT_TRUE(problem.has_lse);
    EXPECT_EQ(problem.window_size_left, 128);
}

TEST(FmhaProblemTest, NumOpsIsNonZero)
{
    auto problem = FmhaProblemBuilder()
                       .api_family(FmhaApiFamily::Fwd)
                       .kernel_family(FmhaKernelFamily::Fwd)
                       .data_type("fp16")
                       .dims(128, 128, 2, 256, 512)
                       .nheads(16, 16)
                       .build();

    EXPECT_GT(problem.num_ops(), 0);
    // 2*batch*nhead*(sq*sk*dq + sq*sk*dv) = 2*2*16*(256*512*128 + 256*512*128)
    std::int64_t expected = 2LL * 2 * 16 * 256 * 512 * (128 + 128);
    EXPECT_EQ(problem.num_ops(), expected);
}

TEST(FmhaProblemTest, ToStringContainsKeyFields)
{
    auto problem = FmhaProblemBuilder()
                       .api_family(FmhaApiFamily::Fwd)
                       .data_type("bf16")
                       .dims(64, 64, 1, 32, 32)
                       .nheads(8, 8)
                       .gfx_arch("gfx950")
                       .build();

    auto s = problem.to_string();
    EXPECT_NE(s.find("bf16"), std::string::npos);
    EXPECT_NE(s.find("gfx950"), std::string::npos);
    EXPECT_NE(s.find("fwd"), std::string::npos);
}

TEST(FmhaProblemTest, TracksSplitKvAndPagedKvFlags)
{
    fmha_fwd_splitkv_traits traits{};
    traits.hdim_q              = 128;
    traits.hdim_v              = 128;
    traits.data_type           = "fp16";
    traits.is_group_mode       = true;
    traits.is_v_rowmajor       = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type           = mask_enum::no_mask;
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = true;
    traits.do_fp8_static_quant = false;

    fmha_fwd_splitkv_args args{};
    args.batch           = 1;
    args.seqlen_q        = 64;
    args.seqlen_k        = 1024;
    args.max_seqlen_q    = 64;
    args.hdim_q          = 128;
    args.hdim_v          = 128;
    args.nhead_q         = 16;
    args.nhead_k         = 16;
    args.num_splits      = 4;
    args.block_table_ptr = reinterpret_cast<void*>(0x1);
    args.page_block_size = 16;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    EXPECT_TRUE(problem.is_valid());
    EXPECT_EQ(problem.api_family, FmhaApiFamily::FwdSplitKv);
    EXPECT_TRUE(problem.use_paged_kv);
    EXPECT_TRUE(problem.has_block_table_ptr);
    EXPECT_EQ(problem.num_splits, 4);
    EXPECT_EQ(problem.page_size, 16);
}
