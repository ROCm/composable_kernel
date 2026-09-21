// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;

namespace {

class MockFmhaKernel : public FmhaKernelInstance
{
    public:
    MockFmhaKernel(FmhaKernelKey key, std::string name)
        : key_(std::move(key)), name_(std::move(name))
    {
    }

    const FmhaKernelKey& get_key() const override { return key_; }

    bool supports(const FmhaProblem& problem) const override
    {
        return key_.signature.family == problem.requested_family &&
               key_.signature.data_type == problem.data_type &&
               problem.hdim_q <= key_.signature.hdim_q && problem.hdim_v <= key_.signature.hdim_v;
    }

    std::string get_name() const override { return name_; }

    void launch(const FmhaInvocation&, const ck_tile::stream_config&) const override {}

    private:
    FmhaKernelKey key_;
    std::string name_;
};

FmhaKernelKey make_key(FmhaKernelFamily family, const std::string& name, int rank = 0)
{
    (void)name;
    FmhaKernelKey key;
    key.signature.family         = family;
    key.signature.data_type      = "fp16";
    key.signature.is_group_mode  = false;
    key.signature.is_v_rowmajor  = true;
    key.signature.hdim_q         = 128;
    key.signature.hdim_v         = 128;
    key.algorithm.selection_rank = rank;
    key.algorithm.tile_shape     = {128, 128, 32, 128, 32, 128};
    key.algorithm.pad_s          = true;
    key.algorithm.pad_sk         = true;
    key.algorithm.pad_d          = true;
    key.algorithm.pad_dv         = true;
    return key;
}

FmhaProblem make_splitkv_problem()
{
    fmha_fwd_splitkv_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;
    traits.has_lse       = true;

    fmha_fwd_splitkv_args args{};
    args.batch        = 1;
    args.seqlen_q     = 128;
    args.seqlen_k     = 1024;
    args.max_seqlen_q = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 16;
    args.num_splits   = 8;

    return FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
}

FmhaProblem make_bwd_problem()
{
    fmha_bwd_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;

    fmha_bwd_args args{};
    args.batch        = 1;
    args.seqlen_q     = 128;
    args.seqlen_k     = 128;
    args.max_seqlen_q = 128;
    args.max_seqlen_k = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 16;

    return FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
}

} // namespace

TEST(FmhaDispatcherTest, PlansSplitKvAsTwoStages)
{
    FmhaRegistry registry;
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(make_key(FmhaKernelFamily::FwdSplitKv, "split"), "split"));
    registry.register_kernel(std::make_shared<MockFmhaKernel>(
        make_key(FmhaKernelFamily::FwdSplitKvCombine, "combine"), "combine"));

    FmhaDispatcher dispatcher(&registry);
    auto plan = dispatcher.plan(make_splitkv_problem());
    ASSERT_TRUE(plan.is_valid());
    ASSERT_EQ(plan.stages.size(), 2u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::FwdSplitKv);
    EXPECT_EQ(plan.stages[1].family, FmhaKernelFamily::FwdSplitKvCombine);
}

TEST(FmhaDispatcherTest, PlansSingleStageFwd)
{
    FmhaRegistry registry;
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(make_key(FmhaKernelFamily::Fwd, "fwd"), "fwd"));

    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;
    traits.has_lse       = false;
    traits.has_dropout   = false;
    traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args args{};
    args.batch        = 1;
    args.seqlen_q     = 128;
    args.seqlen_k     = 128;
    args.max_seqlen_q = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 16;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    auto plan    = dispatcher.plan(problem);
    ASSERT_TRUE(plan.is_valid());
    ASSERT_EQ(plan.stages.size(), 1u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::Fwd);
}

TEST(FmhaDispatcherTest, PlansSingleStagePagedKv)
{
    FmhaRegistry registry;
    registry.register_kernel(std::make_shared<MockFmhaKernel>(
        make_key(FmhaKernelFamily::FwdPagedKv, "pagedkv"), "pagedkv"));

    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_pagedkv_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;

    fmha_fwd_pagedkv_args args{};
    args.batch        = 1;
    args.seqlen_q     = 128;
    args.seqlen_k     = 128;
    args.max_seqlen_q = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 16;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    auto plan    = dispatcher.plan(problem);
    ASSERT_TRUE(plan.is_valid());
    ASSERT_EQ(plan.stages.size(), 1u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::FwdPagedKv);
}

TEST(FmhaDispatcherTest, PlansSingleStageAppendKv)
{
    FmhaRegistry registry;
    auto key = make_key(FmhaKernelFamily::FwdAppendKv, "appendkv");
    registry.register_kernel(std::make_shared<MockFmhaKernel>(key, "appendkv"));

    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_appendkv_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_v_rowmajor = true;
    traits.rope_type     = rope_enum::none;

    fmha_fwd_appendkv_args args{};
    args.batch       = 1;
    args.seqlen_q    = 128;
    args.seqlen_knew = 64;
    args.hdim_q      = 128;
    args.hdim_v      = 128;
    args.nhead_q     = 16;
    args.nhead_k     = 16;

    auto problem = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    auto plan    = dispatcher.plan(problem);
    ASSERT_TRUE(plan.is_valid());
    ASSERT_EQ(plan.stages.size(), 1u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::FwdAppendKv);
}

TEST(FmhaDispatcherTest, SeqtunePrefersSmallerAlignedTile)
{
    FmhaRegistry registry;

    auto key_big                      = make_key(FmhaKernelFamily::Fwd, "big", /*rank=*/0);
    key_big.algorithm.tile_shape.m0   = 128;
    key_big.algorithm.pad_s           = false;
    auto key_small                    = make_key(FmhaKernelFamily::Fwd, "small", /*rank=*/0);
    key_small.algorithm.tile_shape.m0 = 64;
    key_small.algorithm.pad_s         = false;

    registry.register_kernel(std::make_shared<MockFmhaKernel>(key_big, "big"));
    registry.register_kernel(std::make_shared<MockFmhaKernel>(key_small, "small"));

    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_traits traits{};
    traits.hdim_q        = 128;
    traits.hdim_v        = 128;
    traits.data_type     = "fp16";
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;

    fmha_fwd_args args{};
    args.batch        = 1;
    args.seqlen_q     = 128;
    args.seqlen_k     = 128;
    args.max_seqlen_q = 128;
    args.hdim_q       = 128;
    args.hdim_v       = 128;
    args.nhead_q      = 16;
    args.nhead_k      = 16;

    auto problem  = FmhaProblem::from_invocation(FmhaInvocation::make(traits, args), "gfx942");
    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    // Both tiles align to 128; seqtune prefers the smaller tile_m0
    EXPECT_EQ(selected->get_name(), "small");
}

TEST(FmhaDispatcherTest, PlansBackwardAsThreeStagesWhenConvertExists)
{
    FmhaRegistry registry;
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(make_key(FmhaKernelFamily::BwdDotDoO, "dot"), "dot"));
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(make_key(FmhaKernelFamily::BwdDqDkDv, "dq"), "dq"));
    registry.register_kernel(std::make_shared<MockFmhaKernel>(
        make_key(FmhaKernelFamily::BwdConvertDq, "convert"), "convert"));

    FmhaDispatcher dispatcher(&registry);
    auto plan = dispatcher.plan(make_bwd_problem());
    ASSERT_TRUE(plan.is_valid());
    ASSERT_EQ(plan.stages.size(), 3u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::BwdDotDoO);
    EXPECT_EQ(plan.stages[1].family, FmhaKernelFamily::BwdDqDkDv);
    EXPECT_EQ(plan.stages[2].family, FmhaKernelFamily::BwdConvertDq);
}

// #15: BWD with asymmetric head dimensions (hdim_q != hdim_v)
TEST(FmhaDispatcherTest, PlansBackwardWithAsymmetricHdim)
{
    FmhaRegistry registry;
    registry.set_name("test_bwd_asym");

    auto asym_key = [](FmhaKernelFamily family, const std::string& n) {
        auto key             = make_key(family, n);
        key.signature.hdim_q = 96;
        key.signature.hdim_v = 128;
        return key;
    };

    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(asym_key(FmhaKernelFamily::BwdDotDoO, "dot96"), "dot96"));
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(asym_key(FmhaKernelFamily::BwdDqDkDv, "dq96"), "dq96"));

    FmhaDispatcher dispatcher(&registry);
    auto problem   = make_bwd_problem();
    problem.hdim_q = 96;
    problem.hdim_v = 128;
    auto plan      = dispatcher.plan(problem);
    ASSERT_TRUE(plan.is_valid());
    EXPECT_GE(plan.stages.size(), 2u);
    EXPECT_EQ(plan.stages[0].family, FmhaKernelFamily::BwdDotDoO);
    EXPECT_EQ(plan.stages[1].family, FmhaKernelFamily::BwdDqDkDv);
}

// #16: BWD negative test -- no matching kernel returns invalid plan
TEST(FmhaDispatcherTest, PlansBackwardReturnsInvalidWhenNoKernel)
{
    FmhaRegistry registry;
    registry.set_name("test_bwd_neg");

    // Register only a fwd kernel, no bwd kernels
    registry.register_kernel(
        std::make_shared<MockFmhaKernel>(make_key(FmhaKernelFamily::Fwd, "fwd"), "fwd"));

    FmhaDispatcher dispatcher(&registry);
    auto plan = dispatcher.plan(make_bwd_problem());
    EXPECT_FALSE(plan.is_valid());
}

// #17: Canonical key distinguishes dropout seed differences
TEST(FmhaDispatcherTest, CanonicalKeyDistinguishesDropout)
{
    FmhaProblem p1;
    p1.data_type   = "fp16";
    p1.hdim_q      = 128;
    p1.hdim_v      = 128;
    p1.has_dropout = false;

    FmhaProblem p2 = p1;
    p2.has_dropout = true;

    EXPECT_NE(p1.canonical_key(), p2.canonical_key());
}

// Canonical key covers all signature fields
TEST(FmhaDispatcherTest, CanonicalKeyCoversAllFields)
{
    FmhaProblem base;
    base.data_type = "fp16";
    base.hdim_q    = 128;
    base.hdim_v    = 128;

    auto check_differs = [&](auto mutator) {
        FmhaProblem p = base;
        mutator(p);
        EXPECT_NE(base.canonical_key(), p.canonical_key());
    };

    check_differs([](FmhaProblem& p) { p.has_lse = true; });
    check_differs([](FmhaProblem& p) { p.has_dropout = true; });
    check_differs([](FmhaProblem& p) { p.has_logits_soft_cap = true; });
    check_differs([](FmhaProblem& p) { p.has_sink = true; });
    check_differs([](FmhaProblem& p) { p.is_deterministic = true; });
    check_differs([](FmhaProblem& p) { p.has_dbias = true; });
    check_differs([](FmhaProblem& p) { p.is_store_randval = true; });
    check_differs([](FmhaProblem& p) { p.mask_type = 1; });
    check_differs([](FmhaProblem& p) { p.bias_type = 2; });
    check_differs([](FmhaProblem& p) { p.is_group_mode = true; });
}

// BWD workspace sizing
TEST(FmhaDispatcherTest, BwdWorkspaceInfoComputation)
{
    FmhaProblem p;
    p.batch    = 2;
    p.nhead_q  = 8;
    p.seqlen_q = 256;
    p.seqlen_k = 256;
    p.hdim_q   = 128;

    auto info = bwd_workspace_info(p);
    EXPECT_EQ(info.d_bytes, 2u * 8 * 256 * sizeof(float));
    EXPECT_EQ(info.dq_acc_bytes, 2u * 8 * 256 * 128 * sizeof(float));
    EXPECT_EQ(info.d_offset, 0u);
    EXPECT_GT(info.dq_acc_offset, 0u);
    EXPECT_GE(info.dq_acc_offset, info.d_bytes);
    EXPECT_EQ(info.dq_acc_offset % 256, 0u);
    EXPECT_GT(info.total_bytes, info.dq_acc_offset + info.dq_acc_bytes - 1);
}

// Benchmarking control
TEST(FmhaDispatcherTest, SetBenchmarkingControlsTimingFlag)
{
    FmhaRegistry registry;
    FmhaDispatcher dispatcher(&registry);

    EXPECT_FALSE(dispatcher.benchmarking_enabled());
    dispatcher.set_benchmarking(true);
    EXPECT_TRUE(dispatcher.benchmarking_enabled());
    dispatcher.set_benchmarking(false);
    EXPECT_FALSE(dispatcher.benchmarking_enabled());
}

// Verify tie() covers all Signature and Algorithm fields.
// If a new field is added to FmhaKernelKey but not to tie(),
// two keys differing only in that field would compare equal (silent bug).
TEST(FmhaKernelKeyTest, TieCoversAllSignatureFields)
{
    FmhaKernelKey a{};
    a.signature.data_type = "fp16";
    a.gfx_arch            = "gfx950";

    auto flip = [&](auto mutator) {
        FmhaKernelKey b = a;
        mutator(b);
        EXPECT_NE(a, b) << "tie() does not distinguish a Signature/Algorithm field";
    };

    flip([](FmhaKernelKey& k) { k.signature.family = FmhaKernelFamily::BwdDqDkDv; });
    flip([](FmhaKernelKey& k) { k.signature.data_type = "bf16"; });
    flip([](FmhaKernelKey& k) { k.signature.is_group_mode = true; });
    flip([](FmhaKernelKey& k) { k.signature.is_v_rowmajor = false; });
    flip([](FmhaKernelKey& k) { k.signature.has_logits_soft_cap = true; });
    flip([](FmhaKernelKey& k) { k.signature.mask_type = 1; });
    flip([](FmhaKernelKey& k) { k.signature.bias_type = 1; });
    flip([](FmhaKernelKey& k) { k.signature.has_lse = true; });
    flip([](FmhaKernelKey& k) { k.signature.has_dropout = true; });
    flip([](FmhaKernelKey& k) { k.signature.qscale_type = 1; });
    flip([](FmhaKernelKey& k) { k.signature.rope_type = 1; });
    flip([](FmhaKernelKey& k) { k.signature.use_paged_kv = true; });
    flip([](FmhaKernelKey& k) { k.signature.do_fp8_static_quant = true; });
    flip([](FmhaKernelKey& k) { k.signature.skip_min_seqlen_q = true; });
    flip([](FmhaKernelKey& k) { k.signature.has_sink = true; });
    flip([](FmhaKernelKey& k) { k.signature.has_dbias = true; });
    flip([](FmhaKernelKey& k) { k.signature.is_store_randval = true; });
    flip([](FmhaKernelKey& k) { k.signature.is_deterministic = true; });
    flip([](FmhaKernelKey& k) { k.signature.kv_memory_layout = 1; });
    flip([](FmhaKernelKey& k) { k.signature.kv_lookup_table = 1; });
    flip([](FmhaKernelKey& k) { k.signature.page_size = 64; });
    flip([](FmhaKernelKey& k) { k.signature.hdim_q = 256; });
    flip([](FmhaKernelKey& k) { k.signature.hdim_v = 256; });
    flip([](FmhaKernelKey& k) { k.signature.receipt = 1; });

    flip([](FmhaKernelKey& k) { k.algorithm.tile_shape.m0 = 64; });
    flip([](FmhaKernelKey& k) { k.algorithm.pipeline = "qr_async"; });
    flip([](FmhaKernelKey& k) { k.algorithm.pad_s = false; });
    flip([](FmhaKernelKey& k) { k.algorithm.selection_rank = 5; });
    flip([](FmhaKernelKey& k) { k.algorithm.constraint_tag = "special"; });
    flip([](FmhaKernelKey& k) { k.gfx_arch = "gfx942"; });
}

TEST(FmhaDispatcherTest, SelectKernelReturnsNullptrOnEmptyRegistry)
{
    FmhaRegistry registry;
    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_traits traits{};
    traits.hdim_q    = 128;
    traits.hdim_v    = 128;
    traits.data_type = "fp16";
    traits.mask_type = mask_enum::no_mask;
    traits.bias_type = bias_enum::no_bias;

    auto problem =
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_fwd_args{}), "gfx950");
    auto selected = dispatcher.select_kernel(problem);
    EXPECT_EQ(selected, nullptr);
}

TEST(FmhaDispatcherTest, SelectKernelReturnsNullptrOnNoMatch)
{
    FmhaRegistry registry;
    auto key  = make_fwd_key(128, 128, "fp16", "gfx950");
    auto mock = std::make_shared<MockFmhaKernel>(key, "fp16_h128");
    registry.register_kernel(mock);

    FmhaDispatcher dispatcher(&registry);

    fmha_fwd_traits traits{};
    traits.hdim_q    = 256;
    traits.hdim_v    = 256;
    traits.data_type = "bf16";
    traits.mask_type = mask_enum::no_mask;
    traits.bias_type = bias_enum::no_bias;

    auto problem =
        FmhaProblem::from_invocation(FmhaInvocation::make(traits, fmha_fwd_args{}), "gfx950");
    auto selected = dispatcher.select_kernel(problem);
    EXPECT_EQ(selected, nullptr);
}
