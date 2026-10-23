// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/backends/generated_fmha_backend.hpp"

using namespace ck_tile::dispatcher;

namespace {

class StubFmhaKernel : public FmhaKernelInstance
{
    public:
    StubFmhaKernel(FmhaKernelKey key, std::string name)
        : key_(std::move(key)), name_(std::move(name))
    {
    }

    const FmhaKernelKey& get_key() const override { return key_; }
    bool supports(const FmhaProblem& problem) const override
    {
        return key_.signature.family == problem.requested_family &&
               key_.signature.data_type == problem.data_type;
    }
    std::string get_name() const override { return name_; }
    void launch(const FmhaInvocation&, const ck_tile::stream_config&) const override {}

    private:
    FmhaKernelKey key_;
    std::string name_;
};

FmhaKernelKey
make_stub_key(FmhaKernelFamily family, const std::string& dtype, const std::string& arch)
{
    FmhaKernelKey key;
    key.signature.family     = family;
    key.signature.data_type  = dtype;
    key.signature.hdim_q     = 128;
    key.signature.hdim_v     = 128;
    key.gfx_arch             = arch;
    key.algorithm.tile_shape = {128, 128, 32, 128, 32, 128};
    key.algorithm.pad_s      = true;
    key.algorithm.pad_sk     = true;
    key.algorithm.pad_d      = true;
    key.algorithm.pad_dv     = true;
    return key;
}

FmhaKernelKey make_batch_prefill_key(int page_size)
{
    auto key                    = make_stub_key(FmhaKernelFamily::BatchPrefill, "fp16", "gfx1100");
    key.signature.is_group_mode = true;
    key.signature.has_lse       = true;
    key.signature.use_paged_kv  = true;
    key.signature.kv_memory_layout =
        static_cast<int>(ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT);
    key.signature.kv_lookup_table =
        static_cast<int>(ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D);
    key.signature.page_size  = page_size;
    key.algorithm.pipeline   = "batch_prefill_gfx11";
    key.algorithm.tile_shape = {128, 32, 32, 128, 32, 128};
    return key;
}

FmhaProblem make_batch_prefill_problem(int page_size)
{
    FmhaProblem problem;
    problem.api_family       = FmhaApiFamily::BatchPrefill;
    problem.requested_family = FmhaKernelFamily::BatchPrefill;
    problem.gfx_arch         = "gfx1100";
    problem.data_type        = "fp16";
    problem.is_group_mode    = true;
    problem.is_v_rowmajor    = true;
    problem.has_lse          = true;
    problem.use_paged_kv     = true;
    problem.kv_memory_layout =
        static_cast<int>(ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT);
    problem.kv_lookup_table =
        static_cast<int>(ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D);
    problem.page_size    = page_size;
    problem.seqlen_q     = 128;
    problem.seqlen_k     = 1024;
    problem.max_seqlen_q = 128;
    problem.max_seqlen_k = 1024;
    problem.batch        = 1;
    problem.hdim_q       = 128;
    problem.hdim_v       = 128;
    problem.nhead_q      = 8;
    problem.nhead_k      = 8;
    return problem;
}

} // namespace

TEST(FmhaRegistryTest, RegisterAndLookup)
{
    FmhaRegistry reg;
    auto key    = make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950");
    auto kernel = std::make_shared<StubFmhaKernel>(key, "test_fwd_fp16");
    EXPECT_TRUE(reg.register_kernel(kernel));
    EXPECT_EQ(reg.size(), 1u);
    auto found = reg.lookup(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "test_fwd_fp16");
}

TEST(FmhaRegistryTest, GetAllReturnsSorted)
{
    FmhaRegistry reg;
    auto key_a                     = make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950");
    key_a.algorithm.selection_rank = 1;
    auto key_b                     = make_stub_key(FmhaKernelFamily::BwdDqDkDv, "fp16", "gfx950");
    key_b.algorithm.selection_rank = 0;

    reg.register_kernel(std::make_shared<StubFmhaKernel>(key_a, "rank1"));
    reg.register_kernel(std::make_shared<StubFmhaKernel>(key_b, "rank0"));

    auto all = reg.get_all();
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(all[0]->get_name(), "rank0");
    EXPECT_EQ(all[1]->get_name(), "rank1");
}

TEST(FmhaRegistryTest, DispatcherSelectsExactBatchPrefillPageSize)
{
    FmhaRegistry registry;
    const auto register_kernel = [&](int page_size, const std::string& name) {
        auto key      = make_batch_prefill_key(page_size);
        auto supports = backends::make_default_supports_fn(key);
        return registry.register_kernel(std::make_shared<backends::GeneratedFmhaKernelInstance>(
            std::move(key),
            name,
            std::move(supports),
            backends::GeneratedFmhaKernelInstance::LaunchFn{}));
    };

    ASSERT_TRUE(register_kernel(1, "a_page1"));
    ASSERT_TRUE(register_kernel(16, "b_page16"));
    ASSERT_TRUE(register_kernel(1024, "c_page1024"));

    FmhaDispatcher dispatcher(&registry);
    for(const int page_size : {1, 16, 1024})
    {
        SCOPED_TRACE(page_size);
        const auto plan = dispatcher.plan(make_batch_prefill_problem(page_size));
        ASSERT_TRUE(plan.is_valid());
        ASSERT_EQ(plan.stages.size(), 1u);
        EXPECT_EQ(plan.stages[0].kernel_id, make_batch_prefill_key(page_size).encode_identifier());
    }
}

TEST(FmhaRegistryTest, FilterByArch)
{
    FmhaRegistry reg;
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950"), "k950"));
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx942"), "k942"));
    EXPECT_EQ(reg.size(), 2u);

    auto removed = reg.filter_by_arch("gfx950");
    EXPECT_EQ(removed, 1u);
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_NE(reg.lookup(make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950")), nullptr);
}

TEST(FmhaRegistryTest, FilterByPredicate)
{
    FmhaRegistry reg;
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950"), "fwd_fp16"));
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::Fwd, "bf16", "gfx950"), "fwd_bf16"));
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::BwdDqDkDv, "fp16", "gfx950"), "bwd_fp16"));

    auto fwd_only = reg.filter([](const FmhaKernelInstance& k) {
        return k.get_key().signature.family == FmhaKernelFamily::Fwd;
    });
    EXPECT_EQ(fwd_only.size(), 2u);
}

TEST(FmhaRegistryTest, ExportJsonContainsMetadata)
{
    FmhaRegistry reg;
    reg.set_name("test_registry");
    reg.register_kernel(std::make_shared<StubFmhaKernel>(
        make_stub_key(FmhaKernelFamily::Fwd, "fp16", "gfx950"), "fwd_fp16"));

    auto json = reg.export_json();
    EXPECT_NE(json.find("test_registry"), std::string::npos);
    EXPECT_NE(json.find("total_kernels"), std::string::npos);
    EXPECT_NE(json.find("fwd_fp16"), std::string::npos);
}
