// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#include <tuple>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_RUN_LONG_TESTS)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

const ck_tile::stream_config stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

const auto AllHDims = Values(32, 64, 128, 256);

constexpr bool squant = std::is_same_v<DataTypeConfig, FmhaFwdFp8>;

class AllLong : public TestWithParam<
                    std::tuple<int,
                               bool,
                               bool,
                               mode_enum,
                               bool,
                               std::string,
                               float,
                               std::tuple<int, int, int, int, int, int, int, int, std::string>>>
{
    protected:
    void SetUp() override
    {
        if(!ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_RUN_LONG_TESTS)))
        {
            GTEST_SKIP() << "CK_TILE_FMHA_RUN_LONG_TESTS env variable is not set";
        }
    }
};

// Test cases from example/ck_tile/01_fmha/script/smoke_test_fwd.sh

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         AllLong,
                         Combine(AllHDims,
                                 Bool(),
                                 Bool(),
                                 Values(mode_enum::batch, mode_enum::group),
                                 Bool(),
                                 Values("n", "e", "a"),
                                 Values(0.0f, 0.2f),
                                 Values(std::tuple{2, 2, 1, 16, -1, 55, 256, -1, "0"},
                                        std::tuple{1, 3, -1, -1, -1, 100, 51, -1, "0"},
                                        std::tuple{2, 1, -1, 16, -1, 99, 256, -1, "1"},
                                        std::tuple{1, 2, 1, -1, -1, 1024, 256, -1, "2"},
                                        std::tuple{2, 1, -1, -1, 24, 3, 99, -1, "2"},
                                        std::tuple{3, 2, 1, -1, -1, 200, 520, -1, "t:128,30"},
                                        std::tuple{2, 1, -1, -1, -1, 99, 32, -1, "b:4,35"},
                                        std::tuple{1, 2, 1, -1, -1, 33, 0, -1, "2"},
                                        std::tuple{1, 2, 1, -1, -1, 1, 10, 32, "2"})));

TEST_P(AllLong, Test)
{
    auto [hdim, perm, is_v_rowmajor, mode, lse, bias_str, p_drop, dims_mask] = GetParam();
    auto [batch, nhead, nhead_k, hdim_q, hdim_v, seqlen_q, seqlen_k, seqlen_kpad, mask_str] =
        dims_mask;

    hdim_q = hdim_q == -1 ? hdim : hdim_q;
    hdim_v = hdim_v == -1 ? hdim : hdim_v;

    auto bias = bias_info::decode(bias_str);

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim_q,                // hdim_q
                                               hdim_v,                // hdim_v
                                               0,                     // seqlen_knew
                                               {seqlen_kpad},         // seqlen_kpads
                                               0,                     // rotary_dim
                                               perm,                  // i_perm
                                               perm,                  // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               is_v_rowmajor,         // is_v_rowmajor
                                               lse,                   // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               p_drop,                // p_drop
                                               123,                   // drop_seed
                                               1024,                  // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               5675,                  // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

class HDims
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         HDims,
                         Combine(Values(std::tuple{24, 48},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Bool(),
                                 Bool(),
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values(std::tuple{1, 4, 2, 480, -1, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, 512, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, 256, "1"},
                                        std::tuple{1, 2, -1, 900, 256, -1, "0"},
                                        std::tuple{2, 1, -1, 256, 256, -1, "1"})));

TEST_P(HDims, Test)
{
    auto [hdims, perm, is_v_rowmajor, mode, dims_mask]                      = GetParam();
    auto [hdim_q, hdim_v]                                                   = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, seqlen_kpad, mask_str] = dims_mask;

    auto bias = bias_info::decode("n");

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim_q,                // hdim_q
                                               hdim_v,                // hdim_v
                                               0,                     // seqlen_knew
                                               {seqlen_kpad},         // seqlen_kpads
                                               0,                     // rotary_dim
                                               perm,                  // i_perm
                                               perm,                  // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               is_v_rowmajor,         // is_v_rowmajor
                                               false,                 // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               0.0f,                  // p_drop
                                               0,                     // drop_seed
                                               0,                     // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               5675,                  // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

class ElementwiseBias
    : public TestWithParam<std::tuple<int,
                                      bool,
                                      mode_enum,
                                      std::string,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         ElementwiseBias,
                         Combine(AllHDims,
                                 Bool(), // layout of bias is controlled by i_perm
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values("e:0", "e:1", "e:2"),
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"})));

TEST_P(ElementwiseBias, Test)
{
    auto [hdim, i_perm, mode, bias_str, dims_mask]             = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto bias = bias_info::decode(bias_str);

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim,                  // hdim_q
                                               hdim,                  // hdim_v
                                               0,                     // seqlen_knew
                                               {-1},                  // seqlen_kpads
                                               0,                     // rotary_dim
                                               i_perm,                // i_perm
                                               false,                 // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               false,                 // is_v_rowmajor
                                               false,                 // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               0.0f,                  // p_drop
                                               0,                     // drop_seed
                                               0,                     // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               38963,                 // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

class Alibi
    : public TestWithParam<
          std::tuple<int, mode_enum, std::string, std::tuple<int, int, int, int, int>, std::string>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Alibi,
                         Combine(AllHDims,
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values("a:0", "a:1"),
                                 Values(std::tuple{2, 4, 4, 1024, 100},
                                        std::tuple{3, 5, 5, 128, 256},
                                        std::tuple{2, 8, 2, 300, 355}),
                                 Values("0", "t", "b", "t:50,64", "b:32,40")));

TEST_P(Alibi, Test)
{
    auto [hdim, mode, bias_str, dims, mask_str]      = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;

    auto bias = bias_info::decode(bias_str);

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim,                  // hdim_q
                                               hdim,                  // hdim_v
                                               0,                     // seqlen_knew
                                               {-1},                  // seqlen_kpads
                                               0,                     // rotary_dim
                                               true,                  // i_perm
                                               true,                  // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               false,                 // is_v_rowmajor
                                               false,                 // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               0.0f,                  // p_drop
                                               0,                     // drop_seed
                                               0,                     // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               38963,                 // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

class Dropout : public TestWithParam<std::tuple<int,
                                                mode_enum,
                                                float,
                                                std::tuple<uint64_t, uint64_t, bool>,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Dropout,
                         Combine(AllHDims,
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values(0.123f, 0.5f),
                                 Values(std::tuple{10, 123, false},
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 4, 2, 280, 512, std::string("0")},
                                        std::tuple{3, 2, 2, 256, 128, std::string("1")},
                                        std::tuple{4, 3, 1, 100, 768, std::string("2")})));

TEST_P(Dropout, Test)
{
    auto [hdim, mode, p_drop, drop_seed_offset_prefs, dims_mask] = GetParam();
    auto [drop_seed, drop_offset, drop_prefs]                    = drop_seed_offset_prefs;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]   = dims_mask;

    auto bias = bias_info::decode("n");

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim,                  // hdim_q
                                               hdim,                  // hdim_v
                                               0,                     // seqlen_knew
                                               {-1},                  // seqlen_kpads
                                               0,                     // rotary_dim
                                               false,                 // i_perm
                                               false,                 // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               false,                 // is_v_rowmajor
                                               false,                 // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               p_drop,                // p_drop
                                               drop_seed,             // drop_seed
                                               drop_offset,           // drop_offset
                                               drop_prefs,            // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               38963,                 // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

#if CK_TILE_FMHA_FWD_PAGEDKV_API

class PagedKV : public TestWithParam<std::tuple<int,
                                                bool,
                                                bool,
                                                mode_enum,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         PagedKV,
                         Combine(AllHDims,
                                 Bool(), // layouts of k and v are controlled by i_perm
                                 Bool(), // layout of v is controlled by is_v_rowmajor
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values(128, 256),
                                 Values(std::tuple{2, 3, 1, 200, 1024, "0"},
                                        std::tuple{3, 2, -1, 128, 768, "2"},
                                        std::tuple{2, 2, -1, 230, 899, "t:50,64"})));

TEST_P(PagedKV, Test)
{
    auto [hdim, i_perm, is_v_rowmajor, mode, page_block_size, dims_mask] = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]           = dims_mask;

    auto bias = bias_info::decode("n");

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim,                  // hdim_q
                                               hdim,                  // hdim_v
                                               0,                     // seqlen_knew
                                               {-1},                  // seqlen_kpads
                                               0,                     // rotary_dim
                                               i_perm,                // i_perm
                                               false,                 // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               is_v_rowmajor,         // is_v_rowmajor
                                               false,                 // lse
                                               page_block_size,       // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               0.0f,                  // p_drop
                                               0,                     // drop_seed
                                               0,                     // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               38963,                 // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_PAGEDKV_API

#if CK_TILE_FMHA_FWD_SPLITKV_API

class SplitKV : public TestWithParam<std::tuple<int,
                                                bool,
                                                bool,
                                                std::tuple<mode_enum, bool>,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKV,
                         Combine(AllHDims,
                                 Bool(), // layouts of k and v are controlled by i_perm
                                 Bool(), // layout of v is controlled by is_v_rowmajor
                                 Values(std::tuple{mode_enum::batch, false},
                                        std::tuple{mode_enum::batch, true},
                                        std::tuple{mode_enum::group, false}),
                                 Values(3, 4),
                                 Values(std::tuple{4, 3, 1, 200, 1024, "0"},
                                        std::tuple{2, 2, -1, 512, 2000, "0"},
                                        std::tuple{3, 2, -1, 230, 899, "t:128,128"})));

TEST_P(SplitKV, Test)
{
    auto [hdim, i_perm, is_v_rowmajor, mode_use_cache_batch_idx, num_splits, dims_mask] =
        GetParam();
    auto [mode, use_cache_batch_idx]                           = mode_use_cache_batch_idx;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto bias = bias_info::decode("e");

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim,                  // hdim_q
                                               hdim,                  // hdim_v
                                               0,                     // seqlen_knew
                                               {-1},                  // seqlen_kpads
                                               0,                     // rotary_dim
                                               i_perm,                // i_perm
                                               false,                 // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               is_v_rowmajor,         // is_v_rowmajor
                                               true,                  // lse
                                               0,                     // page_block_size
                                               use_cache_batch_idx,   // use_cache_batch_idx
                                               bias,                  // bias
                                               0.0f,                  // p_drop
                                               0,                     // drop_seed
                                               0,                     // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               num_splits,            // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               42534,                 // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_SPLITKV_API
